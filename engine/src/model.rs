//! `.bin` parsing → `WeightLayout` precomputed once via `OnceLock`.
//!
//! The model bytes are embedded into the WASM at compile time and parsed at
//! init. After parsing, all weight accesses are byte-offset lookups into the
//! immutable MODEL_BYTES slice — no copies, no allocations per query.
//!
//! Layout walk mirrors the section order documented in
//! `docs/tern-inference-engine.md` and produced by `training/pack/pack.py`.

use std::sync::OnceLock;

use crate::format::{
    self, Header,
    EMB_FP32, EMB_INT8, EMB_TERNARY, EMB_INT4,
    HEADER_SIZE, SHA256_SIZE,
};

/// The packed `.bin`, embedded at compile time. NOT committed to the repo —
/// the build pipeline writes it into `engine/assets/model.bin` before the WASM
/// build, per the format the build target's Cargo feature requires.
///
/// During scaffolding, this is empty — guard accesses with `is_empty()` and
/// don't try to parse. The first real build will populate this and the
/// `MODEL` OnceLock will succeed.
static MODEL_BYTES: &[u8] = include_bytes!("../assets/model.bin");

static MODEL: OnceLock<LoadedModel> = OnceLock::new();

/// Byte offsets for one packed-ternary BitLinear matrix.
///
/// Storage shape (sequential): `[packed_weights | w_scale (f32) | bias (f32) if present]`.
/// Weights are 2-bit packed at 4 weights per byte, row-major over (out_features, in_features).
#[derive(Debug, Clone, Copy)]
pub struct BitLinearLayout {
    pub weights_offset: usize,
    pub weights_bytes:  usize,   // out_features × (in_features / 4)
    pub scale_offset:   usize,
    pub bias_offset:    Option<usize>,
    pub in_features:    usize,
    pub out_features:   usize,
}

/// Byte offsets for one transformer layer.
///
/// LayerNorm sections are `[weight (d_model fp32) | bias (d_model fp32)]` —
/// 8 × d_model bytes each. Q/K/V have NO bias; W_out/fc1/fc2 do.
#[derive(Debug, Clone, Copy)]
pub struct LayerLayout {
    pub ln1_offset: usize,
    pub w_q:        BitLinearLayout,
    pub w_k:        BitLinearLayout,
    pub w_v:        BitLinearLayout,
    pub w_out:      BitLinearLayout,
    pub ln2_offset: usize,
    pub fc1:        BitLinearLayout,
    pub fc2:        BitLinearLayout,
}

/// Precomputed byte offsets for every weight section in the `.bin`.
///
/// Walks the wire format once at engine init; cached for the lifetime of the
/// WASM instance. After construction, lookups are pure slice indexing into
/// `LoadedModel::body`.
#[derive(Debug, Clone)]
pub struct WeightLayout {
    pub header: Header,

    // Embedding section
    pub embedding_weights_offset: usize,
    pub embedding_weights_bytes:  usize,
    pub embedding_scales_offset:  usize,   // 0 if format is fp32 (no scales)
    pub embedding_scales_bytes:   usize,

    // Per-layer offsets
    pub layers: Vec<LayerLayout>,

    // Final layernorm + output projection
    pub ln_final_offset:        usize,
    pub projection_weights_offset: usize,
    pub projection_bias_offset:    usize,
}

#[derive(Debug)]
pub struct LoadedModel {
    pub layout: WeightLayout,
    pub body:   &'static [u8],   // MODEL_BYTES minus the trailing sha256
}

/// Lazy-init: parse header + sha256 verify + compute layout on first access.
///
/// Panics on any format error — the engine cannot operate with a malformed
/// `.bin`, and recovery isn't meaningful in WASM.
pub fn get() -> &'static LoadedModel {
    MODEL.get_or_init(|| {
        assert!(
            !MODEL_BYTES.is_empty(),
            "model.bin is empty — the build pipeline must populate engine/assets/model.bin \
             with a packed .bin matching the build's embedding-format feature"
        );
        let body = format::verify_sha256(MODEL_BYTES).expect("sha256 mismatch — corrupted model.bin");
        let header = format::parse_header(body).expect("invalid .bin header");
        let layout = compute_layout(&header, body);
        LoadedModel { layout, body }
    })
}

/// Walk the wire format. Single pass, computes every section's byte offset.
///
/// Layout (must match `training/pack/pack.py` exactly):
///   1. Header (already parsed; offsets start at HEADER_SIZE)
///   2. Embedding section (format-dependent)
///   3. Per layer × n_layers: LN1, Q, K, V (no bias), W_out, LN2, fc1, fc2
///   4. Final LayerNorm
///   5. Output projection (fp32 row-major)
fn compute_layout(header: &Header, body: &[u8]) -> WeightLayout {
    let d_model    = header.d_model    as usize;
    let n_layers   = header.n_layers   as usize;
    let ffn_dim    = header.ffn_dim    as usize;
    let output_dim = header.output_dim as usize;
    let vocab_size = header.vocab_size as usize;

    let mut off = HEADER_SIZE;

    // ── Embedding ───────────────────────────────────────────────────────────
    let embedding_weights_offset = off;
    let (embedding_weights_bytes, embedding_scales_bytes) = match header.embedding_format {
        EMB_FP32    => (vocab_size * d_model * 4, 0),
        EMB_INT8    => (vocab_size * d_model,     vocab_size * 4),
        EMB_TERNARY => (vocab_size * (d_model / 4), vocab_size * 4),   // 2 bits/elem → 4 per byte
        EMB_INT4    => (vocab_size * (d_model / 2), vocab_size * 4),   // 4 bits/elem → 2 per byte
        _ => panic!("compute_layout: unknown embedding_format byte"),
    };
    off += embedding_weights_bytes;
    let embedding_scales_offset = off;
    off += embedding_scales_bytes;

    // ── Per-layer transformer blocks ────────────────────────────────────────
    let ln_bytes = d_model * 4 * 2;   // weight (d_model fp32) + bias (d_model fp32)

    let mut layers = Vec::with_capacity(n_layers);
    for _ in 0..n_layers {
        let ln1_offset = off;
        off += ln_bytes;

        // Q/K/V: each is d_model × d_model, NO bias
        let w_q = take_bitlinear(&mut off, d_model, d_model, false);
        let w_k = take_bitlinear(&mut off, d_model, d_model, false);
        let w_v = take_bitlinear(&mut off, d_model, d_model, false);
        // W_out: d_model × d_model, WITH bias
        let w_out = take_bitlinear(&mut off, d_model, d_model, true);

        let ln2_offset = off;
        off += ln_bytes;

        // fc1: d_model → ffn_dim, WITH bias
        let fc1 = take_bitlinear(&mut off, d_model, ffn_dim, true);
        // fc2: ffn_dim → d_model, WITH bias
        let fc2 = take_bitlinear(&mut off, ffn_dim, d_model, true);

        layers.push(LayerLayout {
            ln1_offset, w_q, w_k, w_v, w_out, ln2_offset, fc1, fc2,
        });
    }

    // ── Final LayerNorm ─────────────────────────────────────────────────────
    let ln_final_offset = off;
    off += ln_bytes;

    // ── Output projection (fp32, NOT ternary) ───────────────────────────────
    let projection_weights_offset = off;
    let proj_weight_bytes = d_model * output_dim * 4;
    off += proj_weight_bytes;
    let projection_bias_offset = off;
    let proj_bias_bytes = output_dim * 4;
    off += proj_bias_bytes;

    // Sanity check: every byte of the body should be accounted for. body length is
    // MODEL_BYTES minus the trailing SHA256_SIZE bytes (already stripped by verify_sha256).
    debug_assert_eq!(
        off,
        body.len(),
        "compute_layout consumed {} bytes, body has {} \
         (mismatch indicates wire-format drift between packer and engine)",
        off, body.len(),
    );
    // SHA256_SIZE is part of MODEL_BYTES but not of `body` — referenced here to keep
    // the import live and the assertion meaningful when re-checking against the file.
    let _ = SHA256_SIZE;

    WeightLayout {
        header: *header,
        embedding_weights_offset, embedding_weights_bytes,
        embedding_scales_offset,  embedding_scales_bytes,
        layers,
        ln_final_offset,
        projection_weights_offset, projection_bias_offset,
    }
}

/// Advance `off` past one BitLinear section, returning its layout.
///
/// Storage shape: `[packed weights | scale (f32) | bias (f32) if has_bias]`.
/// Packed weights = `out_features × (in_features / 4)` bytes (2-bit packing).
fn take_bitlinear(off: &mut usize, in_features: usize, out_features: usize, has_bias: bool) -> BitLinearLayout {
    let weights_bytes = out_features * (in_features / 4);
    let weights_offset = *off;
    *off += weights_bytes;
    let scale_offset = *off;
    *off += 4;   // f32 scale
    let bias_offset = if has_bias {
        let o = *off;
        *off += out_features * 4;
        Some(o)
    } else {
        None
    };
    BitLinearLayout {
        weights_offset, weights_bytes, scale_offset, bias_offset,
        in_features, out_features,
    }
}

// ── Debug surface ───────────────────────────────────────────────────────────

/// Human-readable header dump, exposed via wasm-bindgen for debug.
pub fn config_summary() -> String {
    if MODEL_BYTES.is_empty() {
        return "model.bin not yet provisioned — engine scaffolding only".into();
    }
    let m = get();
    let h = &m.layout.header;
    format!(
        "tern-engine v{} | embedding_format={} | vocab={} d_model={} n_layers={} n_heads={} ffn_dim={} output_dim={} max_seq_len={}",
        h.format_version,
        match h.embedding_format {
            EMB_FP32 => "fp32", EMB_INT8 => "int8", EMB_TERNARY => "ternary", EMB_INT4 => "int4",
            _ => "?",
        },
        h.vocab_size, h.d_model, h.n_layers, h.n_heads, h.ffn_dim, h.output_dim, h.max_seq_len,
    )
}
