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

/// Byte offsets for every weight section in the `.bin`, computed once from
/// the header and the architecture's per-layer shape.
///
/// Stored as offsets (not slices) so the layout walk can complete without
/// having to materialize references to the underlying bytes — the slice is
/// constructed lazily at each access site.
#[derive(Debug, Clone)]
pub struct WeightLayout {
    pub header: Header,

    // Embedding section
    pub embedding_weights_offset: usize,
    pub embedding_weights_bytes:  usize,
    pub embedding_scales_offset:  usize,   // 0 if format is fp32 (no scales)
    pub embedding_scales_bytes:   usize,

    // Per-layer offsets — Vec, one entry per layer
    pub layers: Vec<LayerLayout>,

    // Final layernorm + output projection
    pub ln_final_offset:   usize,
    pub projection_offset: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct LayerLayout {
    pub ln1_offset:        usize,
    pub w_q_offset:        usize,
    pub w_q_scale_offset:  usize,
    pub w_k_offset:        usize,
    pub w_k_scale_offset:  usize,
    pub w_v_offset:        usize,
    pub w_v_scale_offset:  usize,
    pub w_out_offset:      usize,   // weights + scale + bias
    pub ln2_offset:        usize,
    pub fc1_offset:        usize,   // weights + scale + bias
    pub fc2_offset:        usize,   // weights + scale + bias
}

#[derive(Debug)]
pub struct LoadedModel {
    pub layout: WeightLayout,
    // The body slice (MODEL_BYTES without the trailing sha256), validated at load.
    // 'static because MODEL_BYTES is itself 'static.
    pub body:   &'static [u8],
}

/// Lazy-init: parse header + sha256 verify + compute layout on first access.
///
/// Panics on any format error — engine cannot operate with a malformed `.bin`,
/// and recovery isn't meaningful in WASM (the binary either works or doesn't).
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

/// Compute byte offsets for every section. Single linear walk through the
/// architecture; cached for the lifetime of the WASM instance.
///
/// TODO: implement. The walk is mechanical given the section order in
/// `docs/tern-inference-engine.md`; the only format-dependent piece is the
/// embedding-section size.
fn compute_layout(header: &Header, _body: &[u8]) -> WeightLayout {
    let _ = header;
    todo!("compute_layout — port from tern-core's WeightLayout::compute() walking the v1 spec")
}

/// Public helper: human-readable header dump, exposed via wasm-bindgen for debug.
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
