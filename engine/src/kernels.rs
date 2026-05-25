//! Numerical kernels: BitLinear forward + embedding lookup (feature-gated).
//!
//! `bitlinear_forward` is the load-bearing math — it MUST match
//! `training/pack/unpack.py::bitlinear_forward` to within the per-format
//! tolerance documented in `docs/tern-inference-engine.md`. The Python
//! function is the reference; the Rust function below is the implementation.
//!
//! Embedding lookup is feature-gated: each build compiles in exactly one
//! `embedding_lookup_*` variant matching its embedding-format Cargo feature.

// ─────────────────────────────────────────────────────────────────────────────
// BitLinear forward — NOT feature-gated (BitLinear weights are always ternary)
// ─────────────────────────────────────────────────────────────────────────────

/// Engine-equivalent BitLinear forward, mirroring `bitlinear==2.4.6`'s
/// `BitLinear.forward()` with the model's training-time defaults:
///   weight_range = (-1, 1), weight_measure = AbsMedian
///   activation_range = (-128, 127), activation_measure = AbsMax
///   norm = parameterless LayerNorm
///   strategy = round_clamp, lambda = 1 (inference)
///
/// Inputs:
///   x:          [batch_seq, in_features] activations, fp32 row-major
///   w_quant:    [out_features, in_features] ternary weights stored as i8 ∈ {-1, 0, +1}
///   w_scale:    scalar fp32 (from packer)
///   bias:       Optional [out_features] fp32 (None for Q/K/V; Some for W_out/fc1/fc2)
///   in_features, out_features
///
/// Output: [batch_seq, out_features] fp32, written into `out`.
///
/// TODO: implement. Math is documented; this is a 1:1 port of
/// `unpack.bitlinear_forward` plus the parameterless LN step.
pub fn bitlinear_forward(
    _x:            &[f32],
    _w_quant:      &[i8],
    _w_scale:      f32,
    _bias:         Option<&[f32]>,
    _in_features:  usize,
    _out_features: usize,
    _out:          &mut [f32],
) {
    todo!("port from unpack.bitlinear_forward — parameterless LN + AbsMax x_scale + round_clamp + matmul + bias + rescale")
}

// ─────────────────────────────────────────────────────────────────────────────
// Embedding lookup — feature-gated, one variant per build
// ─────────────────────────────────────────────────────────────────────────────

/// Embedding lookup for fp32 builds. Pure gather — copies 4 × d_model bytes per token id.
#[cfg(feature = "emb_fp32")]
pub fn embedding_lookup(
    _ids:     &[u32],
    _table:   &[u8],   // raw bytes from .bin (fp32 row-major)
    _d_model: usize,
    _out:     &mut [f32],
) {
    todo!("emb_fp32: gather row, copy fp32 bytes")
}

/// Embedding lookup for int8 builds. Per-row dequant: row_i × scale_i → fp32.
#[cfg(feature = "emb_int8")]
pub fn embedding_lookup(
    _ids:     &[u32],
    _weights: &[i8],
    _scales:  &[f32],
    _d_model: usize,
    _out:     &mut [f32],
) {
    todo!("emb_int8: gather int8 row, multiply by per-row scale → fp32")
}

/// Embedding lookup for ternary builds. 2-bit unpack: codes {00→0, 01→+1, 10→-1, 11→reserved}.
#[cfg(feature = "emb_ternary")]
pub fn embedding_lookup(
    _ids:     &[u32],
    _packed:  &[u8],
    _scales:  &[f32],
    _d_model: usize,
    _out:     &mut [f32],
) {
    todo!("emb_ternary: 2-bit unpack, multiply by per-row scale → fp32")
}

/// Embedding lookup for int4 builds. Nibble unpack: lower nibble = element 2k, upper = 2k+1.
/// Signed symmetric [-7, +7] (excludes -8) — sign-extend the nibble before dequant.
#[cfg(feature = "emb_int4")]
pub fn embedding_lookup(
    _ids:     &[u32],
    _packed:  &[u8],
    _scales:  &[f32],
    _d_model: usize,
    _out:     &mut [f32],
) {
    todo!("emb_int4: nibble unpack with sign-extend, multiply by per-row scale → fp32")
}
