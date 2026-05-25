//! End-to-end forward pass. Mirrors `training/distill/model.py::StudentEncoder::forward`
//! and `training/pack/unpack.py::UnpackedModel::forward`.
//!
//! Flow:
//!   tokenize(text)
//!     → embedding_lookup(input_ids)
//!     → transformer_layer × n_layers (LN1 → attention → +residual → LN2 → FFN → +residual)
//!     → final LayerNorm
//!     → mean pool over non-padding positions
//!     → fp32 projection (NOT ternary — see postmortem)
//!     → L2 normalize
//!     → 384-dim Float32Array

/// Primary forward pass. Implements the StudentEncoder forward graph using the
/// kernels in `crate::kernels` against the layout in `crate::model::get()`.
///
/// TODO: implement. Most of the wiring is structural — the only load-bearing
/// math is in `kernels::bitlinear_forward`.
pub fn embed(_text: &str) -> Vec<f32> {
    todo!("end-to-end forward — token IDs → transformer → projection → L2 norm")
}
