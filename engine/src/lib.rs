//! tern-engine — Wasm inference engine for the tern semantic embedding model.
//!
//! Public surface (wasm-bindgen exports):
//!   - `embed(text)` — text in, 384-dim L2-normalized Float32Array out
//!   - `tokenize(text)` — debugging helper, returns token IDs
//!
//! Internal modules:
//!   - tokenizer — HuggingFace BERT tokenizer wrapper (embedded vocab, OnceLock init)
//!   - model — .bin format parser, weight layout, byte-offset readers
//!   - inference — forward pass: embedding lookup → attention → FFN → projection → L2 norm
//!
//! Math reference: docs/training/model-internals.md
//! Postmortem on the inference math we got wrong initially:
//!   docs/training/postmortem-bitlinear-asymmetry.md

use wasm_bindgen::prelude::*;

// TODO: port modules from prototype:
//   mod inference;
//   mod model;
//   mod tokenizer;

/// Smoke-test export. Replace with the real `embed()` once modules are ported.
#[wasm_bindgen]
pub fn hello(input: &str) -> usize {
    input.len()
}