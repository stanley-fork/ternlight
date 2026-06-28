# Tern Inference Engine — Design

> Rust crate at [`engine/`](../engine/), compiled to WebAssembly via `wasm-pack`. Loads a packed `.bin` produced by [`training/pack/`](../training/distill/), tokenizes input text, runs the BitLinear-faithful forward pass, returns a 384-dim L2-normalized embedding.

---

## Overview

| | |
|---|---|
| **Input** | UTF-8 string (any length; tokenizer truncates to 128) |
| **Output** | `Float32Array(384)`, L2-normalized |
| **WASM target** | `wasm32-unknown-unknown` (no WASI — runs in Node.js, browsers, Cloudflare Workers, Vercel Edge) |
| **Build tool** | `wasm-pack` + `wasm-opt -Oz` |
| **Format coupling** | Single-format per build (Cargo feature picks the embedding loader at compile time — no runtime dispatch) |

---

## Design principles

- **One build = one format.** Each WASM artifact bakes in exactly one embedding loader. No runtime format dispatch — bundles stay tight, hot paths stay branchless.
- **BitLinear weights are always ternary.** That's the project's reason for existing. Only the embedding-table precision varies across builds.
- **Parity-first.** Engine output must match the Python reference at [`training/pack/unpack.py`](../training/pack/unpack.py) (`UnpackedModel.forward()`). Computational inference-level parity, not byte-level.
- **Static artifacts, baked at compile time.** Tokenizer vocab + model weights are embedded into the WASM via `include_bytes!` - no runtime file I/O, no network at inference time.

---

## File layout

```
engine/
├── Cargo.toml         build features (mutually exclusive)
├── src/
│   ├── lib.rs         WASM API (#[wasm_bindgen])
│   ├── format.rs      .bin header parse + verify
│   ├── tokenizer.rs   BERT tokenizer (vocab embedded)
│   ├── model.rs       weight layout precompute
│   ├── kernels.rs     BitLinear + embedding kernels
│   └── inference.rs   forward pass
├── assets/
│   ├── tokenizer.json BERT vocab (committed)
│   └── model.bin      weights (release artifact, not committed)
└── tests/             parity tests
```

---

## Build targets

The engine compiles to one of several feature-gated variants per build:

| Target | Role | Embedding format | Cargo feature | Bundle (WASM + `.bin`) |
|---|---|---|---|---|
| `tern-engine-emb-int4` | **primary ship** | 4-bit per-row PTQ + per-row fp32 scale | `emb_int4` | ~7 MB |
| `tern-engine-emb-int8` | higher-quality variant | 8-bit per-row + per-row fp32 scale | `emb_int8` | ~11 MB |
| `tern-engine-emb-ternary` | size-extreme variant | packed ternary {-1, 0, +1} + per-row fp32 scale | `emb_ternary` | ~5 MB |
| `tern-engine-emb-fp32` | parity reference only — **NOT for user ship** | fp32 row-major | `emb_fp32` | ~40 MB |

Exactly one feature per build, enforced at compile time via `compile_error!` in `lib.rs`.

---

## Tokenizer

HuggingFace `tokenizers` crate with the `unstable_wasm` feature, `fancy-regex` backend (pure Rust, no C deps).

```toml
tokenizers = { version = "0.19", default-features = false, features = ["unstable_wasm", "fancy-regex"] }
```

Vocab is BERT-base-uncased, loaded once via `OnceLock<Tokenizer>`, embedded into the WASM at compile time:

```rust
static TOKENIZER_BYTES: &[u8] = include_bytes!("../assets/tokenizer.json");
```

No parity test against Python is required — the same Rust crate underpins both training-time tokenization (via Python `transformers`) and engine-time tokenization. Spot-check ~10 strings during integration.

---

## `.bin` wire format (v1)

The packer writes a single `.bin` per build target. The engine reads it linearly during load; the only random access at inference time is into the embedding section.

```
header (32 bytes, little-endian):
  magic                 4 bytes   "TERN" (0x5445524E)
  format_version        2 bytes   uint16 — currently 1
  embedding_format      1 byte    0=fp32, 1=int8_per_row, 2=ternary_packed, 3=int4_per_row
  weights_format        1 byte    0=ternary_packed (only option in v1)
  vocab_size            4 bytes   uint32
  d_model               2 bytes   uint16
  n_layers              1 byte
  n_heads               1 byte
  ffn_dim               2 bytes   uint16
  output_dim            2 bytes   uint16
  max_seq_len           2 bytes   uint16
  reserved             10 bytes   zero-padding to 32-byte alignment

embedding section:
  layout depends on embedding_format byte — see "Embedding layouts" below

per-layer transformer block (× n_layers):
  ln_1.weight                       d_model fp32
  ln_1.bias                         d_model fp32
  attn.q/k/v.weight                 packed ternary, each is d_model × d_model
  attn.q/k/v.weight_scale           fp32 per matrix
  attn.out.weight                   packed ternary, d_model × d_model
  attn.out.weight_scale             fp32
  attn.out.bias                     d_model fp32
  ln_2.weight, ln_2.bias            d_model fp32 each
  ffn.up.weight                     packed ternary, d_model × ffn_dim
  ffn.up.weight_scale               fp32
  ffn.up.bias                       ffn_dim fp32
  ffn.down.weight                   packed ternary, ffn_dim × d_model
  ffn.down.weight_scale             fp32
  ffn.down.bias                     d_model fp32

final layer norm:
  ln_final.weight                   d_model fp32
  ln_final.bias                     d_model fp32

output projection (fp32, NOT ternary):
  out.weight                        d_model × output_dim × 4 bytes fp32 row-major
  out.bias                          output_dim × 4 bytes fp32

trailing:
  sha256                            32 bytes — hash of all preceding bytes
```

Q/K/V matrices carry no bias (matches the source architecture). The output projection stays fp32 by design — quantization noise on the layer that bridges the student's coordinate frame to the teacher's directly corrupts the distillation signal. LayerNorm parameters stay fp32 (tiny, not worth quantizing). Per-matrix `w_scale` for each BitLinear is mandatory.

### Embedding layouts

All quantized formats carry **per-row fp32 scales** — one scale per vocab row. Per-row scaling preserves dynamic range per token; do not collapse to a single global scale.

**`emb_fp32` (format_id = 0)**
```
weights:  vocab_size × d_model × 4 bytes, row-major fp32
```
~31 MB for our config (vocab=30522, d_model=256).

**`emb_int8` (format_id = 1)**
```
weights:  vocab_size × d_model bytes, int8
scales:   vocab_size × 4 bytes, fp32 (one per row)
```
~7.9 MB.

**`emb_ternary` (format_id = 2)**
```
weights:  vocab_size × ceil(d_model × 2 / 8) bytes packed
          (2 bits per element: 00=zero, 01=+1, 10=-1, 11=reserved)
scales:   vocab_size × 4 bytes, fp32 (one per row)
```
~2.1 MB. Tighter 1.585-bit packing (5 ternary values per byte) is feasible but deferred — would require a `format_version` bump.

**`emb_int4` (format_id = 3)** — **primary ship variant**
```
weights:  vocab_size × (d_model / 2) bytes packed
          (4 bits per element, signed symmetric [-7, +7]; lower nibble = element 2k, upper = 2k+1)
scales:   vocab_size × 4 bytes, fp32 (one per row)
```
~4.0 MB.

---

## Public API (`#[wasm_bindgen]`)

| Function | Signature | Purpose |
|---|---|---|
| `embed(text)` | `&str → Vec<f32>` (length = output_dim = 384) | Primary entry point; returns L2-normalized embedding |
| `tokenize(text)` | `&str → Vec<u32>` | Debug — exposes token IDs for parity tests |
| `config_summary()` | `→ String` | Debug — human-readable header dump |

JS bridge stays thin — `similarity(a, b)` and `classify(text, labels)` live above the WASM boundary as pure JS over the returned float32 vectors.


---

## Future format work

Tighter packing (5 ternary values per byte → ~18% smaller `.bin`), transport-layer compression, sparse embedding rows, and fp16 output projection are tracked as future byte-savings. They're format-level changes that would require a `format_version` bump; v1 stays as specified above.
