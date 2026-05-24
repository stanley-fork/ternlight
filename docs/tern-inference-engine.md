# Tern Inference Engine — Design

> The math half of `tern`. Loads a packed `.bin`, runs the BitLinear-faithful forward pass, returns a 384-dim L2-normalized embedding. Rust crate at [`engine/`](../engine/) compiles to `wasm32-unknown-unknown` via `wasm-pack`.

This doc captures the design for the **multi-format-capable** engine that consumes Phase 5's packed artifacts. The existing v0.1 engine code in [`engine/`](../engine/) is single-format (ternary embedding only) and predates the Stage A finding that embedding precision is a load-bearing knob.

---

## Design philosophy

- **One build = one format.** Each WASM artifact ships with exactly one embedding loader + dequant kernel baked in via Cargo features. No runtime format dispatch. A configurable engine would mean larger bundles, branchier hot paths, and N × the test matrix — we pay that complexity in the *source code* (multiple feature-gated modules) rather than in the *shipped binary* (one path, optimized hard).
- **BitLinear weights are always ternary.** That's the project's reason for existing. Only the *embedding table* precision varies across build targets. The transformer block kernels are identical across all builds.
- **Parity-first.** Engine output must match the Python reference eval ([`evaluation.py`](../training/distill/evaluation.py)) within tolerance for the same input. Without this contract, engine quality numbers are unfalsifiable. See "Verification" below.
- **The `.bin` is opaque without the sidecar.** Always travel together; the sidecar identifies the embedding format and engine build target it pairs with.

---

## Scope of this doc

1. [Build targets](#build-targets) — the embedding-precision matrix
2. [`.bin` wire format](#bin-wire-format) — header, sections, embedding layouts
3. [Unpacking](#unpacking--bin--in-memory-model) — engine load path
4. [Engine runtime](#engine-runtime) — forward pass, kernels
5. [Cargo feature wiring](#cargo-feature-wiring) — how the build flag maps to compiled code
6. [Verification — parity contract](#verification--parity-contract)
7. [Target-device perf](#target-device-perf) — placeholder, much later

---

## Build targets

Three planned, gated on a single Cargo feature flag selected at `wasm-pack` build time:

| Target | Embedding format | `--features` | Bundle size (est) | Use case |
|---|---|---|---|---|
| `tern-engine-emb-ternary` | packed ternary {-1,0,+1} + per-row fp32 scale | `emb_ternary` | ~6 MB | smallest ship artifact |
| `tern-engine-emb-int8` | int8 per-row + fp32 scale | `emb_int8` | ~8 MB | quality-conscious ship |
| `tern-engine-emb-fp32` | fp32 row-major | `emb_fp32` | ~31 MB | research / quality ceiling reference; likely too large for production ship |

Exactly one of `emb_ternary | emb_int8 | emb_fp32` must be enabled per build. Compile error if zero or more than one are set — enforced via a `compile_error!` in [`engine/src/lib.rs`](../engine/src/lib.rs).

Future formats (NF4, int4, fp16) would each add a feature gate; the core architecture doesn't change.

---

## `.bin` wire format

The packer ([`training/pack/pack.py`](../training/pack/), Phase 5) writes a single `.bin` per build target. The engine reads it linearly during load — no random access except into the embedding table at inference time.

```
header (32 bytes, aligned):
  magic                 4 bytes  "TERN" (0x5445524E)
  format_version        1 byte   currently 1
  embedding_format      1 byte   0=fp32, 1=int8_per_row, 2=ternary_packed
  weights_format        1 byte   0=ternary_packed (only option v1)
  reserved              1 byte   alignment
  vocab_size            4 bytes  little-endian uint32
  d_model               2 bytes  uint16
  n_layers              1 byte
  n_heads               1 byte
  ffn_dim               2 bytes  uint16
  output_dim            2 bytes  uint16
  dropout_x1000         2 bytes  uint16  (dropout × 1000, info-only — not used at inference)
  reserved              6 bytes  alignment to 32

embedding section:                  ← layout depends on embedding_format
  (see "Embedding layouts" below)

per-layer transformer block (× n_layers):
  ln_1.weight                       d_model fp32
  ln_1.bias                         d_model fp32
  attn.q/k/v/out.weight             packed ternary (3 × d_model × d_model + d_model × d_model)
  attn.q/k/v/out.weight_scale       fp32 per matrix
  attn.q/k/v/out.bias               d_model fp32 each
  ln_2.weight, ln_2.bias            d_model fp32 each
  ffn.up.weight                     packed ternary, d_model × ffn_dim
  ffn.up.weight_scale               fp32
  ffn.up.bias                       ffn_dim fp32
  ffn.down.weight                   packed ternary, ffn_dim × d_model
  ffn.down.weight_scale             fp32
  ffn.down.bias                     d_model fp32

output projection:
  out.weight                        packed ternary, d_model × output_dim
  out.weight_scale                  fp32
  out.bias                          output_dim fp32

trailing:
  sha256                            32 bytes — hash of all preceding bytes
```

LayerNorm parameters stay fp32 — they're tiny (d_model × 2 per LN) and quantizing them isn't worth the complexity.

### Embedding layouts

**`emb_fp32` (format_id = 0):**
```
weights:  vocab_size × d_model × 4 bytes, row-major fp32
```
Total: `vocab_size × d_model × 4` bytes. For our config (30522 × 256): ~31 MB.

**`emb_int8` (format_id = 1):**
```
weights:  vocab_size × d_model bytes, row-major int8
scales:   vocab_size × 4 bytes fp32 (one scale per row)
```
Total: `vocab_size × (d_model + 4)` bytes. For our config: ~7.9 MB. Per-row scaling preserves dynamic range per token.

**`emb_ternary` (format_id = 2):**
```
weights:  vocab_size × ceil(d_model × 2 / 8) bytes packed (2 bits per element: 00=zero, 01=+1, 10=-1, 11=reserved)
scales:   vocab_size × 4 bytes fp32 (one scale per row)
```
Total: `vocab_size × (ceil(d_model × 2/8) + 4)` bytes. For our config: ~2.1 MB weights + ~120 KB scales ≈ 2.2 MB.

Note: theoretical limit for ternary is 1.58 bits/element (log₂(3)). We use 2 bits for byte-alignment and decode simplicity; the 25% overhead is worth the kernel speedup. A future format version could switch to true 1.58-bit packing if bundle size becomes critical.

---

## Unpacking — `.bin` → in-memory model

Engine load (cold-start path) does:

1. **Validate header.** Magic + version + format match the compile-time `embedding_format` feature. Bail with a typed error if the `.bin` was packed for a different target.
2. **Read sections sequentially** into pre-allocated buffers. WASM linear memory is the destination.
3. **Hold weights in their on-disk format** — do NOT pre-dequantize. Dequant happens lazily per-token at inference time. Pre-dequantizing would defeat the whole bundle-size argument.
4. **Verify trailing sha256.** Fail load if mismatch.

Total cold-start work is dominated by network download + WASM linear memory growth — actual parsing is O(file_size) byte-copies into bounded buffers, no compute.

---

## Engine runtime

```
embed(text: &str) -> [f32; output_dim]:
  1. tokenize(text)                    → input_ids: Vec<u32>
  2. embedding_lookup(input_ids)       → hidden: [seq_len, d_model] fp32
                                         (feature-gated: dispatches to fp32/int8/ternary loader)
  3. for layer in 0..n_layers:
       hidden = transformer_block(hidden)
       (LayerNorm → BitLinear-q/k/v → attn → BitLinear-out + residual
        → LayerNorm → BitLinear-up → activation → BitLinear-down + residual)
  4. pooled = mean_pool(hidden)        → [d_model]
  5. out = bitlinear(pooled, out.weight, out.bias)  → [output_dim]
  6. l2_normalize(out)                 → [output_dim]
  return out
```

### Embedding-lookup kernel (feature-gated)

```rust
#[cfg(feature = "emb_fp32")]
fn embed_lookup(ids: &[u32], table: &[f32], d_model: usize, out: &mut [f32]) {
    // simple gather: copy 4 × d_model bytes per id
}

#[cfg(feature = "emb_int8")]
fn embed_lookup(ids: &[u32], table: &[i8], scales: &[f32], d_model: usize, out: &mut [f32]) {
    // gather int8 row, multiply by per-row scale → fp32
    // SIMD-friendly: process 16 int8 → 4 × f32x4 in WASM128
}

#[cfg(feature = "emb_ternary")]
fn embed_lookup(ids: &[u32], packed: &[u8], scales: &[f32], d_model: usize, out: &mut [f32]) {
    // 2-bit unpack: extract sign code, multiply by ±scale or zero
    // SIMD via lookup table + shuffle; benchmark before committing to specific impl
}
```

Same signature otherwise. Caller doesn't know which path it called.

### BitLinear forward kernel (NOT feature-gated)

`bitlinear_forward()` lives in [`engine/src/inference.rs`](../engine/src/) (per the existing engine README) and mirrors `BitLinear.forward` from the training-time library exactly. This is the bitlinear-asymmetry-postmortem-critical code path — see [`docs/training/postmortem-bitlinear-asymmetry.md`](training/postmortem-bitlinear-asymmetry.md) for what went wrong the first time and why parity matters.

The math:
```
hidden_ln = LayerNorm(hidden)
hidden_q  = activation_quant_int8(hidden_ln)          // per-token activation scale
result    = matmul_int8_ternary(hidden_q, weight)     // {-128..127} × {-1,0,+1} → int32 accumulator
result_f  = result * (activation_scale * weight_scale)
return result_f + bias
```

This kernel does NOT change across embedding-precision build targets.

---

## Cargo feature wiring

In [`engine/Cargo.toml`](../engine/Cargo.toml):

```toml
[features]
default = []
emb_fp32    = []
emb_int8    = []
emb_ternary = []
```

In [`engine/src/lib.rs`](../engine/src/lib.rs):

```rust
#[cfg(not(any(feature = "emb_fp32", feature = "emb_int8", feature = "emb_ternary")))]
compile_error!("Exactly one embedding format feature must be enabled: emb_fp32, emb_int8, or emb_ternary");

#[cfg(any(
    all(feature = "emb_fp32", feature = "emb_int8"),
    all(feature = "emb_fp32", feature = "emb_ternary"),
    all(feature = "emb_int8", feature = "emb_ternary"),
))]
compile_error!("Only one embedding format feature may be enabled at a time");
```

Build invocation per target ([`scripts/build-engine.sh`](../scripts/) extended):

```bash
wasm-pack build --target nodejs --features emb_ternary  # smallest
wasm-pack build --target nodejs --features emb_int8     # quality-conscious
wasm-pack build --target nodejs --features emb_fp32     # research only
```

CI builds all three (eventually). Production release picks one based on whichever embedding format wins the Stage A retrieval bake-off (see [tern-training-pipeline.md](tern-training-pipeline.md#stage-a--close-phase-4-retrieval-task--int8-ptq-ablation)).

---

## Verification — parity contract

For every build target, a parity test:

1. Load `.bin` matching the build's feature flag
2. Tokenize the same N inputs the Python eval uses
3. Forward through the engine
4. Compare engine output to the Python reference dump (from `evaluation.py`) at the L2-normalized embedding level

**Tolerance:**

| Build | Tolerance per dim |
|---|---|
| `emb_fp32` | 1e-5 (effectively bit-exact) |
| `emb_int8` | 1e-4 (round-trip is lossless per-element ±0.5 ULP × scale) |
| `emb_ternary` | 5e-3 OR document the looser bound after first measurement |

Failure modes the parity test must catch:
- Endianness drift (`.bin` written little-endian, read big-endian or vice versa)
- Sign-table mismatch in ternary unpack (the bitlinear-asymmetry-class bug)
- Off-by-one in per-row scale indexing for int8 / ternary
- Wrong activation_scale × weight_scale combination order in BitLinear
- LayerNorm epsilon mismatch between Python and Rust

Without this contract you cannot trust on-device perf numbers — slow numbers might just be "engine got the wrong answer fast."

Reference dumps live in [`engine/tests/`](../engine/tests/) per the existing engine README.

---

## Target-device perf

*Placeholder — to be filled in once the engine has a working multi-format baseline.*

Will cover:
- **Cold-start**: network download + WASM instantiation + first-inference latency, by target
- **Steady-state**: p50 / p99 per-query latency at typical UI-search query length (3–30 tokens)
- **Memory footprint**: peak vs steady-state, WASM linear memory size
- **Devices in scope**: TBD — at minimum Mac M-series, mid-tier x86 laptop, generic WASM in browser. Mobile + low-end laptop as expansion targets.

Benchmark harness lives in [`eval/benchmarks/`](../eval/benchmarks/) at repo root (separate concern from training-time eval, which lives in [`training/distill/evaluation.py`](../training/distill/evaluation.py)). The two should *not* be merged — training eval measures the `.pt` ckpt, target-device perf measures the shipped `.wasm` artifact. Different artifacts, different questions.

This section gets fleshed out post-Phase 5 once we have at least one build target producing a working WASM that loads a real `.bin`. Until then, latency estimates in this doc are projections from kernel-level analysis, not measurements.
