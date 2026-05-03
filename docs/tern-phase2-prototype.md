# @tern — Phase 2 Prototype: Wasm Engine & Packaging

> **Scope:** This document covers the Rust Wasm engine build — tokenizer integration, ternary inference computation graph, JS bridge, and package assembly. It begins after Phase 1 exits cleanly with a validated ternary model checkpoint and a confirmed tokenizer choice.
>
> For the training pipeline, see [tern-phase1-prototype.md](tern-phase1-prototype.md).
> For architecture reference, see [tern-architecture.md](tern-architecture.md).

---

## 1. Phase 2 Goals

Phase 2 must answer three questions:

1. **Does the Rust Wasm engine compile cleanly to `wasm32-unknown-unknown` with the chosen tokenizer included?**
2. **Does the tokenizer running inside Wasm produce identical token IDs to the Python training tokenizer?**
3. **Does end-to-end inference (raw string → embedding vector) produce embeddings that match Phase 1 eval quality?**

These are the go/no-go gates. Phase 3 (packaging, benchmarking, API surface) does not begin until all three are answered yes.

---

## 2. What Phase 2 Inherits from Phase 1

| Artifact | Source | Used For |
|---|---|---|
| Hardened ternary checkpoint (`.pt`) | Phase 1 training | Weight export to `.bin` |
| `tokenizer.json` (bert-base-uncased) | Downloaded during Phase 1 setup | Embedded in Wasm binary |
| Eval task corpus | Phase 1 results | Regression test in Phase 2 |
| Per-layer weight dimensions | Phase 1 model config | Hardcoded into engine |

---

## 3. Wasm Compilation Target

**Target: `wasm32-unknown-unknown`**

This is the V8-native Wasm target. It runs without modification in Node.js, browsers, Cloudflare Workers, and Vercel Edge. No WASI runtime is required.

The alternative — `wasm32-wasip1` (WASI) — was considered and rejected. WASI would ease some Rust dependency constraints (particularly threading) but removes support for browsers, Cloudflare Workers, and Vercel Edge, which are three of the five primary deployment targets in the product doc. Relaxing the Wasm target shrinks the addressable surface; it is not the right trade.

```bash
rustup target add wasm32-unknown-unknown
cargo build --target wasm32-unknown-unknown --release
```

---

## 4. The Tokenizer Decision

The tokenizer is the first thing to resolve in Phase 2 — before any inference engine code is written. The outcome determines whether a parity test is required.

### The Decision Tree

```
TASK 0 (first thing in Phase 2):
Test compile HuggingFace `tokenizers` crate to wasm32-unknown-unknown
    ↓
Compiles with no linker errors?
    │
    ├── YES
    │     Use HF tokenizers crate (unstable_wasm feature)
    │     Symmetry is structural — same Rust core as Python training
    │     No parity test needed
    │     → Proceed to Section 5
    │
    └── NO (likely rayon threading linker errors)
          Write custom Rust WordPiece (~150 lines)
          Load tokenizer.json via include_bytes!()
          Run parity test against Phase 1 Python output
          → Parity test must pass before proceeding to Section 5
```

### Option A: HuggingFace `tokenizers` Crate

```toml
[dependencies]
tokenizers = { version = "0.19", default-features = false, features = ["unstable_wasm", "fancy-regex"] }
```

- Disables `onig` (C regex — won't compile to Wasm) in favour of `fancy-regex` (pure Rust)
- Disables `esaxx_fast` (C++ backend)
- The `unstable_wasm` flag is not in CI at the HuggingFace repo — it is experimental
- `rayon` (threading) is a non-optional dependency in the crate; `unstable_wasm` does not explicitly address it
- **If rayon causes linker errors, fall through to Option B**

Why `encode()` likely avoids the rayon problem: rayon is used for `encode_batch()` — parallel processing of multiple strings. @tern calls `encode()` on a single string per inference call. Rayon's parallel code paths may not be invoked at all, meaning the dependency compiles in but never triggers thread spawning at runtime.

### Option B: Custom Rust WordPiece (~150 lines)

If Option A fails, implement WordPiece directly. The algorithm is well-specified and the full BERT variant is:

```
1. Unicode normalisation (NFD, lowercase, strip accents)
2. Whitespace + punctuation splitting
3. For each word: greedy longest-match subword lookup against vocab HashMap
4. Emit [UNK] (id=100) for words with no subword coverage
5. Prepend [CLS] (id=101), append [SEP] (id=102)
6. Pad or truncate to max_seq_len=128
```

The vocab is the same `tokenizer.json` loaded via `include_bytes!()`. The HashMap is built once at init from the embedded JSON. No regex library needed.

Zero non-standard Rust dependencies. Compiles to `wasm32-unknown-unknown` with no flags.

### Parity Test (Required for Option B, Spot-Check for Option A)

The parity test validates that the Wasm tokenizer and the Phase 1 Python tokenizer produce identical token ID sequences.

Test corpus: 500 strings — natural language queries, developer error messages, camelCase and snake_case identifiers, mixed punctuation, edge cases (empty string, very long input, unicode).

```python
# parity_check.py
# Load Phase 1 Python tokenizer (HF tokenizers library)
# Tokenize all 500 strings, save token ID sequences
# Load Wasm module, tokenize same 500 strings via embed() introspection
# Compare sequences exactly
# PASS: 100% match
# FAIL: any divergence → debug before proceeding
```

**For Option A:** spot-check 50 strings. Full parity is implied by shared Rust core, but unicode edge cases should be verified.
**For Option B:** full 500-string test required. Hard gate.

---

## 5. Vocab Embedding Strategy

The `bert-base-uncased` vocabulary (~115KB binary) is embedded in the Wasm binary at compile time. No separate file ships with the package.

```rust
// Committed to the repo at assets/tokenizer.json
// Downloaded once during development via:
//   python -c "from tokenizers import Tokenizer; \
//              Tokenizer.from_pretrained('bert-base-uncased').save('assets/tokenizer.json')"

static TOKENIZER_BYTES: &[u8] = include_bytes!("../assets/tokenizer.json");

static TOKENIZER: OnceLock<Tokenizer> = OnceLock::new();

pub fn get_tokenizer() -> &'static Tokenizer {
    TOKENIZER.get_or_init(|| {
        Tokenizer::from_bytes(TOKENIZER_BYTES).expect("embedded tokenizer.json is invalid")
    })
}
```

The `tokenizer.json` file is committed to the repository as a static asset. It is not downloaded at build time or runtime.

---

## 6. Model Weight Export (`.bin` File)

Before the Wasm engine can be tested end-to-end, the Phase 1 checkpoint must be exported to the `.bin` format the engine expects.

```python
# export.py — run once after Phase 1 training completes
# 1. Load hardened ternary checkpoint
# 2. Materialize {-1, 0, +1} weights
# 3. Pack: 4 weights per byte using 2-bit encoding (00=0, 01=+1, 10=-1)
# 4. Write 24-byte header + packed weight matrices sequentially
```

Header format (24 bytes):

```
Offset  Size  Field
0       4B    Magic: 0x5445524E ("TERN")
4       2B    Format version
6       2B    d_model
8       2B    n_layers
10      2B    n_heads
12      2B    ffn_dim
14      2B    vocab_size
16      2B    max_seq_len
18      2B    Reserved
20      4B    Total weight bytes (excluding header)
```

The engine reads dimensional constants from the header at init and allocates memory accordingly. This is what allows one engine binary to serve nano, micro, and base tier models.

---

## 7. The Inference Engine

The engine is a hardcoded computation graph — not a generic inference framework. It is structurally coupled to the transformer architecture defined in Phase 1.

### Execution Flow

```
wasm_bindgen export: embed(text: &str) -> Vec<f32>
    ↓
tokenize(text) → [u32; 128]          // via embedded tokenizer
    ↓
embedding_lookup(ids) → [f32; 256]   // read from packed embedding table
    ↓
for each layer (×2):
    layer_norm(x)
    attention(x) → q, k, v           // BitLinear: additions/subtractions only
    scaled_dot_product(q, k, v)
    output_projection(attn_out)
    residual_add(x, attn_out)
    layer_norm(x)
    ffn(x)                            // BitLinear: up 256→1024, down 1024→256
    residual_add(x, ffn_out)
    ↓
mean_pool(sequence_output) → [f32; 256]   // average over non-PAD positions
    ↓
l2_normalize(pooled) → [f32; 256]         // unit vector for cosine similarity
```

### BitLinear Execution

At inference, BitLinear layers contain hardened ternary weights — no float32 shadow weights, no straight-through estimator. Matrix multiplication reduces to:

```rust
// For each output neuron:
// accumulate += input[i] when weight[i] == +1
// accumulate -= input[i] when weight[i] == -1
// skip         when weight[i] == 0
```

No floating-point multiplication anywhere in attention or FFN. Layer norm and mean pooling remain float32.

### Memory Layout

Single contiguous allocation at startup:

```
[model header: 24B][embedding table][layer 0 weights][layer 1 weights]
                    ↑ all bit-packed, read directly, no copy
```

The model `.bin` is loaded into this region. No deserialization step.

---

## 8. JS Bridge

The Node.js wrapper is intentionally thin. All tokenization and inference happens inside Wasm.

```js
// index.js
const { readFileSync } = require('fs');
const { join } = require('path');

let _instance = null;

async function init() {
  if (_instance) return;
  const wasmBytes = readFileSync(join(__dirname, 'engine.wasm'));
  const modelBytes = readFileSync(join(__dirname, 'model.bin'));
  const { instance } = await WebAssembly.instantiate(wasmBytes);
  // write model bytes into Wasm linear memory
  const mem = new Uint8Array(instance.exports.memory.buffer);
  mem.set(modelBytes, instance.exports.model_offset());
  instance.exports.init();
  _instance = instance;
}

function embed(text) {
  // returns Float32Array of length 256
  return _instance.exports.embed(text);
}

function similarity(a, b) {
  const va = embed(a);
  const vb = embed(b);
  return cosineSimilarity(va, vb); // pure JS dot product
}
```

`similarity()` and `classify()` are pure JS operations over float32 vectors returned by `embed()`. No Wasm involvement beyond the embedding call.

---

## 9. Wasm Binary Size Validation

After the first successful build, measure the actual binary size before proceeding:

```bash
cargo build --target wasm32-unknown-unknown --release
wasm-opt -O3 -o engine_opt.wasm target/wasm32-unknown-unknown/release/tern_engine.wasm
ls -lh engine_opt.wasm
```

Expected breakdown:

| Component | Expected |
|---|---|
| Inference engine (Rust) | ~400KB |
| `tokenizers` crate or custom WordPiece | ~200–300KB (Option A) / ~20KB (Option B) |
| BERT vocab embedded | ~115KB |
| **Total Wasm binary** | **~715–815KB (A) / ~535KB (B)** |

If the binary exceeds 1.2MB after `wasm-opt`, investigate which dependency is the source before continuing. `twiggy` (Wasm size profiler) can attribute bytes to specific Rust functions.

---

## 10. Risks, Thresholds, and Go/No-Go Criteria

### Risk 1: `tokenizers` Crate Fails to Compile to Wasm

**What it is:** `rayon` or another non-optional dependency causes linker errors on `wasm32-unknown-unknown`.

**How to detect:** First test compile (Task 0).

| Outcome | Action |
|---|---|
| Compiles cleanly | Proceed with Option A |
| Linker errors (rayon/threading) | Fall through to Option B (custom WordPiece) |
| Linker errors from other dependencies | Investigate — may be fixable with feature flags |

**This is the first task in Phase 2. It costs one hour and resolves the largest unknown.**

---

### Risk 2: Parity Test Failures (Option B Only)

**What it is:** Custom Rust WordPiece produces different token IDs than the Python `tokenizers` reference on edge case inputs.

**Common divergence sources:**
- Unicode normalization (NFD vs NFC, accent stripping)
- Punctuation splitting rules (BERT has specific handling for CJK, accents)
- `[UNK]` emission logic for out-of-vocabulary subwords

| Threshold | Status |
|---|---|
| 100% match on 500-string corpus | Proceed |
| Failures only on unicode / non-ASCII | Fix normalization, retest |
| Failures on standard ASCII text | **Hard stop — implementation error, debug** |

---

### Risk 3: Wasm Binary Size Exceeds Budget

**What it is:** The compiled Wasm binary is larger than estimated, consuming too much of the 5MB package budget.

| Binary size (post wasm-opt) | Status |
|---|---|
| < 800KB | Acceptable |
| 800KB – 1.2MB | Marginal — profile with `twiggy`, look for removable dependencies |
| > 1.2MB | **No-go — switch to Option B tokenizer, audit all dependencies** |

---

### Risk 4: End-to-End Embedding Quality Regression

**What it is:** Wasm inference produces embeddings that score materially worse than Phase 1 Python eval, indicating a bug in weight loading, bit-unpacking, or the computation graph.

Run the Phase 1 eval task corpus through the Wasm engine and compare:

| Metric | Acceptable Regression | No-go |
|---|---|---|
| Mean cosine sim vs Phase 1 | < 0.02 drop | > 0.05 drop |
| STS proxy AUC | < 0.02 drop | > 0.05 drop |
| Recall@3 | < 0.02 drop | > 0.05 drop |

Any regression beyond acceptable thresholds indicates a bug — not a model quality issue. Likely candidates: bit-unpacking error, incorrect weight matrix ordering in `.bin`, wrong layer norm placement.

---

### Risk 5: SIMD Availability

**What it is:** SIMD instructions (`wasm32 simd128`) improve inference throughput but are not guaranteed across all Wasm runtimes. Cloudflare Workers and modern Node.js support SIMD; older environments may not.

**Strategy:** Implement inference without SIMD first, validate correctness. Add SIMD optimisation in a separate pass once the baseline is confirmed. Do not block Phase 2 exit on SIMD.

---

## 11. Phase 2 Exit Criteria

Phase 2 is complete and Phase 3 (packaging, benchmarking, API surface) can begin when **all of the following are true**:

- [ ] Test compile to `wasm32-unknown-unknown` succeeds (tokenizer decision resolved)
- [ ] Parity test passes (100% for Option B, spot-check for Option A)
- [ ] `.bin` export script produces a valid model file readable by the engine
- [ ] End-to-end `embed(text)` call returns a 256-dim float32 vector in Node.js
- [ ] Embedding quality regression vs Phase 1 is within acceptable thresholds on all three eval tasks
- [ ] Wasm binary size is under 1.2MB post `wasm-opt`
- [ ] `similarity(a, b)` returns a score in `[0, 1]` on a sanity-check pair ("dog" / "cat" > "dog" / "quarterly earnings")

---

## 12. Outputs

| Artifact | Description |
|---|---|
| `engine/` | Rust crate — tokenizer + inference engine, compiles to Wasm |
| `engine/assets/tokenizer.json` | Committed vocab file, embedded at compile time |
| `export.py` | Phase 1 checkpoint → `.bin` converter |
| `engine.wasm` | Compiled, `wasm-opt`-optimised Wasm binary |
| `model.bin` | Exported micro-tier model weights |
| `index.js` | Thin Node.js wrapper — init, embed, similarity, classify |
| `parity_check.py` | Python vs Wasm tokenizer comparison (required for Option B) |
| `regression_test.js` | Runs Phase 1 eval corpus through Wasm, reports quality metrics |
