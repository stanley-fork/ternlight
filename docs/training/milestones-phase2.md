# Phase 2 Milestones — Wasm Engine Build

> This document breaks Phase 2 into small, observable steps. Each step has a clear goal, a deliverable you can verify, and a decision point before moving forward. The full spec lives in [tern-phase2-prototype.md](../tern-phase2-prototype.md) — this is the working breakdown of how to get there.
>
> Phase 1 conclusion: [phase-1-conclusion.md](phase-1-conclusion.md)

---

## Directory Structure

```
tern-distill-prototype/
├── poc/                       # Phase 1 training (complete)
│
├── export/                    # Step 1: .bin export
│   ├── export.py              # .pt checkpoint → .bin converter
│   ├── verify.py              # read .bin back, compare against .pt
│   └── out/                   # generated .bin files
│
├── engine/                    # Steps 2–5: Rust → Wasm
│   ├── Cargo.toml
│   ├── src/
│   │   ├── lib.rs             # wasm-bindgen entry point
│   │   ├── tokenizer.rs       # HF tokenizers crate or custom WordPiece
│   │   ├── model.rs           # .bin loading + bit-unpacking
│   │   └── inference.rs       # forward pass (hardcoded computation graph)
│   └── assets/
│       └── tokenizer.json     # bert-base-uncased vocab (embedded at compile time)
│
├── bridge/                    # Step 6: JS integration
│   ├── index.js               # thin Node.js wrapper
│   ├── regression_test.js     # Phase 1 eval corpus through Wasm engine
│   └── parity_check.py        # Python vs Wasm tokenizer comparison
│
├── milestones.md              # Phase 1 milestones (existing)
├── milestones-phase2.md       # this file
├── phase-1-conclusion.md
├── design.md
└── ...
```

---

## Step 0 — Export .bin File

**Directory:** `export/`

**Goal:** Convert the Phase 1 `.pt` checkpoint into the `.bin` binary format the Rust engine will read. This is pure Python — no Rust needed. Start here.

**What to build:**

`export.py` — loads the checkpoint, applies ternary quantization to the embedding table (same absmean formula used in `eval.py`), packs all weights to 2 bits per value (4 weights per byte), writes the 24-byte header + packed weight data.

**Bit encoding:** `-1 → 0b10`, `0 → 0b00`, `+1 → 0b01`

**Weight ordering in .bin:**
```
[24B header]
[embedding table: 30,522 × 256, packed]
[layer 0: W_q, W_k, W_v, W_out, fc1, fc2, layernorm1, layernorm2]
[layer 1: W_q, W_k, W_v, W_out, fc1, fc2, layernorm1, layernorm2]
[final layernorm]
```

LayerNorm parameters (scale and bias) are stored as float32 — they're small (256 values each) and not ternary.

`verify.py` — reads the `.bin` back, unpacks the weights, loads the original `.pt` checkpoint, compares every weight matrix. This is your round-trip correctness test.

**What you should see:**
- A `.bin` file in `out/` — expect ~2.7MB for micro tier
- `verify.py` reports 0 mismatches on every weight matrix
- Header values match the model config

**Decision:** If verify passes with 0 mismatches, proceed. If any weight matrix diverges, debug the packing/unpacking logic before touching Rust.

---

## Step 1 — Rust + Wasm Hello World

**Directory:** `engine/`

**Goal:** Prove the Rust → Wasm → Node.js pipeline works end-to-end. No model code. Just a function that takes a string and returns a number.

**What to build:**

A minimal Cargo project:

```toml
# engine/Cargo.toml
[package]
name = "tern-engine"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
```

```rust
// engine/src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn hello(input: &str) -> usize {
    input.len()
}
```

Build and call from Node.js:

```bash
# Install toolchain (one-time)
rustup target add wasm32-unknown-unknown
cargo install wasm-pack

# Build
cd engine
wasm-pack build --target nodejs --release
```

```js
// quick test
const { hello } = require('./engine/pkg');
console.log(hello("reset my password"));  // → 17
```

**What you should see:**
- `wasm-pack build` succeeds with no errors
- `pkg/` directory appears with `.wasm` file + JS bindings
- Node.js call returns the correct string length

**Decision:** If this works, the toolchain is set up. If `wasm-pack` fails to install or build, fix the environment before adding any complexity.

---

## Step 2 — Tokenizer in Wasm

**Directory:** `engine/src/tokenizer.rs`

**Goal:** Get BERT WordPiece tokenization running inside Wasm. This is the first real risk — the HuggingFace tokenizers crate may or may not compile to `wasm32-unknown-unknown`.

**The decision tree (from Phase 2 spec):**

```
Try compiling HF tokenizers crate to wasm32-unknown-unknown
    │
    ├── Compiles? → Use it (Option A). Parity is structural.
    │
    └── Linker errors? → Write custom Rust WordPiece (Option B, ~150 lines)
```

**Option A — try first:**

```toml
# Add to engine/Cargo.toml
tokenizers = { version = "0.19", default-features = false, features = ["unstable_wasm"] }
```

```rust
// engine/src/tokenizer.rs
use tokenizers::Tokenizer;

static TOKENIZER_BYTES: &[u8] = include_bytes!("../assets/tokenizer.json");

pub fn tokenize(text: &str) -> Vec<u32> {
    let tokenizer = Tokenizer::from_bytes(TOKENIZER_BYTES).unwrap();
    let encoding = tokenizer.encode(text, true).unwrap();
    encoding.get_ids().to_vec()
}
```

Wire it into `lib.rs` and export via `wasm_bindgen`. Call from Node.js, print the token IDs.

**Option B — custom Rust WordPiece (fallback):**

If Option A fails to compile (most likely cause: `rayon` linker errors on `wasm32-unknown-unknown`), implement WordPiece directly. The algorithm is well-specified and the full BERT variant is ~150 lines of Rust.

**Algorithm:**

```
1. Unicode normalization
   - NFD form (decompose accented characters)
   - Lowercase
   - Strip accents (remove combining diacritical marks)

2. Whitespace + punctuation splitting
   - Break input into word tokens at whitespace
   - Split punctuation into separate tokens

3. For each word: greedy longest-match subword lookup
   - Try to match the full word against the vocab HashMap
   - If no match, try the longest prefix that exists in vocab
   - For the remainder, prefix with "##" and recurse
   - Example: "playing" → ["play", "##ing"]

4. Emit [UNK] (id=100) for words with no subword coverage at all

5. Prepend [CLS] (id=101), append [SEP] (id=102)

6. Pad with [PAD] (id=0) or truncate to max_seq_len=128
```

**Vocab loading:**

```rust
// engine/src/tokenizer.rs (sketch)
use std::collections::HashMap;
use std::sync::OnceLock;

static TOKENIZER_BYTES: &[u8] = include_bytes!("../assets/tokenizer.json");
static VOCAB: OnceLock<HashMap<String, u32>> = OnceLock::new();

fn get_vocab() -> &'static HashMap<String, u32> {
    VOCAB.get_or_init(|| {
        // Parse tokenizer.json (just the "vocab" field — ignore the rest)
        // Build HashMap<String, u32> from token → id
        // ...
    })
}
```

The same `tokenizer.json` is loaded via `include_bytes!()` — same vocab as Option A. Only the algorithm implementation differs. No regex library, no Unicode crate beyond what's needed for normalization (`unicode-normalization` is small and Wasm-compatible).

**Trade-offs vs Option A:**

| | Option A (HF crate) | Option B (custom) |
|---|---|---|
| Code to write | Minimal glue | ~150 lines |
| Wasm binary size | ~200–300KB | ~20KB |
| Dependencies | Heavy (regex, serde, rayon) | Zero non-standard |
| Parity confidence | Structural (same Rust core as Python) | Must be tested exhaustively |
| Parity test requirement | Spot-check 50 strings | Full 500-string test (hard gate) |

**Why Option B is the fallback, not the default:**

Option A gives **structural parity** — it's the literal same Rust code Python uses, just compiled to a different target. If it compiles, near-zero risk of token divergence.

Option B reimplements the algorithm. You have to prove via testing that your implementation matches the reference exactly, including edge cases:

- Unicode normalization differences (NFD vs NFC, accent stripping rules)
- Punctuation splitting (BERT has specific handling for CJK characters and accents)
- `[UNK]` emission logic for out-of-vocabulary subwords
- Empty input, single-character input, very long input

**Parity test:**

Run from `bridge/parity_check.py` — tokenize a corpus in both Python and Wasm, compare token ID sequences exactly.

| Option | Test corpus | Threshold |
|---|---|---|
| A (HF crate) | 50 strings spot-check | 100% match |
| B (custom) | 500 strings full corpus | 100% match — hard gate |

The 500-string corpus for Option B should include: natural language queries, developer error messages, camelCase and snake_case identifiers, mixed punctuation, edge cases (empty string, very long input, unicode characters).

**What you should see:**
- `wasm_bindgen` export `tokenize("reset my password")` returns `[101, 25801, 2026, 4957, 102, 0, 0, ...]`
- Same IDs as Python `AutoTokenizer.from_pretrained("bert-base-uncased")`
- Parity test: 100% match

**Decision:** If parity passes, proceed. If Option A has linker errors, switch to Option B and retest. If Option B has parity failures on ASCII text, debug before proceeding — this is a hard gate. Failures on unicode-only inputs are fixable by adjusting normalization rules; failures on plain ASCII indicate a fundamental implementation bug.

---

## Step 3 — Weight Loading

**Directory:** `engine/src/model.rs`

**Goal:** Read the `.bin` file from Step 0, parse the header, unpack ternary weights into usable arrays. No inference yet — just prove the data gets in correctly.

**What to build:**

```rust
// engine/src/model.rs

// Parse 24-byte header → ModelConfig struct
// Unpack 2-bit packed weights → Vec<f32> (expand -1/0/+1 to float for now)
// Read LayerNorm float32 params directly
// Expose weight arrays for inference.rs to consume
```

**Verification:** Write a `wasm_bindgen` export that loads the `.bin`, unpacks the embedding table, and returns the first row's values. Compare against Python:

```python
# In Python
ckpt = torch.load("runs/micro-qat-150k-100ep_ep100.pt")
emb_row_1 = ckpt["model_state"]["embedding.weight"][1]
# Apply ternary quantization
scale = emb_row_1.abs().mean()
ternary = torch.sign(emb_row_1) * (emb_row_1.abs() > 0.5 * scale)
print(ternary[:10])
```

The Rust unpacked values should match exactly.

**What you should see:**
- Header parsed correctly (d_model=256, n_layers=2, vocab_size=30522)
- Embedding row 1 unpacked values match Python ternary output
- All weight matrices have expected shapes

**Decision:** If unpacked weights match Python, the export→load pipeline is correct. Proceed to inference.

---

## Step 4 — Inference Engine

**Directory:** `engine/src/inference.rs`

**Goal:** Implement the forward pass — the actual computation graph from token IDs to embedding vector. This is the core engineering piece.

**Build incrementally, testing each piece against Python.**

### Testing approach — Python ↔ Rust symmetry

Every sub-step below must be verified by running the same operation in both Python and Rust on the same input, then comparing outputs. This catches bugs at the smallest possible scope — if 4c (attention) disagrees, you know it's not an embedding or normalization issue because 4a and 4b already passed.

Create a Python helper script (`export/forward_reference.py`) that loads the hardened checkpoint and dumps intermediate values for a known input string, e.g. `"reset my password"`:

```python
# For each sub-step, save the intermediate tensor:
#   ref/embedding_output.npy     — after embedding lookup (128 × 256)
#   ref/norm1_output.npy         — after first layer norm
#   ref/attention_output.npy     — after attention + residual
#   ref/ffn_output.npy           — after FFN + residual
#   ref/pooled_output.npy        — after mean pool (256,)
#   ref/final_output.npy         — after L2 normalize (256,)
```

The Rust engine loads the same `.bin`, processes the same token IDs, and at each sub-step compares its output against the corresponding `.npy` file. This is the symmetry test — if the numbers match at every stage, the engine is correct by construction.

**Tolerance:** ternary matmul should match exactly (integer add/sub, no float rounding). Float operations (layer norm, softmax, GELU) should match to ~1e-5 (float32 precision). The full forward pass accumulates rounding error across layers — final output should match to ~1e-4.

### 4a. Embedding lookup

```
token IDs [u32; 128] → look up rows from embedding table → [f32; 128 × 256]
```

**Python reference:** tokenize "reset my password", look up each row from the hardened embedding table, save as `ref/embedding_output.npy`.

**Rust test:** unpack the same rows from the `.bin`, compare. Should match exactly — this is a table lookup, no arithmetic.

### 4b. Layer norm

```
input [f32; 256] → normalize → [f32; 256]
```

**Python reference:** apply `LayerNorm` to the embedding output from 4a using the checkpoint's norm weights/bias, save result.

**Rust test:** implement `layer_norm(input, weight, bias)`, compare output. Must match to ~1e-5.

### 4c. Attention

```
input [f32; 128 × 256]
→ Q = ternary_matmul(input, W_q)    // additions/subtractions only
→ K = ternary_matmul(input, W_k)
→ V = ternary_matmul(input, W_v)
→ scores = Q @ K.T / sqrt(d_head)
→ mask padding positions
→ softmax(scores) @ V
→ output = ternary_matmul(attn_out, W_out) + bias
```

The ternary matmul is the key operation — no float multiplication, just accumulate additions and subtractions based on weight values {-1, 0, +1}.

**Python reference:** run one attention layer on the norm'd output from 4b, save Q, K, V, attention scores, and final attention output separately. Having intermediates means if the final output disagrees, you can pinpoint whether it's the matmul, the softmax, or the score scaling.

**Rust test:** compare each intermediate. Ternary matmul results should match exactly. Softmax and score scaling should match to ~1e-5.

### 4d. Feed-forward

```
input [f32; 256]
→ ternary_matmul(input, fc1) + bias → GELU → [f32; 1024]
→ ternary_matmul(hidden, fc2) + bias → [f32; 256]
```

**Python reference:** run FFN on the attention output from 4c, save pre-GELU and post-GELU intermediates.

**Rust test:** compare. Ternary matmul exact, GELU to ~1e-5.

### 4e. Mean pool + normalize

```
sequence [f32; 128 × 256] + attention_mask
→ mean over non-padding positions → [f32; 256]
→ L2 normalize → [f32; 256]   (unit vector)
```

**Python reference:** pool and normalize the full forward output, save both the pre-normalize pooled vector and the final normalized output.

**Rust test:** compare. Should match to ~1e-5.

### 4f. Full forward pass

Wire 4a–4e together for both layers. Export `embed(text: &str) -> Vec<f32>` via `wasm_bindgen`.

**Final symmetry test:** run `embed("reset my password")` in Rust, compare the 256-dim output vector against `ref/final_output.npy`. Should match to ~1e-4 (accumulated rounding across all layers).

**Also test a second string** (e.g. `"webpack config not working"`) to confirm the result isn't accidentally correct for just one input.

**What you should see:**
- `embed("reset my password")` returns a 256-dim float32 vector
- The vector is L2-normalized (length ≈ 1.0)
- Output matches Python reference to within ~1e-4
- All intermediate checkpoints (4a–4e) match individually

**Decision:** If the full forward pass output matches Python within tolerance, the engine is correct. Proceed to integration.

---

## Step 5 — JS Bridge + Regression Test

**Directory:** `bridge/`

**Goal:** Wire up the Node.js wrapper and validate that Wasm inference matches Phase 1 eval quality.

**What to build:**

`index.js` — thin wrapper: `init()`, `embed(text)`, `similarity(a, b)`.

`regression_test.js` — runs the Phase 1 eval corpus through the Wasm engine:
- Task 1: Teacher alignment (cosine sim on held-out queries)
- Task 3: Recall@3 on general + tech pairs (from eval.py's hardcoded corpus)
- Compare results against Phase 1 Python eval numbers

**Acceptable regression:** < 0.02 drop on any metric. A larger drop indicates a bug in the engine, not a model quality issue.

**Sanity check:**

```js
const { init, similarity } = require('./index');
await init();

const a = similarity("reset my password", "I forgot my password");
const b = similarity("reset my password", "quarterly earnings");
console.log(a, b);  // a should be >> b
```

**What you should see:**
- `similarity()` returns values in [0, 1]
- Semantically similar pairs score high, unrelated pairs score low
- Regression test metrics within 0.02 of Phase 1 Python eval

**Decision:** If regression passes, Phase 2 is complete. The full pipeline works: training → export → Wasm engine → JS API → correct embeddings.

---

## Summary

| Step | Directory | Language | Risk | Estimated effort |
|---|---|---|---|---|
| 0. Export .bin | `export/` | Python | Low | Half day |
| 1. Rust + Wasm hello world | `engine/` | Rust | Low | Half day |
| 2. Tokenizer in Wasm | `engine/` | Rust | **Medium** | 1–2 days |
| 3. Weight loading | `engine/` | Rust | Low | Half day |
| 4. Inference engine | `engine/` | Rust | **Medium** | 2–3 days |
| 5. JS bridge + regression | `bridge/` | JS + Python | Low | Half day |

Steps 0 and 1 are learning ramps with no risk. Step 2 is the first real unknown (tokenizer compilation). Step 4 is the most code but straightforward engineering against a known spec. Each step is independently verifiable before moving on.
