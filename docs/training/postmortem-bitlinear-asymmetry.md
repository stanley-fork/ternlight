# Postmortem: BitLinear Forward-Pass Asymmetry

> Engine inference math diverged from training/eval math. Caught by the
> regression test against Phase 1 baselines. Took ~2 days of investigation
> across two sessions to diagnose and fix.

---

## TL;DR

The Wasm engine was computing a **simplified approximation** of the trained model, not the trained model itself. The simplification preserved coarse semantic structure (Task 3 retrieval still worked) but degraded fine semantic alignment (Task 2 STS-B AUC dropped 0.10, mean teacher-cosine-sim dropped from 0.81 to 0.58).

The bug was hidden because our parity test (`test_embed.js`) compared the engine against a **hand-written Python reference (`dump_embed.py`)** that had been written to mirror the engine's simplified math. Both sides were wrong in the same way, so the test passed at 1e-7 precision — while silently being a different model from the one `eval.py` ran.

Root cause: the BitLinear forward pass at training/eval time runs five operations the engine didn't replicate:

1. Internal parameter-less LayerNorm before each ternary matmul
2. Per-token activation int8 quantization (`round_clamp`)
3. Output rescaling by `1/(w_scale·x_scale)` after each matmul
4. AbsMedian (not AbsMean) weight quantization
5. Exact (erf-based) GELU instead of tanh approximation

Fix: implemented full BitLinear forward in `inference.rs::bitlinear_forward()`, added per-matrix `w_scale` to the .bin format (v2). After fix: engine matches trained model to 4e-4 max diff, all three regression tasks PASS at Phase 1 quality.

---

## Timeline

- **2026-04-26** — Phase 2 Step 5 regression test built. First run shows large gaps on Tasks 1 and 2 vs Phase 1 baselines. Initially attributed to the missing output projection layer (which we'd dropped at export per the original Phase 1 design).
- **2026-04-26 evening** — Implemented Option A1 from `tern-future-work.md` Section 14: ship the f32 projection layer. `test_embed.js` passes at 2e-7 against `dump_embed.py`. Regression test improves but Task 2 still ~0.10 below Phase 1 baseline.
- **2026-05-02** — Switched Task 1 metric to mean per-query cosine sim (matching `eval.py`'s `task1_teacher_alignment`). Result: 0.5841 vs Phase 1's 0.812. Confirmed gap is real and substantial, not metric mismatch.
- **2026-05-02** — Ran `eval.py` directly in Python: `Task 1 = 0.8117` (matches Phase 1's 0.812 exactly). Confirmed the trained checkpoint reproduces Phase 1 quality when run through the *real* model — engine is the divergence source.
- **2026-05-02** — Inspected `refs/bitlinear/bitlinear/bitlinear.py` and `refs/bitlinear/bitlinear/measures.py`. Found that `BitLinear.forward()` does substantially more than ternary matmul + bias.
- **2026-05-02** — Rewrote `dump_embed.py` to call `model_scratch.py` directly (no manual reimplementation). Re-ran `test_embed.js`: cosine sim 0.7054, 376/384 elements fail. Diagnostic confirmed: engine ≠ trained model.
- **2026-05-02** — Option A (AbsMedian export): cosine 0.7054 → 0.7898.
- **2026-05-02** — Option B (full BitLinear forward): cosine 0.7898 → 0.99999591.
- **2026-05-03** — Full regression test re-run: all three tasks PASS, Task 1 mean cosine sim 0.8137 (Phase 1: 0.8120).

---

## Symptom

`bridge/regression_test.js` against Phase 1 baselines, before any fix:

| Metric | Wasm | Phase 1 | Δ |
|---|---|---|---|
| Task 1 mean cos sim (vs teacher) | 0.5841 | 0.8120 | -0.228 |
| Task 2 STS-B AUC | 0.7444 | 0.8390 | -0.095 |
| Task 2 STS-B Spearman | 0.6090 | 0.7200 | -0.111 |
| Task 3 General R@3 | 0.75 | 0.75 | 0.00 |
| Task 3 Tech R@3 | 0.95 | 1.00 | -0.05 |

The pattern is informative:

- **Task 3 (coarse retrieval) mostly survives** — coarse topical separation tolerates magnitude errors because it only needs the right *winner* among 20 candidates, not the right absolute scores.
- **Task 2 (STS-B human similarity) regresses ~10 points** — fine semantic distinctions need accurate magnitudes, which the simplified engine doesn't produce.
- **Task 1 (per-query teacher alignment) regresses 23 points** — direct cosine alignment is the most sensitive metric. It immediately reflects any forward-pass divergence.

This pattern is a fingerprint: "engine produces *some* model, but not the trained one." Random noise would fail Task 3 too. A fundamentally broken engine (e.g., wrong indexing) would fail much harder. We were in the precise middle: structurally similar but numerically wrong.

---

## Why parity tests passed despite the bug

The Phase 2 build process layered correctness checks at each step. We had:

- `test_weights.js` — verified embedding row unpacking matches Python
- `test_layer_norm.js` — verified LN matches PyTorch
- `test_qkv.js` — verified Q/K/V projections match Python
- `test_attention.js` — verified attention matches Python
- `test_attention_block.js` — verified full attention block matches Python
- `test_embed.js` — verified end-to-end forward pass matches Python

Every one of these passed at ~1e-7 precision. So how did we ship a divergent forward pass?

**Each Python reference (`dump_*.py`) was hand-written to mirror the engine, not to mirror the actual training model.** When we wrote `dump_attention_block.py`, we implemented the same simplified math the engine implemented:

```python
# dump_embed.py — what we wrote (matches the engine):
Q = quantize_ternary_absmean(W_q) → matmul → + bias

# bitlinear.py — what eval.py actually executes:
x_norm = LayerNorm(x)                              # internal LN
x_scale = 128 / max(|x_norm|).clamp(min=eps)      # per-token AbsMax
x_quant = (x_norm * x_scale).round().clamp(-128, 127)
w_scale = 1 / median(|weight|).clamp(min=eps)     # AbsMedian
w_quant = (weight * w_scale).round().clamp(-1, 1)
y = (x_quant @ w_quant.T + bias) / (w_scale * x_scale)
```

The engine matched `dump_embed.py` to 1e-7. `dump_embed.py` did not match `eval.py`. So all parity tests passed while the engine was running a different model entirely.

**This is the core lesson.** A parity test against a hand-written reference is only as correct as the reference. We never validated that the reference matched the *real* training-time forward pass.

---

## Root cause — what BitLinear actually does

Reference: `refs/bitlinear/bitlinear/bitlinear.py:65-79`

Each `BitLinear` layer in the trained model — twelve of them across the two transformer layers (Q, K, V, W_out, fc1, fc2) — runs this forward pass:

```python
def forward(self, x):
    x_norm  = self.norm(x)                              # internal LayerNorm (no affine)
    x_scale = scale(x_norm, (-128, 127), AbsMax(), True, eps)
    x_quant = round_clamp(x_norm * x_scale, [-128, 127])

    w_scale = scale(self.weight, (-1, 1), AbsMedian(), False, eps)
    w_quant = round_clamp(self.weight * w_scale, [-1, 1])

    y_quant = self.kernel(x_quant, w_quant, self.bias)  # F.linear: x @ w.T + b
    y       = y_quant / (w_scale * x_scale)
    return y
```

Where:

- `self.norm = LayerNorm(in_features)` — `torch.layer_norm(input, [in_features])` with **no learnable parameters** (just normalize-only, eps=1e-5)
- `scale()` returns `max(|range|) / measure(input).clamp(min=eps)` — for weights, that's `1 / median(|weight|)`
- `round_clamp(x, [a, b])` is `x.round().clamp(a, b)` at inference (the gradient form is for backprop)
- `kernel = TorchLinear` — standard `F.linear(input, weight, bias)` = `x @ w.T + bias`

What our engine was doing instead:

```rust
// inference.rs (before fix):
fn project(input, layer_idx, which) -> Vec<f32> {
    let weight = read_ternary(...);                     // W ∈ {-1, 0, +1}
    ternary_matmul(input, &weight, ...)                 // input @ W.T
}
// then add_bias_per_row(...) for layers with bias
```

Five differences from the real BitLinear:

| # | Step | Real BitLinear | Engine (before fix) | Why it matters |
|---|---|---|---|---|
| 1 | Pre-matmul norm | Parameter-less LN per BitLinear | None (used only the explicit `norm1`/`norm2`) | Two LNs in a row vs one — different distributions enter the matmul |
| 2 | Activation quantization | int8 round_clamp | None (f32 throughout) | Activations have different precision/distribution |
| 3 | Weight quantization formula | AbsMedian round_clamp | AbsMean + sign threshold | Different weights snap to zero, different ternary patterns |
| 4 | Output rescaling | `÷ (w_scale · x_scale)` | None | **Magnitudes off by a learned scale factor** — the dominant error |
| 5 | GELU | Exact (erf-based) | tanh approximation | ~1e-4 drift per application, compounds across layers |

(4) is the largest single contributor. The matmul output magnitudes feed into nonlinearities (GELU, softmax, the final projection) that are *not* scale-invariant. Even though the explicit `norm1`/`norm2` partially absorbs scale errors at the next layer's input, the final projection layer has no such absorber — wrong magnitudes there propagate directly to the L2-normalized output direction.

---

## The fix

Two stages, applied independently to verify their contributions.

### Option A — AbsMedian weight quantization (export.py only)

Single change in `export.py`: use `quantize_ternary_absmedian()` (round-clamp formula matching BitLinear's runtime) for BitLinear weights. Embedding stays AbsMean (matches `eval.py`'s `quantize_embedding_to_ternary`).

```python
def quantize_ternary_absmedian(tensor):
    eps = 1e-5
    w_scale = 1.0 / tensor.abs().median().clamp(min=eps).item()
    quantized = (tensor * w_scale).round().clamp(-1.0, 1.0)
    return quantized, w_scale  # also return w_scale for the .bin
```

**Result:** cosine sim 0.7054 → 0.7898. Closed ~30% of the gap. Confirmed quantization detail matters but is not the dominant fix.

### Option B — full BitLinear forward in the engine

Implemented in `inference.rs::bitlinear_forward()`. Replaces every `ternary_matmul + add_bias` call inside `apply_attention_block` and `apply_ffn_block`.

```rust
pub fn bitlinear_forward(input, w_quant, w_scale, bias, seq_len, in_dim, out_dim) -> Vec<f32> {
    // 1. Internal LayerNorm (no affine)
    let x_norm = layer_norm_no_affine_seq(input, seq_len, in_dim);

    // 2. Per-token activation scale and quantization
    for t in 0..seq_len {
        let max_abs = x_norm[t*in_dim..(t+1)*in_dim].iter().map(|v| v.abs()).fold(0.0, f32::max);
        let x_scale = 128.0 / max_abs.max(1e-5);
        for j in 0..in_dim {
            x_quant[t*in_dim + j] = (x_norm[t*in_dim + j] * x_scale).round().clamp(-128.0, 127.0);
        }
        x_scales[t] = x_scale;
    }

    // 3. Ternary matmul + bias
    let mut y_quant = ternary_matmul(&x_quant, w_quant, seq_len, in_dim, out_dim);
    if let Some(b) = bias { add_bias_per_row(&mut y_quant, b, seq_len, out_dim); }

    // 4. Per-token rescale
    for t in 0..seq_len {
        let inv = 1.0 / (w_scale * x_scales[t]);
        for d in 0..out_dim { y_quant[t*out_dim + d] *= inv; }
    }
    y_quant
}
```

`.bin` format bumped to v2 to add per-BitLinear `w_scale` (12 floats, 48 bytes total). `model.rs` parses the new layout; engine asserts `format_version == 2` so old binaries fail loudly.

**Result:** cosine sim 0.7898 → 0.99999591. All 384/384 elements within 1e-3 tolerance. Engine now mathematically equivalent to the trained model.

---

## Results

### Engine vs Python reference (`test_embed.js`)

| Stage | Cosine sim (engine vs real model) | Max element diff | Failed elements |
|---|---|---|---|
| Before any fix | 0.7054 | 0.118 | 376/384 |
| After Option A | 0.7898 | 0.104 | 368/384 |
| After Option A + B | **0.99999591** | **4.119e-4** | **0/384** PASS |

### Downstream eval (`regression_test.js`)

| Metric | Before fix | After A+B | Phase 1 ref |
|---|---|---|---|
| Task 1 mean cos sim | 0.5841 | **0.8137** | 0.8120 |
| Task 2 AUC | 0.7444 | **0.8525** | 0.8390 |
| Task 2 Spearman | 0.6090 | 0.7066 | 0.7200 |
| Task 3 General R@3 | 0.75 | 0.75 | 0.75 |
| Task 3 Tech R@3 | 0.95 | 1.00 | 1.00 |

All three regression tasks PASS. Task 1 mean cosine sim is *slightly above* the Phase 1 reported number, well within sample variance. Task 2 AUC also exceeds Phase 1 baseline (likely sample variance from the 200-pair STS-B subsample).

The remaining ~4e-4 max diff is the tanh-approx GELU vs exact GELU — within tolerance, optional polish.

---

## Lessons

### 1. Parity tests against hand-written references are circular

A reference implementation written to "mirror" the engine doesn't validate the engine — it validates internal consistency between two pieces of *your own* code. To validate against the real system, the reference must be the real system, or independently derived from it.

**Going forward:** Python references for the engine should call into `model_scratch.py` (or whatever the actual training-time model is), not hand-roll the math. The post-fix `dump_embed.py` does this in 30 lines.

### 2. Coarse semantic tasks are dangerously forgiving

Task 3 (Recall@3 retrieval) showed *zero* regression even when the engine was producing a fundamentally different model. The semantic ordering of 20 candidates per query was preserved by sheer redundancy in the ternary weights. Had we shipped without Tasks 1 and 2 in the regression suite, the engine would have looked fine.

**Going forward:** any future eval suite must include both fine-grained metrics (mean cosine sim, STS-B Spearman) and coarse-grained ones (R@K). Coarse metrics passing doesn't mean the engine is correct.

### 3. Library defaults can be load-bearing

`refs/bitlinear/bitlinear/bitlinear.py` configures BitLinear with sensible defaults: internal LayerNorm, AbsMedian weights, AbsMax activations, int8 quantization, scale rescaling. These defaults aren't called out anywhere in the README — they're just *what BitLinear does*. We never inspected them because we assumed BitLinear was "ternary matmul + bias." It is not.

**Going forward:** when integrating any ML library at the inference level, read the actual `forward()` of each layer being used, not just the README. If you're going to reimplement it elsewhere (Wasm, mobile, edge), trace through every operation including any non-obvious normalization, quantization, or rescaling.

### 4. The diagnostic ladder matters

The fix was reached by climbing this ladder, in order:

1. Believe the regression numbers (don't dismiss as noise)
2. Rule out baseline mismatch (run `eval.py` directly to confirm Phase 1 reproduces)
3. Rule out engine-vs-reference parity (re-derive the reference from the actual model)
4. With both confirmed real, find the divergence (read the library source)
5. Fix incrementally (Option A first to size the contribution, then Option B for the rest)

Skipping any step would have led to either (a) months of guessing or (b) blindly implementing fixes without knowing which mattered. The ladder is reusable for any future "outputs look weird" investigation.

### 5. Format versioning paid off

The `.bin` header had a `format_version` field from day one. When we added `w_scale` per BitLinear, bumping to v2 and asserting it in the engine made the upgrade safe — old binaries fail loudly with a clear message instead of silently misaligning offsets and producing garbage. Cheap insurance, paid off the first time it was needed.

---

## Impact on future work

### Immediate

- `tern-future-work.md` Section 14 ("Projection Head Strategy") is no longer the bottleneck for downstream quality. The projection layer is shipped (Option A1), and now the encoder body underneath it actually matches training.
- The v0.1 polish workstream (perf + packaging) can proceed against a correct engine. Performance optimizations (SIMD, weight unpacking caching) won't be working around buggy math.

### New optimization paths unlocked

The full BitLinear forward as implemented does activation quantization to int8 — but currently keeps activations as f32 representing those int8 values. This unlocks `tern-future-work.md` Section 6 (SIMD): **int8 activations × ternary weights = pure integer matmul, no float multiplies**. Modern Wasm SIMD has 16 int8 lanes per `v128` op, potentially 4-8× speedup on the dominant inner loop.

### Process changes

- New parity tests should call into the real model, not hand-rolled code.
- Any future Phase 2-style re-implementation effort (different target, different inference framework) should add a "diagnostic ladder" step in its milestones doc, before the parity tests are written.

---

## Files changed (this fix)

- `tern-distill-prototype/export/export.py` — added `quantize_ternary_absmedian`, captures `w_scale`, writes per-BitLinear w_scale to .bin, bumped FORMAT_VERSION to 2
- `tern-distill-prototype/export/dump_embed.py` — rewrote to use `model_scratch.py` + BitLinear directly (was hand-written mirror of engine)
- `tern-distill-prototype/engine/src/model.rs` — added `read_f32`, expanded `LayerOffsets` with 6 new `*_scale` fields, layout walk includes w_scale bytes, `get_config` asserts format_version=2
- `tern-distill-prototype/engine/src/inference.rs` — added `layer_norm_no_affine_seq`, added `bitlinear_forward` (full BitLinear math), refactored `apply_attention_block` and `apply_ffn_block` to use it
- `tern-distill-prototype/bridge/regression_test.js` — switched Task 1 metric to mean per-query cosine sim (matches `eval.py`'s reported 0.812 baseline) with within-batch Spearman as diagnostic

No training code, no checkpoint changes, no model architecture changes. Pure inference-side fix.
