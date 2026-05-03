# Model Sizing: Target Architecture for @tern/semantic

## Budget Decomposition

The 5MB total package constraint breaks down across three components:

| Component | Estimated Size | Notes |
|---|---|---|
| Wasm engine | ~300–600KB | Rust/Zig compiled, stripped |
| JS wrapper + tokenizer | ~50–100KB | Zero-dependency |
| **Model weights** | **~2.4–4MB** | Effective model budget |

Ternary weights pack at ~1.58 bits per parameter. With byte-alignment and packing overhead, a practical working figure is **~2 bits per parameter**, meaning:

- 2.4MB budget → ~9.6M parameters
- 4.0MB budget → ~16M parameters

The scoping doc's 10–15M parameter target sits in the middle of this range and is architecturally sound.

---

## Why Parameter Count Alone Is Misleading

For short-query semantic similarity (64–128 tokens), what matters more than raw count is **how parameters are distributed**. Two models with identical counts can have very different capability profiles depending on vocabulary size and model width.

### Vocab Size: The Hidden Budget Killer

A 30,000-token vocabulary at `d_model=128` costs:

```
30,000 × 128 × 2 bits = ~960KB — just for embeddings
```

This is why the scoping doc's 10,000-token vocabulary target is a hard architectural constraint, not a preference.

### d_model Drives Everything Else

The FFN is typically 4× d_model wide. At `d_model=64` (the scoping doc's current student dimension), the model is likely too narrow for meaningful semantic nuance. At `d_model=256`, each layer has substantially more representational bandwidth per token.

**Key insight:** Ternary quantization hurts narrow models more than wide ones. A 2-layer model at `d_model=256` with ternary weights will likely outperform a 3-layer model at `d_model=128`, because width compensates for the expressivity loss of {-1, 0, 1} weights.

---

## Recommended Architecture

| Hyperparameter | Target | Rationale |
|---|---|---|
| **d_model** | 256 | 64 is too narrow for semantic nuance; 256 gives headroom without blowing budget |
| **n_layers** | 2 | Floor for meaningful composition; matches scoping doc's 2-layer design |
| **n_heads** | 8 | Standard; gives d_k=32 per head at d_model=256 |
| **FFN width** | 1024 (4× d_model) | Standard ratio; do not compress this |
| **Vocab size** | 10,000 | BPE on English+code corpus; matches scoping doc |
| **Context length** | 128 tokens | Matches FAQ/intent routing use case |
| **Total params** | ~7M | Leaves deliberate headroom |

### Parameter Count Breakdown

```
Embedding table:  10,000 × 256                = 2,560,000 params  (~640KB packed)
Attention × 2:    2 × 4 × (256 × 256)         = 524,288 params    (~131KB packed)
FFN × 2:          2 × 2 × (256 × 1024)        = 1,048,576 params  (~262KB packed)
Layer norms, etc:                              = ~50,000 params    (~13KB packed)
─────────────────────────────────────────────────────────────────────────────────
Total:                                         ≈ 7M params         ≈ 1.75MB packed
```

This leaves **~2MB+ for the Wasm engine and JS wrapper**, comfortably under the 5MB ceiling with room to spare.

---

## Sizing Scenarios

| Scenario | d_model | Layers | Vocab | Params | Packed Size | Notes |
|---|---|---|---|---|---|---|
| **Conservative (recommended)** | 256 | 2 | 10k | ~7M | ~1.75MB | Strong baseline; significant headroom |
| **Extended** | 384 | 2 | 12k | ~12M | ~3.0MB | More robust distillation target; still fits |
| **At-limit** | 512 | 3 | 12k | ~25M | ~6.25MB | Exceeds budget — avoid |
| **Scoping doc baseline** | 64 | 2 | 10k | ~2.5M | ~625KB | Too narrow; likely underperforms |

---

## Recommendation Summary

**Anchor at ~7M parameters (d_model=256, 2-layer) rather than targeting the full 10–15M range.** Reasons:

1. **Task specificity beats scale.** This model matches short strings against known intents — a narrow task where a well-distilled 7M-param model trained on focused data will outperform a bloated 15M-param general model.

2. **Width beats depth for ternary models.** Expressivity loss from {-1, 0, 1} weights is better compensated by wider layers than by adding depth.

3. **Headroom has value.** The remaining budget keeps the Wasm engine and tokenizer unconstrained, and preserves flexibility for future architecture iterations without requiring a re-engineering of the packaging pipeline.

If the distillation results show the 7M model undershooting quality targets, scaling to d_model=384 / ~12M params is the natural next step before reconsidering depth or vocabulary expansion.
