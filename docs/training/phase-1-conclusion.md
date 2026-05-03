# POC Training Review — What Was Built and Why

> **Last updated:** 2026-03-29. This is a review sheet for catching up on the distillation training POC after time away. It covers the architecture, training approach, loss functions, measurements, and final results.

---

## The Goal

Train a tiny 2-layer transformer (the "student") to produce sentence embeddings that match a much larger model (the "teacher"). The student uses ternary weights {-1, 0, +1} so it can be packed to ~2 bits per weight and shipped in a Wasm binary under 5MB.

**Teacher:** `all-MiniLM-L6-v2` — 22M parameters, 384-dim output, pre-trained on massive data. We never modify it. It generates the target embeddings the student tries to match.

**Student:** 9.5M parameters, 256-dim internal, projected to 384-dim output. Trained from scratch via knowledge distillation.

---

## Model Architecture

```
input_ids  (batch, 128)                      ← integer token IDs
    │
    ▼
┌─────────────────────────────────────┐
│  Embedding Table                    │
│  30,522 × 256                       │  ← largest single block (~82% of params)
│  float32 during training            │     post-training quantized to ternary at export
│  padding_idx=0 → zero vector        │
└─────────────────────────────────────┘
    │
    ▼  (batch, 128, 256)
┌─────────────────────────────────────┐
│  Transformer Layer 1                │
│  ┌───────────────────────────────┐  │
│  │ LayerNorm                     │  │  ← Pre-LN: normalize before each sublayer
│  │ Multi-Head Self-Attention     │  │     4 heads × 64 dims each
│  │   W_q, W_k, W_v, W_out       │  │     ← these become BitLinear in QAT
│  │ + residual connection         │  │
│  ├───────────────────────────────┤  │
│  │ LayerNorm                     │  │
│  │ Feed-Forward Network          │  │     256 → 1024 → 256
│  │   fc1, fc2                    │  │     ← these become BitLinear in QAT
│  │ + residual connection         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼  (batch, 128, 256)
┌─────────────────────────────────────┐
│  Transformer Layer 2                │
│  (same structure as Layer 1)        │
└─────────────────────────────────────┘
    │
    ▼  (batch, 128, 256)
┌─────────────────────────────────────┐
│  LayerNorm                          │
│  Mean Pool (ignore padding tokens)  │  ← collapse sequence → one vector per sentence
└─────────────────────────────────────┘
    │
    ▼  (batch, 256)
┌─────────────────────────────────────┐
│  Projection  [256 → 384]           │  ← stays float32, bridges to teacher's space
│  L2 Normalize                       │  ← output lives on unit sphere
└─────────────────────────────────────┘
    │
    ▼  (batch, 384)                      ← compared against teacher embedding via cosine sim
```

### Key design choices in the architecture

**Pre-LN (LayerNorm before sublayer):** More stable for training than Post-LN (original BERT). Especially important under ternary quantization where gradient flow is already constrained.

**Mean pooling over tokens:** Averages all non-padding token representations into a single sentence vector. The attention_mask is used to exclude padding positions from the average.

**L2 normalization at output:** Forces all embeddings to length 1 (unit sphere). This makes cosine similarity equal to a dot product and keeps the loss in a predictable [0, 2] range.

**Projection stays float32:** The 256→384 linear layer bridges the student's internal space to the teacher's embedding space. It's excluded from BitLinear because it's sensitive to quantization noise — it's the interface between two different dimensional spaces.

### Two model files

- `model_scratch.py` — hand-rolled attention with explicit W_q, W_k, W_v, W_out as nn.Linear. Used for QAT because `replace_modules()` can find and replace each one individually.
- `model.py` — same architecture built on PyTorch's `nn.TransformerEncoderLayer`. Cleaner code but bundles Q/K/V into a single Parameter that `replace_modules()` can't reach. Used for the float32 baseline only.

Both expose the same `StudentEncoder` class with identical `forward(input_ids, attention_mask) → embedding` interface.

---

## Why a Float32 Baseline First (Milestone 2)

Before adding ternary quantization, we trained the student in pure float32 to establish a quality ceiling. This answers: "can this 2-layer architecture learn the teacher's embeddings at all, ignoring quantization?"

If the float32 baseline is poor, the architecture is too small — no amount of QAT tuning will fix that. If it's good, we know exactly how much quality the ternary constraint costs.

**Float32 baseline result (25k samples, 30 epochs):**

```
train/cosine_sim:  0.787
val/spearman:      0.562
```

This became the ceiling. The QAT model's target was ≥80% of 0.787 = 0.630.

---

## QAT — Quantization-Aware Training (Milestone 3)

QAT means the model trains knowing its weights will be quantized. During each forward pass, weights are projected to ternary {-1, 0, +1} using absmean scaling:

```
scale = mean(|weight|)
ternary = sign(weight) × (|weight| > 0.5 × scale)
```

But gradients still flow through the float32 "shadow" weights via the **Straight-Through Estimator (STE)** — the backward pass pretends the quantization step didn't happen, allowing the optimizer to make smooth updates to the underlying float32 values.

### BitLinear swap

The `schneiderkamplab/bitlinear` library provides a `replace_modules()` function that recursively walks the model and swaps every `nn.Linear` with a `BitLinear` layer. A regex filter excludes the output projection:

```python
replace_modules(model, match_name=r"^(?!.*projection)")
```

After the swap, the model has:
- **12 BitLinear layers** — W_q, W_k, W_v, W_out, fc1, fc2 × 2 transformer layers
- **1 float32 Linear** — the projection layer

### Warmup phase

The first 5 epochs run with `set_lambda_(model, 0.0)` — quantization is disabled, the model trains in pure float32. This lets the weights find a reasonable starting region before the ternary constraint is imposed. At epoch 6, `set_lambda_(model, 1.0)` activates QAT. A small loss spike is expected and normal.

---

## Loss Functions

### Distillation loss (primary, weight=1.0)

```python
loss = 1 - cosine_similarity(student_emb, teacher_emb)
```

Both embeddings are L2-normalized (unit vectors), so cosine similarity is just their dot product. The loss is 0 when they point in exactly the same direction, 1 when orthogonal, 2 when opposite.

This is the main training signal — "make your embedding point the same direction as the teacher's for every input."

### Contrastive loss (guardrail, weight=0.15)

```python
sim_matrix = student_emb @ student_emb.T    # pairwise cosine sim within batch
contrastive = sim_matrix.fill_diagonal_(0).pow(2).mean()
```

Penalizes high similarity between different sentences in the same batch. Without this, the model could "cheat" by mapping many inputs to the same region — each close to its teacher target but not distinguishable from each other at retrieval time.

### Variance loss (removed)

Originally included to keep all embedding dimensions active. It targeted `std = 1.0` per dimension, but our embeddings are L2-normalized — a 384-dim unit vector has per-dimension std of ~0.05 by geometric necessity. The loss was always active but could never be satisfied. Removed after diagnosing as a constant noise term with no useful gradient signal.

---

## What We Measure and Why

### During training (per epoch)

| Metric | What it means |
|---|---|
| `train/loss` | Total loss (distillation + contrastive). Lower = better. |
| `train/cosine_sim` | 1 - distillation_loss. How close student embeddings are to teacher targets. Higher = better. |
| `val/loss` | Same as train loss but on held-out 10% the model never trains on. Tracks generalization. |
| `val/spearman` | Spearman rank correlation between student and teacher pairwise similarities. Measures whether the student *ranks* sentence pairs the same way the teacher does, not just individual alignment. |
| `weights/zero_frac_avg` | Average fraction of weights that would snap to zero under ternary projection. 20–40% = healthy sparsity. 40–60% = watch. >60% + flat loss = collapse. |

**Train vs val gap:** If train loss keeps falling but val loss plateaus or rises, the model is memorizing training data instead of learning general patterns (overfitting). We never saw this — both tracked closely throughout.

**Spearman correlation:** This is closer to the real product task than cosine_sim. @tern needs to correctly rank "reset password" as more similar to "forgot password" than to "quarterly earnings." Spearman measures exactly this — do the student and teacher agree on which pairs are more similar?

### During eval (Milestone 5, on final checkpoint)

| Task | What it tests | Metric |
|---|---|---|
| Task 1: Teacher Alignment | Are individual embeddings close to the teacher's? | Mean cosine similarity on 2,000 held-out queries |
| Task 2: STS-B Ranking | Does the model rank sentence similarity correctly? | AUC on STS-B benchmark (human-labeled sentence pairs, scored 0–5) |
| Task 3: Recall@3 | Can the model retrieve the right answer from a corpus? | Fraction of queries where the correct match is in the top-3 nearest neighbors |

Task 3 runs on two corpora — **general** (password resets, order tracking, etc.) and **tech** (webpack errors, OAuth tokens, k8s pods). The gap between them reveals domain coverage issues.

### Embedding quantization at eval — simulating the shipped model

The saved `.pt` checkpoint contains **float32 embeddings** — QAT only quantizes the BitLinear layers during training, not the embedding table. To simulate what the shipped model will actually look like, `eval.py` applies post-training ternary quantization to the embedding table in-memory before running the eval tasks:

```python
scale = emb_weight.abs().mean()
emb_ternary = sign(emb_weight) * (abs(emb_weight) > 0.5 * scale)
```

This is the default behavior — `python eval.py` runs with ternary embedding. The `--no-quant-embedding` flag skips this step and runs eval on the float32 embedding for comparison only.

**This distinction matters because model sizing depends on it.** Without ternary embedding the model doesn't fit the target size:

```
Float32 embedding:  30,522 × 256 × 4 bytes  = ~31MB  ← doesn't ship
Ternary embedding:  30,522 × 256 × 2 bits   = ~1.95MB ← fits in budget
Full model packed:  embedding + attention/FFN = ~2.7MB
```

The embedding table is ~82% of total parameters. The entire product premise (< 5MB total package) depends on ternary embedding. The final GO decision (0.812 cosine_sim) was made **with ternary embedding active** — confirming that post-training quantization of the embedding is viable at this training scale.

**Eval runs comparison — embedding quantization impact:**

| Checkpoint | Embedding | Task 1 | Overall |
|---|---|---|---|
| POC (25k/30ep) | Ternary | 0.555 | NO-GO |
| POC (25k/30ep) | Float32 | 0.639 | MARGINAL |
| Scaled (150k/100ep) | **Ternary** | **0.812** | **GO** |

At POC scale, ternary embedding cost ~0.084 on cosine_sim. At scaled training, the model learned more quantization-robust representations — the GO result was achieved with ternary embedding, meaning the quality gap is no longer a blocking concern.

---

## Results — POC Progression

### WandB runs summary

| Run | Data | Epochs | cosine_sim | val/spearman | zero_frac |
|---|---|---|---|---|---|
| `micro-fp32-baseline` | 5k | 5 | 0.290 | 0.245 | — |
| `micro-fp32-baseline-30ep` | 25k | 30 | 0.787 | 0.562 | — |
| `micro-qat-warmup5` | 25k | 30 | 0.755 | 0.558 | 0.285 |
| `micro-qat-full-loss` | 25k | 30 | 0.753 | 0.559 | 0.285 |
| `micro-qat-150k-100ep` | 150k | 100 | **0.894** | **0.813** | 0.446 |

### Eval results — final scaled run (150k/100ep, ternary embedding)

| Task | Score | Threshold | Verdict |
|---|---|---|---|
| Task 1: Teacher Alignment | **0.812** | >0.75 pass | **PASS** |
| Task 2: STS-B AUC | **0.839** | >0.80 pass | **PASS** |
| Task 3: Recall@3 General | **0.75** | >0.70 pass | **PASS** |
| Task 3: Recall@3 Tech | **1.00** | >0.70 pass | **PASS** |
| **Overall** | | | **GO** |

### Key takeaway from results progression

1. The first run (5k/5ep) was too small to learn anything meaningful — cosine 0.29.
2. Scaling data to 25k and epochs to 30 was the breakthrough — cosine jumped to 0.79.
3. QAT retained ~96% of float32 quality (0.755 vs 0.787) — ternary weights work.
4. The contrastive loss term had negligible effect at small scale (0.013) — it's a guardrail for longer runs.
5. The final scaled run (150k/100ep) passed all eval tasks with ternary embedding — **the architecture is validated at d_model=256**.

---

## Architecture Decision: Confirmed

Based on the scaled run results:

- **d_model=256** — sufficient. No need for d_model=384.
- **Ternary embedding** — viable. Passes eval with post-training quantization.
- **2 transformer layers** — enough for the target use cases.
- **384-dim output** — matches teacher space, projection stays float32.

The architecture is frozen. The next phase is .bin export and Rust/Wasm engine.

---

## File Map

```
tern-distill-prototype/poc/
├── prepare.py          # Download MS MARCO, tokenize, teacher encode, save .pt cache
├── model.py            # StudentEncoder using PyTorch built-in TransformerEncoder
├── model_scratch.py    # StudentEncoder with hand-rolled attention (used for QAT)
├── train.py            # Float32 baseline training (Milestone 2)
├── train_qat.py        # QAT training with BitLinear + full loss (Milestones 3–4)
├── eval.py             # Eval suite: teacher alignment, STS-B, Recall@3 (Milestone 5)
├── cache/              # .pt cache files (gitignored)
└── runs/               # Checkpoints (gitignored)
```
