# @tern — Phase 1 Prototype: Distillation & Ternary Training

> **Scope:** This document covers only the training prototype — distillation from a teacher model into a ternary student, with eval defined as a simulated inference step using the same tokenizer. It does not cover the Wasm engine, JS bridge, or packaging pipeline.

---

## 1. Prototype Goals

The Phase 1 prototype must answer three questions before any further architecture investment is justified:

1. **Can a 2-layer ternary transformer, distilled from a floating-point teacher, produce semantically meaningful embeddings?**
2. **Does the BERT WordPiece tokenizer, as used via the HuggingFace `tokenizers` Python bindings, produce sufficient token quality for the student to learn?**
3. **Is the training process stable under ternary quantization constraints?**

These are go/no-go questions. If the answer to any of them is "no", the architecture needs to be revised before Phase 2 begins.

---

## 2. What This Prototype Is Not

- It does not produce a shipping `.bin` file
- It does not exercise the Wasm engine or JS bridge
- It does not validate package size, cold-start latency, or RAM footprint
- It runs entirely in Python (PyTorch) — no JS involved in Phase 1

---

## 3. The Tokenizer (Shared Contract)

Training and inference must produce **bit-for-bit identical token ID sequences** for the same input. Any divergence is silent — the model will return garbage embeddings at runtime without throwing an error.

### How Symmetry Is Achieved

The HuggingFace `tokenizers` library is written in Rust. Both the Python package (`tokenizers`) and the Wasm runtime (Rust compiled to Wasm) bind to the **same Rust core**. This makes training/inference symmetry structural rather than a convention to maintain.

```python
# Phase 1 training (Python)
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
encoding = tokenizer.encode("my screen is black")
ids = encoding.ids  # e.g. [101, 2026, 3898, 2003, 2304, 102]
```

```rust
// Phase 2 Wasm engine (same Rust crate, compiled in)
let encoding = tokenizer.encode("my screen is black", false)?;
let ids = encoding.get_ids();  // identical output guaranteed
```

The same `bert-base-uncased` vocabulary is used in both environments. In the Wasm build, the vocab is embedded at compile time via `include_bytes!()` — no separate file.

### Tokenizer Configuration

| Property            | Value                                            |
| ------------------- | ------------------------------------------------ |
| Algorithm           | BERT WordPiece                                   |
| Vocabulary          | `bert-base-uncased` (30,522 tokens)              |
| Casing              | Lowercase                                        |
| Special tokens      | `[PAD]=0`, `[UNK]=100`, `[CLS]=101`, `[SEP]=102` |
| Max sequence length | 128 (truncate, add `[CLS]`/`[SEP]`)              |
| Library             | HuggingFace `tokenizers` crate (Apache 2.0)      |

### Why BERT WordPiece Over n-gram Hashing

An earlier design considered n-gram hashing (no vocab file, FNV-1a hash mod vocab_size). That approach was rejected because:

1. **Cross-language reimplementation is still required.** A custom hasher written in Python and JS can diverge silently on unicode edge cases, punctuation handling, or integer overflow behavior. The `tokenizers` Rust crate eliminates this entirely — it's the same binary in both environments.
2. **WordPiece is a known quality baseline.** n-gram hashing has not been validated for distillation quality. Using a well-established tokenizer removes one variable when evaluating whether ternary distillation works.
3. **The vocab file doesn't need to ship separately.** Because the Wasm engine is already Rust, the vocab embeds into the same `.wasm` binary at compile time (~115KB addition, well within budget).

### Note on Vocab Size

`bert-base-uncased` has 30,522 tokens, not the 10,000 originally scoped. The student model's embedding table is sized to this vocabulary. This increases the embedding table parameter count:

```
Original estimate:  10,000 × 256 = 2,560,000 params  (~640KB packed)
Revised:           30,522 × 256 = 7,813,632 params  (~1.95MB packed)
```

This pushes total packed model size from ~1.75MB to approximately **~2.3MB** for the micro tier. Still within the 5MB total package budget. The model sizing doc should be updated after Phase 1 validation confirms this tokenizer choice.

---

## 4. Teacher Model

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` (HuggingFace sentence-transformers) |
| Parameters | 22M |
| Output dimension | 384 |
| Tokenizer | Internal (handled by `SentenceTransformer` — not used for student inputs) |
| Role | Generates soft embedding targets for the training corpus |

The teacher tokenizes inputs internally via the `SentenceTransformer` API. The student is trained using the `bert-base-uncased` WordPiece tokenizer independently. The two tokenizers operating on the same raw string are not expected to produce the same tokens — what matters is that the student learns to produce similar output vectors given its own token representation.

---

## 5. Student Architecture

| Hyperparameter | Value | Notes |
|---|---|---|
| d_model | 256 | Width is the primary expressivity lever for ternary models |
| n_layers | 2 | Minimum for meaningful compositional attention |
| n_heads | 8 | d_k = 32 per head |
| ffn_dim | 1024 | 4× d_model; do not compress |
| vocab_size | 30,522 | `bert-base-uncased` WordPiece vocabulary |
| max_seq_len | 128 | |
| Embedding dim | 256 | Tied output projection optional |
| Output dim | 384 | Projected to match teacher output via a linear head |
| Total params | ~7M | |

**Output projection:** A float32 linear layer (`256 → 384`) is attached during training to align student output to teacher embedding space. This layer is **discarded at export** — the prototype eval uses the 256-dim student output directly for similarity comparisons.

### BitLinear Layers

All `nn.Linear` layers in attention (Q, K, V, O projections) and FFN are replaced with `BitLinear`:

```python
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = None  # no bias terms

    def forward(self, x):
        # Quantize weights to {-1, 0, +1} using scaled threshold
        scale = self.weight.abs().mean()
        w_ternary = torch.sign(self.weight) * (self.weight.abs() > 0.5 * scale).float()
        # Straight-through estimator: gradients flow through as if no quantization
        w = self.weight + (w_ternary - self.weight).detach()
        return F.linear(x, w)
```

**Layer norms and the output projection remain float32.** Only the attention and FFN linear layers are ternary.

### ⚠️ Important Design Decision: Embedding Table Quantization

> **This is a deliberate deviation from the BitNet b1.58 paper. Understand this before the export step.**

#### What the paper does

BitNet b1.58 (arXiv 2402.17764) explicitly keeps the embedding table at **full float precision** throughout training and inference. The paper states:

> "we preserve the precision for the input/output embedding because the language models have to use high-precision probabilities to perform sampling"

The "1.58-bit" claim refers strictly to the BitLinear weight matrices. Embeddings are excluded from that claim in the paper's reported numbers.

#### Why that reasoning does not apply to @tern

The paper's justification is specific to **language models** that sample the next token. The output embedding feeds into a softmax over the vocabulary, and you need fine-grained float precision to distinguish between closely-scored tokens.

@tern is an **encoder**, not a language model. The embedding output feeds into attention layers and mean pooling — never into a vocabulary softmax or sampling step. The paper's reason for preserving embedding precision simply does not exist in our architecture.

#### Why @tern must quantize the embedding anyway

Keeping the embedding at float precision is not viable at our target model size:

```
Float32:  30,522 × 256 × 4 bytes  = ~31MB   — blows the entire package budget
Float16:  30,522 × 256 × 2 bytes  = ~15.6MB — still way over budget
Int8:     30,522 × 256 × 1 byte   = ~7.8MB  — over budget
Ternary:  30,522 × 256 × 0.25B   = ~1.95MB  — fits within budget
```

The embedding table is ~82% of total parameters. Without quantizing it, BitLinear savings on the linear layers (~6.3MB → ~0.39MB) are irrelevant — the model is still ~32MB and the product premise collapses.

**The majority of @tern's size compression comes from the embedding table, not from BitLinear.**

#### How the embedding gets quantized in Phase 1

The embedding table (`nn.Embedding`) is **not a linear layer** — it performs a lookup (array index), not a matrix multiply. `replace_modules()` only targets `nn.Linear`, so BitLinear and QAT do not apply during training. It remains float32 for the entire training run.

At export time (Phase 1→2 bridge), the embedding is **post-training quantized** to ternary using the same absmean formula as the linear layers:

```python
scale = emb_weight.abs().mean()
emb_ternary = torch.sign(emb_weight) * (emb_weight.abs() > 0.5 * scale)
# pack to 2 bits/weight → ~1.95MB
```

| Stage | Embedding table | Attention + FFN |
|---|---|---|
| Training | Float32 | Float32 shadow weights + ternary forward (QAT) |
| Export | Post-training snap to ternary | Already ternary from QAT |
| Inference | Ternary (2-bit packed) | Ternary (2-bit packed) |

This is post-training quantization, not QAT: the model trained with float32 embeddings and the attention layers learned to operate on those. At inference they receive ternary-quantized vectors instead. The quality gap this introduces is **unknown until Milestone 5 eval is run on the exported model**.

#### The unresolved quality question

A QAT-aware embedding (a custom `nn.Embedding` subclass that applies the same STE trick during training) would let the model adapt to ternary embeddings during training rather than being surprised at export. This is marked High complexity in `tern-future-work.md` Section 5.

**Action item:** Milestone 5 eval must be run on the post-training-quantized checkpoint (embedding snapped to ternary), not the float32 training checkpoint. That is the only way to know whether the quality holds. If the gap is significant, QAT-aware embedding moves from future work into Phase 1 scope before export.

---

## 6. Training Setup

### Dataset

For the prototype, a synthetic/curated dataset of (text, teacher_embedding) pairs is generated offline:

1. Collect ~100,000 short strings: FAQ questions, chatbot prompts, search queries, error messages, code comments. Public sources: MS MARCO, Natural Questions, Stack Overflow titles.
2. Run teacher model once to produce float32 embeddings for all strings. Cache to disk.
3. Training samples are `(bert_wordpiece_token_ids, teacher_embedding_vector)`.

For the prototype, dataset scale can be reduced to **20,000–50,000 samples** to validate the training loop before committing to full-scale training.

### Loss Function

Primary loss: cosine embedding loss between student output (projected to 384-dim) and teacher embedding.

```python
loss = 1 - F.cosine_similarity(student_output, teacher_embedding, dim=-1).mean()
```

Optional secondary loss: mean squared error between normalized vectors (adds sensitivity to magnitude, not just direction).

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-4 with linear warmup (10% of steps) |
| Warmup strategy | First 5 epochs: shadow weights only (no ternary clipping) |
| Batch size | 64 |
| Epochs | 30 (prototype); eval after every epoch |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Device | Single GPU (T4 or better); CPU feasible for prototype scale |

**Warmup note:** The first N epochs train with float32 shadow weights and no ternary projection. QAT activates after warmup. This prevents early ternary collapse before the model has learned useful representations.

---

## 7. Eval — Simulated Inference

Eval is designed to mirror inference as closely as possible. It does **not** use the teacher tokenizer.

### Eval Pipeline

```
raw string
    ↓
HuggingFace `tokenizers` Python bindings
(bert-base-uncased WordPiece — same Rust core as Wasm build)
    ↓
token IDs (padded/truncated to 128)
    ↓
student forward pass (ternary weights active, no float32 shadow)
    ↓
256-dim embedding vector
    ↓
cosine similarity / task eval
```

Ternary weights are hardened (no straight-through) during eval to reflect true inference conditions.

### Eval Tasks

**Task 1: Teacher Alignment (primary)**
On a held-out set of 2,000 strings, compute cosine similarity between student embedding and teacher embedding for the same input.

Report: mean cosine similarity, distribution (p25/p50/p75/p95).

**Task 2: Semantic Similarity Ranking (STS proxy)**
Take 200 string pairs labeled as (similar / not similar). Compute student cosine similarity for each pair. Report area under the ROC curve (AUC).

This is the closest proxy to the actual product use case without a formal STS benchmark.

**Task 3: Nearest Neighbor Retrieval**
Given 50 query strings and a corpus of 500 candidate strings, check whether the correct match appears in the top-3 nearest neighbors by cosine similarity.

Report: Recall@1, Recall@3.

---

## 8. Risks, Thresholds, and Go/No-Go Criteria

### Risk 1: Ternary Weight Collapse

**What it is:** During QAT, the model learns to push all shadow weights toward zero. The ternary projection outputs all-zero weight matrices, and the model degenerates to a near-identity function.

**How to detect:** After QAT activation, monitor the fraction of weights hardened to zero each epoch. A healthy ternary model typically has 20–40% zero weights (the zero state is meaningful sparsity). Collapse looks like >70% zeros and a loss plateau.

| Threshold | Status |
|---|---|
| <40% zero weights after epoch 10 | Acceptable |
| 40–60% zero weights | Watch closely — may still recover |
| >60% zero weights + loss not decreasing | **No-go — revisit warmup schedule and initialization** |

**Mitigations if triggered:** Extend float32 warmup, reduce learning rate at QAT activation, try a higher zero-band threshold in the ternary projection.

---

### Risk 2: Distillation Quality Floor

**What it is:** The student cannot compress the teacher's representations sufficiently. The 2-layer ternary architecture may lack the capacity to produce embeddings that are semantically useful, regardless of training duration.

**How to detect:** Task 1 (teacher alignment) and Task 2 (STS proxy AUC) after full training.

| Metric | Acceptable | Marginal | No-go |
|---|---|---|---|
| Mean cosine sim (Task 1) | > 0.75 | 0.60–0.75 | < 0.60 |
| STS proxy AUC (Task 2) | > 0.80 | 0.70–0.80 | < 0.70 |
| Recall@3 (Task 3) | > 0.70 | 0.55–0.70 | < 0.55 |

**If marginal:** Scale to `d_model=384` / ~12M params (base tier) and re-run before concluding the approach is invalid.

**If no-go at base tier:** The ternary constraint may be fundamentally incompatible with the required task quality. This would require re-evaluating the quantization scheme (e.g., switching from pure ternary to 4-bit for the embedding table only).

---

### Risk 3: Tokenizer Parity (Significantly Mitigated)

**What it is:** Training and inference produce different token IDs for the same input, causing the model to silently return wrong embeddings at runtime.

**Why this risk is substantially lower than with a custom tokenizer:** The Python training environment and the Wasm inference engine both use the HuggingFace `tokenizers` Rust crate — the same underlying binary. There is no cross-language reimplementation to maintain or test. Parity is structural.

**Residual risk:** The Wasm build of the `tokenizers` crate could theoretically behave differently to the Python build if there are platform-specific code paths (e.g., SIMD-optimized unicode normalization that behaves differently on Wasm vs. native). This is unlikely but not impossible.

| Threshold | Status |
|---|---|
| Token IDs match on spot-check of 50 strings (Python vs. Wasm) | Acceptable — proceed |
| Any mismatch identified | Investigate `tokenizers` crate Wasm build — likely a known issue with a crate version fix |
| Post-training Wasm eval score significantly below Python eval score | Investigate tokenizer parity first before assuming model quality issue |

**Spot-check should still be done at the start of Phase 2** when the Wasm build is first compiled, before full integration testing.

---

### Risk 4: Training Instability

**What it is:** Loss diverges or oscillates without converging. More likely with ternary models than float32 due to the discontinuous gradient landscape.

| Threshold | Status |
|---|---|
| Loss decreasing consistently by epoch 5 | Acceptable |
| Loss oscillating but trending down by epoch 10 | Marginal — reduce LR |
| Loss diverging or flat after epoch 10 | **No-go — revisit optimizer config and warmup** |

---

### Risk 5: Dataset Domain Mismatch

**What it is:** The training corpus doesn't cover the vocabulary or phrasing patterns of real @tern use cases (developer queries, chatbot prompts, FAQ strings). The student produces good embeddings for generic English but poor ones for "webpack config", "OAuth token", "null pointer exception".

**How to detect:** Task 3 (Recall@3) on a corpus of developer-domain strings specifically. If Task 1 looks good but Task 3 on tech vocabulary is poor, this is the cause.

| Threshold | Status |
|---|---|
| Recall@3 on general corpus and tech corpus within 0.05 of each other | Acceptable |
| Tech corpus Recall@3 >0.10 below general corpus | Expand training data with more Stack Overflow, GitHub, and documentation text |
| Tech corpus Recall@3 <0.50 | **No-go — fundamental domain gap, retrain with richer corpus** |

---

## 9. Prototype Exit Criteria

The prototype is complete and Phase 2 (Wasm engine) can begin when **all of the following are true**:

- [ ] Tokenizer confirmed as `bert-base-uncased` WordPiece via HuggingFace `tokenizers` crate in both training and eval
- [ ] Task 1 mean cosine similarity ≥ 0.75 on held-out set
- [ ] Task 2 STS proxy AUC ≥ 0.80
- [ ] Task 3 Recall@3 ≥ 0.70 on both general and tech-domain corpus
- [ ] Zero weight fraction stays below 60% across all BitLinear layers at end of training
- [ ] Training loss converges (no divergence after epoch 15)

If marginal thresholds are hit on quality metrics, the `d_model=384` config must be evaluated before declaring a no-go.

---

## 10. Outputs

| Artifact | Description |
|---|---|
| `student_model.py` | Student architecture with BitLinear layers |
| `train.py` | Training loop with QAT warmup schedule |
| `eval.py` | Inference-mirroring eval pipeline using `tokenizers` Python bindings |
| `results/` | Per-epoch eval metrics, weight distribution histograms |

No custom tokenizer implementation is needed. The `tokenizers` Python package (`pip install tokenizers`) is the only addition to the training environment.

The trained model weights (float32 shadow + hardened ternary checkpoint) are saved but are **not** the shipping artifact. The `.bin` export pipeline is Phase 2.
