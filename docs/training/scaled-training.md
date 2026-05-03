# Scaled Training Phase — From POC to Robust Checkpoint

> **Context:** The POC (milestones 1–5) de-risked the core architecture. QAT works, BitLinear is stable, the student's embedding space has correct semantic structure. This phase scales up training to produce a checkpoint robust enough to settle architecture decisions before the Rust/Wasm engine build begins.

---

## What the POC Proved

| Question | Result |
|---|---|
| Can a 2-layer ternary transformer learn from distillation? | Yes — cosine_sim 0.75 during training |
| Is QAT stable with BitLinear? | Yes — zero fraction 28%, no collapse |
| Does the student's space preserve semantic structure? | Yes — STS AUC 0.81, Recall@3 up to 1.0 |
| Does embedding quantization break ranking? | No — retrieval unaffected |
| Can we train locally on M4 Max? | Yes — full run under 7 minutes |

## What's Still Unproven

1. Can d_model=256 reach the 0.75 eval threshold with more data/epochs?
2. What is the true embedding quantization quality gap at scale?
3. Does quality generalize to domains not in the training set?

---

## Architecture Decisions This Phase Must Settle

These two questions gate the Rust/Wasm engine spec. They cannot be deferred.

### Decision 1: d_model — 256 or 384?

The POC used d_model=256 (~9.5M params, ~2.7MB packed). If the scaled run reaches Task 1 cosine_sim > 0.75, the architecture is confirmed. If it lands marginal (0.60–0.75), d_model=384 (~15M params, ~4.5MB packed) is the fallback.

**Impact on Rust engine:** d_model determines every weight matrix shape, SIMD loop bounds, and Wasm memory allocation. Changing it after the engine is built means reworking matrix dimensions throughout.

### Decision 2: Embedding precision — ternary or int8?

The POC showed embedding quantization costs ~0.084 on absolute teacher alignment but doesn't hurt ranking. At scale, this gap may shrink (more training = more quantization-robust representations) or persist.

**How to test:** Every scaled eval run should be executed twice — with and without `--no-quant-embedding`. If the gap narrows below ~0.03, ternary is fine. If it persists above ~0.05, int8 embedding (combined with future vocab pruning) becomes the path.

**Impact on Rust engine:** Ternary embedding = single unpacking codepath, simple .bin format. Int8 embedding = two codepaths, scale factors in .bin header, mixed-precision memory layout.

---

## Experiment Plan

### Experiment 1 — Scaled d_model=256 (kick off today)

**Hypothesis:** The model is undertrained, not underpowered. More data and epochs will push eval cosine_sim past 0.75.

| Parameter | POC Value | Scaled Value |
|---|---|---|
| Dataset | MS MARCO 25k | MS MARCO 150k |
| Epochs | 30 | 100 |
| QAT warmup | 5 epochs | 5 epochs |
| Batch size | 64 | 64 |
| d_model | 256 | 256 |
| Loss | distill + contrastive | distill + contrastive |
| LR | 1e-4 | 1e-4 |

**Estimated runtime:** ~2.5 hrs on MPS (~15 min prepare + ~134 min train)

**Eval:** Run `eval.py` twice on the checkpoint:
- With ternary embedding → measures shipped model quality
- Without embedding quant → isolates the quantization gap

**Success criteria:**

| Outcome | Task 1 cosine_sim | Next step |
|---|---|---|
| Pass | > 0.75 (ternary emb) | Architecture confirmed. Proceed to Rust engine. |
| Pass without quant, marginal with | > 0.75 float, 0.65–0.75 ternary | d_model=256 confirmed. Embedding precision is the remaining problem — flag for future work. |
| Marginal | 0.60–0.75 (both) | Run Experiment 2 (d_model=384). |
| Fail | < 0.60 | Investigate — likely data quality or loss tuning issue. |

### Experiment 2 — d_model=384 fallback (only if Experiment 1 is marginal)

Same dataset and epochs as Experiment 1, only architecture changes:

| Parameter | Value |
|---|---|
| d_model | 384 |
| n_heads | 6 |
| ffn_dim | 1536 (4x) |
| output_dim | 384 |

**Estimated packed size:** ~4.5MB (vs ~2.7MB for d_model=256). Still within the 5MB budget but with less headroom.

**Estimated runtime:** ~45 min on MPS.

---

## Code Changes Required

### 1. Update prepare.py

Scale dataset from 25k to 150k:

```python
N_SAMPLES   = 150_000
CACHE_FILE  = CACHE_DIR / "msmarco_150k.pt"
```

**Runtime:** ~15 min for teacher encode on 150k queries.

### 2. Update train_qat.py

Point to the new cache and extend epochs:

```python
CACHE_FILE   = Path("cache/msmarco_150k.pt")
EPOCHS       = 100
RUN_NAME     = "micro-qat-150k-100ep"
```

### 3. Eval

Run eval.py on the resulting checkpoint, both with and without `--no-quant-embedding`.

### 4. If Experiment 2 needed

Update model architecture constants:

```python
D_MODEL  = 384
N_HEADS  = 6
FFN_DIM  = 1536
```

---

## Dataset Strategy — Beyond MS MARCO

The current runs use MS MARCO exclusively. MS MARCO is Q&A-heavy — short questions with a "seeking information" tone. That's a slice of what @tern needs to handle, but not the full picture.

### The principle

The teacher model (all-MiniLM-L6-v2) was trained on a massive diverse corpus. It already knows what "webpack config" and "quarterly earnings" mean. But when distilling from it, the student can only learn patterns that appear in **the student's training data**. If the student never sees tech strings, it can't learn the teacher's representation of them.

**The training corpus should cover the distribution of inputs the model will see at inference** — not the same inputs, but the same *types* of inputs. Short questions, statements, commands, error messages, titles.

### What's missing from MS MARCO

MS MARCO is almost entirely questions. @tern's target use cases also include:

- Statements and assertions ("your payment has been processed")
- Commands and intents ("cancel my subscription")
- Error messages and technical strings ("null pointer exception in line 42")
- Titles and headings ("Getting Started with OAuth 2.0")
- Near-duplicates with subtle differences ("reset password" vs "reset username")

### Recommended mixed corpus for production training

| Source | HuggingFace ID | What it adds | Suggested size |
|---|---|---|---|
| MS MARCO | `ms_marco` v2.1 | Search queries (questions) | 50k |
| Quora Question Pairs | `quora` | Near-duplicate detection — directly matches @tern's dedup use case | 30k |
| AllNLI | `sentence-transformers/all-nli` | Premise/hypothesis pairs — statements, not questions | 30k |
| Stack Overflow titles | `pacovaldez/stackoverflow-questions` | Tech/developer domain coverage | 20k |
| ParaNMT | `sentence-transformers/parallel-sentences-talks` | Paraphrases — same meaning, diverse phrasing | 20k |
| **Total** | | | **150k** |

Each source covers a gap the others don't. The pipeline stays the same — tokenize, teacher encode, cache — just with a mixed-source input list.

### When to apply this

Don't change the dataset mid-run. The current 150k MS MARCO run settles the architecture question (d_model). Once that's locked, dataset diversification is the next lever to pull for quality improvement. A mixed-corpus run at the same 150k scale would be a direct comparison against the MS MARCO-only baseline.

---

## What Comes After This Phase

Once d_model and embedding precision are settled, the architecture is frozen. The next steps, in order:

| Phase | Work | Depends on |
|---|---|---|
| Vocab/embedding exploration | Custom 10k BPE, int8 embedding trade-off | Decision 2 outcome |
| Output dim confirmation | Keep 384 projection or drop to d_model | Settled architecture |
| .bin export format | Header spec, packing script | All decisions above |
| Rust/Wasm engine | Build against frozen spec | .bin format finalized |

Vocab pruning and output dim are optimizations that can be explored after the Rust engine has a working v1. They don't gate the engine build — they improve the shipped model quality and size.

---

## Tracking

All runs should follow the WandB naming convention:

```
{tier}-{mode}-{dataset_size}-{epochs}

Examples:
  micro-qat-150k-100ep          # Experiment 1
  micro-qat-150k-100ep-eval    # Experiment 1 eval (ternary emb)
  micro-qat-150k-100ep-eval-fp # Experiment 1 eval (float emb)
  base-qat-150k-100ep          # Experiment 2 (d_model=384)
```
