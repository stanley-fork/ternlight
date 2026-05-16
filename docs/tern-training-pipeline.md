# Tern Training Pipeline — Rigorous Run Plan

> First training run after the POC concluded GO. The POC was **discovery** — "does this architecture work at all?" This run is **replication with rigor** — "we believe it works; now produce a defensible, reproducible artifact." Most decisions are locked from POC results; the *process* gets a step change, not the model.

---

## What's locked from the POC

| Decision | Value | Source |
|---|---|---|
| Teacher | `sentence-transformers/all-MiniLM-L6-v2` | [tern-phase1-prototype.md](tern-phase1-prototype.md) |
| Student architecture | d_model=256, 2 layers, 4 heads, ffn=1024, output=384 | [training/phase-1-conclusion.md](training/phase-1-conclusion.md) |
| Loss recipe | distillation (cosine, w=1.0) + contrastive (w=0.15) | POC scaled run |
| Quant math | AbsMedian round-clamp for BitLinear, AbsMean for embedding, STE | [training/postmortem-bitlinear-asymmetry.md](training/postmortem-bitlinear-asymmetry.md) |
| QAT warmup | proportional float32 warmup, then `lambda=1.0` for ternary | POC `train_qat.py` |
| Tier | micro only this round | scoped decision |

## Decisions for this run

| Decision | Value | Rationale |
|---|---|---|
| Dataset size | **1M** samples | 4× POC's 150k; clearly bigger; still trainable locally |
| Data composition | **60% MS MARCO + 40% `sentence-transformers/embedding-training-data`** | UI demo target favors general semantic search; see Phase 1 below |
| Compute | **Local M-series (MPS)** | Cloud GPU deferred — start with what we have, escalate if wall-clock is unacceptable |
| Epochs | **~60** (QAT) / **~20** (fp32 baseline) | Fewer than POC's 100 — at 1M samples each epoch covers more variety |
| Warmup | **6 epochs** (10% of total) | POC's 5/100 was arguably too short |

---

## Phase 1 — Data prep

**What changes from POC:**

- **Scale**: 150k → 1M samples
- **Two sources, demo-oriented mix**:
  - **60% MS MARCO** (`ms_marco/v2.1`) — real search queries → passages. Distribution = what users type into a search bar.
  - **40% `sentence-transformers/embedding-training-data`** — meta-dataset containing AllNLI, Quora paraphrases, Reddit titles, StackExchange Q&A, etc. Gives paraphrase robustness, which is what makes a semantic search UI feel impressive vs. a regex.
  - StackExchange-only tech subset deliberately deprioritized — useful but doesn't showcase the "find the right FAQ" use case a UI demo lives on.
- **Fixed split, seeded once**: train / val / test stratified by source, written to a manifest. The split never gets regenerated; subsequent runs reuse it. POC re-split each run, which made comparisons unreliable.
- **Cache provenance**: each `.pt` cache stamped with `{source_manifest, teacher_id, teacher_revision, tokenizer_revision, code_commit, sample_count, seed}`.
- **Pre-training sanity dashboard**: length histogram, teacher-emb norm distribution, dedup count, per-source ratio. One look before kicking off a multi-hour run.

**Approximate wall-clock on M-series**: teacher-encoding 1M sentences with MiniLM at batch 256 is ~30–60 min. Done once, reused.

## Phase 2 — Float32 baseline (the ruler)

**What changes from POC:**

- **Run on the same 1M dataset** Phase 3 will use, otherwise the comparison isn't apples-to-apples
- **Same hyperparameters** as Phase 3 except no QAT — the only variable is quantization
- **Shorter — ~20 epochs**, fp32 distillation converges fast at this scale
- **One run, no sweep**. Architecture is locked; if the ceiling drops vs. POC's 0.787, the data prep is the suspect, not the model.

This phase exists to answer one question: *"did Phase 1 produce data that lets the architecture hit a healthy ceiling?"* If yes, proceed to Phase 3. If no, debug Phase 1.

## Phase 3 — QAT training (the ship)

**What changes from POC:**

- **Scale**: 150k / 100 epochs → 1M / 60 epochs
- **Proportional warmup**: 6 epochs (10% of total)
- **Resumable checkpoints every 5 epochs** — at this scale a crash mid-run is real cost
- **Inline eval every 5 epochs**: run a fast Task 1 (1k samples, mean cosine) — see the quality curve in real time, not just at the end. Lets you early-stop if it plateaus.
- **Expanded diagnostic logging**: zero-fraction, gradient norms, per-layer activation magnitudes, val/spearman, embedding std. POC tracked zero-fraction; the rest were ad-hoc inspection.
- **Compute reality (M-series)**: 1M × 60 epochs is materially longer than POC. Suggested staging:
  1. **Smoke** — 10k samples, 2 epochs (~5 min): validates the full pipeline end-to-end
  2. **Pilot** — 250k samples, 30 epochs: validates training at scale, eyeballs quality at the halfway point of full data
  3. **Full** — 1M / 60 epochs: the rigorous run
  
  Going straight to step 3 without 1 and 2 is the kind of decision that loses two days to a bug in step 1.

## Phase 4 — Post-train eval

**What changes from POC:**

- **Run on every checkpoint** Phase 3 saved, not just the last. The quality curve is more useful than a point estimate.
- **Ablation matrix as default output**, not a `--no-quant-embedding` flag:

  | Variant | Embedding | BitLinear weights | What it tells you |
  |---|---|---|---|
  | fp32 ceiling | f32 | f32 | architecture upper bound |
  | QAT training-time | f32 | ternary | what QAT alone costs |
  | QAT + ternary-emb (ship) | ternary | ternary | what the shipped `.bin` actually delivers |

  Reporting all three side-by-side makes the quantization cost visible per stage.
- **Expanded Task 3 corpora**: POC had 20 pairs per domain. Bump to 100+ per domain. Domain set adjusted for UI demo: general consumer (passwords, refunds, shipping), FAQ-paraphrase (Quora-style), product-search (item descriptions). Tech-domain deprioritized.
- **Add MTEB as Task 4**: pick 4–6 subsets — STS22, BIOSSES, TwitterPara, SciDocsRR, BankingIntent, RedditClustering. MTEB is the credibility benchmark in this space; "tern scores X on MTEB subset Y" is what gets external trust.
- **Versioned scorecard**: emit `eval/results/<run_id>.json` per training run, committed to repo. Future runs diff against it automatically.

## Phase 5 — Pack (`.pt` → `.bin`)

**What changes from POC:**

- **Parity as contract**: pack the `.bin`, load it back through a Python reference impl (`pack/verify.py`), re-run Phase 4 tasks, assert scores within ~1e-4 of the `.pt` scores. This is the regression test that catches the bitlinear-asymmetry-style bug *before* it ships.
- **Sidecar manifest**: `model.bin` + `model.bin.json` with `{training_run_id, code_commit, eval_scorecard, source_data_manifest, packed_at_iso, sha256}`. The `.bin` alone is opaque; the sidecar makes it provenanced.
- **Single source of quant math** — `training/distill/quantization.py` and `training/pack/pack.py` share the implementation. They don't re-derive from the postmortem doc independently.

---

## Cross-cutting principles (the real step change)

1. **Reproducibility is mechanical, not aspirational.** Every artifact (cache, checkpoint, `.bin`, scorecard) stamped with inputs. Anyone with the manifest + code commit can re-derive.
2. **Resumability everywhere.** At this scale, "restart from scratch" is unacceptable cost. Phase 1 caches by source, Phase 3 checkpoints every 5 epochs, Phase 4 evaluates per-checkpoint.
3. **Continuous eval, not endpoint eval.** Wait 12 hours for one number, find out it's bad? No. Eval per checkpoint, plot the curve.
4. **Diagnostics are first-class output.** Collapse detection, distribution drift, gradient health surface automatically — not "I noticed the loss looked weird."
5. **Honest negatives.** When a run fails or plateaus, that's a result. Commit the scorecard with the failure verdict; don't silently retry.

---

## Success criteria

The run is GO if the **ternary-embedding** ablation column passes all four thresholds:

| Task | Threshold | POC result (reference) |
|---|---|---|
| Task 1 — Teacher alignment | mean cosine > 0.75 | 0.812 |
| Task 2 — STS-B AUC | > 0.80 | 0.839 |
| Task 3 — Recall@3 (min across domains) | > 0.70 | 0.75 / 1.00 |
| Task 4 — MTEB average (4–6 subsets) | > 0.50 | not measured in POC |

Anything FAIL → NO-GO. Anything MARGINAL → consider d_model=384 (small tier).

---

## Open items not yet decided

- **Cloud GPU escalation criteria**: at what wall-clock does local M-series stop being viable? Suggested: if Phase 3 full run exceeds 48h wall-clock, move to single cloud GPU for the next run.
- **MTEB subset selection**: 4–6 subsets chosen for UI-search relevance, but exact list TBD.
- **Quora paraphrase pairs ratio inside `embedding-training-data`**: the meta-dataset's mix may not weight Quora high enough for the UI-demo target. May need to over-sample Quora within the 40% allocation.
