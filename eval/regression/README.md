# eval/regression

End-to-end regression tests against Phase 1 quality baselines. These run the *shipped engine* (compiled Wasm + bundled model) through the same eval tasks the trained checkpoint was scored on.

## Quickstart

```bash
# 1. One-time: cache reference data (MS MARCO queries + teacher embeddings, STS-B, retrieval corpora)
python prepare_eval_data.py

# 2. Run regression — uses the engine in packages/semantic
node regression_test.js
```

## What gets tested

Three tasks, mirroring `training/distill/evaluate.py`:

- **Task 1** — Teacher alignment (mean per-query cosine sim on held-out MS MARCO)
- **Task 2** — STS-B sentence pair ranking (AUC + Spearman against human scores)
- **Task 3** — Recall@K nearest-neighbor retrieval (general + tech corpora)

PASS = within 0.02 of the Phase 1 baseline reported in [../../docs/training/phase-1-conclusion.md](../../docs/training/phase-1-conclusion.md).

## Why this is separate from `engine/tests/`

`engine/tests/` answers "does the Rust math match Python at the element level?" — a *parity* test against a single reference vector.

`eval/regression/` answers "does the shipped engine produce *quality* embeddings on real tasks?" — a *quality* test against eval baselines.

The Phase 2 BitLinear bug ([../../docs/training/postmortem-bitlinear-asymmetry.md](../../docs/training/postmortem-bitlinear-asymmetry.md)) showed why both layers are necessary. Engine parity passed at 1e-7 against a hand-rolled reference while the engine was actually computing different math from the trained model. Quality regression caught it.

## Status

Pre-alpha — code migration from `tern-distill-prototype/bridge/` pending.