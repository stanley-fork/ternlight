# training

Python pipeline that produces the `.bin` model file consumed by the Wasm engine. Two stages:

```
training/
├── distill/    Stage 1 — distillation training (Python, GPU)
│   ├── train.py        QAT training loop
│   ├── evaluate.py     Phase 1 eval suite (Tasks 1, 2, 3 — go/no-go criteria)
│   ├── model/          StudentEncoder definition (transformer + projection)
│   ├── training/       loss functions, optimizer setup, training utilities
│   ├── eval/           training-time eval (per-epoch val/spearman)
│   ├── data/           data prep + caching (MS MARCO, teacher embeddings)
│   ├── config/         per-tier configs (micro.yaml, small.yaml, ...)
│   └── requirements.txt
└── export/     Stage 2 — checkpoint → packed .bin (Python, CPU)
    └── export.py       reads .pt checkpoint, applies post-training quantization,
                        packs ternary weights to 2 bits, writes .bin file
```

## When to touch what

- **New training run** — modify `distill/config/*.yaml`, run `distill/train.py`
- **New loss function** — `distill/training/loss.py`
- **Architecture change** — `distill/model/student.py`, then update `distill/config/*.yaml`
- **Post-training quantization change** — `export/export.py`
- **Eval methodology change** — `distill/evaluate.py` for go/no-go; `eval/regression/` (at repo root) for ongoing quality regression

## Outputs

```
distill/runs/<run-name>/
    └── checkpoint_ep<N>.pt          float32 shadow weights + config + optimizer state
                                     (NOT committed — see .gitignore)
        ↓ export.py
export/out/model.bin                 packed ternary + projection (the shipped artifact)
                                     (also NOT committed — attached to GitHub Release)
        ↓ scripts/release-model.sh
github.com/.../releases/v0.1.0       per-version model binary download
        ↓ at npm publish time
packages/semantic/model.bin          bundled into the published package
```

## Math reference

What the model actually computes — forward pass, backprop, distillation dynamics — is documented in [../docs/training/model-internals.md](../docs/training/model-internals.md). Read that before changing the training code.

## Status

Pre-alpha. Source code migration from `tern-distill-prototype/poc/` and `tern-distill-prototype/export/` pending.