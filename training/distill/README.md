# training/distill

Stage 1 of the training pipeline: distill a small ternary student from a frozen teacher. Produces a `.pt` checkpoint that `training/export/` consumes.

## Quickstart

```bash
cd training/distill
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. One-time: cache MS MARCO queries + teacher embeddings to disk
python data/prepare.py

# 2. Train (uses GPU if available)
python train.py --config config/micro.yaml

# 3. Run go/no-go eval suite on the final checkpoint
python evaluate.py --checkpoint runs/micro-qat-150k-100ep/checkpoint_ep100.pt
```

## Layout

```
distill/
├── config/         per-tier YAML configs (micro = d_model=256, n_layers=2)
├── data/           MS MARCO loader, teacher embedder, .pt caching
├── model/          StudentEncoder (model_scratch.py — uses explicit Q/K/V Linear layers
│                   so BitLinear can replace them; nn.MultiheadAttention won't work)
├── training/       loss functions (distillation, contrastive), optimizer setup
├── eval/           per-epoch validation during training (val/spearman)
├── train.py        main training loop with QAT activation at epoch 5
├── evaluate.py     final-checkpoint eval suite — Tasks 1, 2, 3 (Phase 1 go/no-go)
└── requirements.txt
```

## Math reference

Forward pass, backprop, loss formulation, why each design choice was made — all in [../../docs/training/model-internals.md](../../docs/training/model-internals.md). The Phase 2 postmortem ([../../docs/training/postmortem-bitlinear-asymmetry.md](../../docs/training/postmortem-bitlinear-asymmetry.md)) covers what the *real* BitLinear forward pass computes (relevant if you're modifying the inference engine to match).

## Status

Pre-alpha — code migration from `tern-distill-prototype/poc/` pending.