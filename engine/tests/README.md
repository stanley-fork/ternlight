# engine/tests — parity tests

Element-level tests that the Rust engine's output matches a Python reference produced by the *real* training-time model.

```bash
node tests/test_embed.js
```

## What goes here

- `test_embed.js` — full forward-pass parity test (engine vs `dump_embed.py` output)
- `test_data/` — reference JSON files (engine output dumps for known inputs)

## What does NOT go here

- Engine quality eval against held-out tasks → that lives in [`../../eval/regression/`](../../eval/regression/)
- Performance benchmarks → [`../../eval/benchmarks/`](../../eval/benchmarks/)
- Package integration tests → `packages/*/tests/`

## The reference must be the real model

Phase 2 caught a bug where parity tests passed against a hand-written Python reference that mirrored our (incorrect) engine, while diverging from the actual trained model. References in this directory MUST be produced by calling the real `model_scratch.py` model directly — not by reimplementing the forward pass.

See [../../docs/training/postmortem-bitlinear-asymmetry.md](../../docs/training/postmortem-bitlinear-asymmetry.md) for the full lesson.