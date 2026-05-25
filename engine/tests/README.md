# engine/tests — parity tests

Element-level tests that the Rust engine's output matches the Python reference at
[`training/pack/unpack.py`](../../training/pack/unpack.py) (`UnpackedModel.forward()`).

```bash
# Build a feature first
wasm-pack build --target nodejs --features emb_int8
# Then run the tests
node tests/test_embed.js
```

## What goes here

- `test_embed.js` — full forward-pass parity test (engine vs `unpack.UnpackedModel.forward()`)
- `test_tokenizer.js` — spot-check `tokenize()` against the Python `tokenizers` library
- `test_embedding.js` — per-format embedding lookup parity
- `test_bitlinear.js` — single-BitLinear forward parity
- `reference_dumps/` — captured Python reference outputs (gitignored; regenerated via script)

## What does NOT go here

- Engine quality eval against held-out tasks → that lives in [`../../eval/regression/`](../../eval/regression/)
- Performance benchmarks → [`../../eval/benchmarks/`](../../eval/benchmarks/)
- Package integration tests → `packages/*/tests/`

## The reference must be the real model

The single source of truth for engine forward-pass correctness is `training/pack/unpack.py`.
That module uses the same quant math as the pinned `bitlinear==2.4.6` library — so matching
`unpack.UnpackedModel.forward()` IS matching the training-time model.

DO NOT hand-write Python reference math separately. The prior tern-core POC hit exactly that
trap: parity tests passed at 1e-7 because both the engine and the hand-written reference were
buggy in the same way. See
[`../../docs/training/postmortem-bitlinear-asymmetry.md`](../../docs/training/postmortem-bitlinear-asymmetry.md).

Per-format tolerances are defined in
[`../../docs/tern-inference-engine.md`](../../docs/tern-inference-engine.md#verification--parity-contract).

## Status

Scaffolding only. Tests land after the engine modules in `../src/` are implemented.
