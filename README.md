# tern

[![CI](https://github.com/soycaporal/ternlight/actions/workflows/ci.yml/badge.svg)](https://github.com/soycaporal/ternlight/actions/workflows/ci.yml)
[![Build Engine](https://github.com/soycaporal/ternlight/actions/workflows/build-engine.yml/badge.svg)](https://github.com/soycaporal/ternlight/actions/workflows/build-engine.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **On-device semantic embeddings for JavaScript.** A ~3 MB engine + model that gives you vector search, similarity matching, and intent routing locally — no API keys, no backend, no GPU.

`tern` is the SQLite of semantic matching: zero-config, entirely self-contained, and built to embed inside your app rather than sit behind a service.

```js
import { embed, cosineSim, similar } from '@tern/semantic';

const v1 = await embed("how do I reset my password");
const v2 = await embed("forgot my password");
cosineSim(v1, v2);                                  // 0.78

// Find nearest matches in a corpus
const matches = await similar(query, corpus, { topK: 3 });
```

> **Status:** v0.1 (pre-alpha). The engine works end-to-end and matches Phase 1 quality baselines. Packaging, performance polish, and public release are still in progress. Not yet on npm.

---

## What this gives you

- **`embed(text)`** — turn a string into a 384-dim L2-normalized vector
- **`cosineSim(a, b)`** — compute similarity between two vectors (0 = unrelated, 1 = identical meaning)
- **`similar(query, corpus, opts)`** — nearest-neighbor search over a corpus

That's it. One primitive, well-built. Composition into classifiers, filters, and other use cases is straightforward — see [docs/](docs/).

## Why this exists

Existing options for semantic embeddings in JavaScript don't fit the on-device niche:

| Option | Size | Network calls | Cost | API key |
|---|---|---|---|---|
| `transformers.js` + MiniLM | ~80 MB | none | free | none |
| OpenAI embedding API | n/a | every call | $/call | yes |
| Cloudflare AI Workers | n/a | every call | $/call | yes |
| **`tern`** | **~3 MB** | **none** | **free** | **none** |

If you're building a browser extension, a static-site search, an Obsidian plugin, an edge-runtime app, or anything that should work offline and respect user data, `tern` exists for you.

## Repository layout

```
ternlight/
├── packages/         JS packages (npm, pnpm workspace)
│   ├── semantic/     @tern/semantic — embedding API
│   └── core/         @tern/core — shared types
├── engine/           Rust → Wasm inference engine
├── training/         Python distillation pipeline
├── eval/             Engine quality + perf benchmarks
├── docs/             Design docs, architecture, postmortems
├── models/           Model release registry pointers
├── scripts/          Build orchestration
└── .github/          CI workflows
```

See [docs/tern-monorepo.md](docs/tern-monorepo.md) for the full structure rationale and contributor layers.

## Documentation map

- **Start here:** [docs/tern-scoping.md](docs/tern-scoping.md) — what `tern` is and the problem it solves
- **Architecture:** [docs/tern-architecture.md](docs/tern-architecture.md) — system design, .bin format, runtime model
- **Training:** [docs/training/model-internals.md](docs/training/model-internals.md) — forward pass, backprop, distillation math (canonical reference)
- **Postmortem:** [docs/training/postmortem-bitlinear-asymmetry.md](docs/training/postmortem-bitlinear-asymmetry.md) — the engine bug we caught in Phase 2 and how
- **Future work:** [docs/tern-future-work.md](docs/tern-future-work.md) — open questions, deferred optimizations

## Contributing

Each subdirectory is self-contained — the toolchain you need depends on what you're touching:

| Touching | Toolchain needed |
|---|---|
| `packages/` (JS API) | Node.js, pnpm |
| `engine/` (Wasm engine) | Rust, wasm-pack |
| `training/` (model training) | Python, PyTorch, GPU |
| `eval/` (quality + perf) | Node.js + Python |
| `docs/` | Markdown |

A JS contributor doesn't need Rust installed; an ML researcher doesn't need to understand Wasm. The compiled engine ships as a build artifact.

See [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).