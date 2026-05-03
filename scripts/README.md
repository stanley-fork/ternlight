# scripts

Build orchestration. One script per orchestration concern, kept simple and runnable from the repo root.

| Script | What it does |
|---|---|
| `build-engine.sh` | `cargo build` → `wasm-opt -Oz` → copy `engine.wasm` into `packages/semantic/` |
| `release-model.sh` | Push the latest `.bin` to a GitHub Release as a versioned asset |
| `run-eval.sh` | Run all evals (regression, benchmarks, compatibility), regenerate `eval/REPORT.md` and `eval/results/v<X.Y.Z>.json` |
| `publish.sh` | Bump version, build everything, run evals, publish to npm (gated on PASS) |

All scripts assume the repo root as the working directory.