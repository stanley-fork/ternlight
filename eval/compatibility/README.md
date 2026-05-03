# eval/compatibility

Target-runtime matrix tests. Verifies the engine instantiates, runs `embed()`, and produces the expected output across all supported deployment targets.

## Targets

Defined in `runtimes.yaml`. Initial v0.1 target list:

| Target | Why it matters |
|---|---|
| Node.js 18+ | Server-side default |
| Browser (Chromium, Firefox, Safari) | The "static site search" use case |
| Cloudflare Workers | Edge-runtime flagship |
| Deno (latest stable) | Growing usage in indie projects |
| Bun (latest stable) | Growing fast, common alternative to Node |
| Vercel Edge Functions | Serverless edge target |

## What gets checked

For each target:

1. Engine loads and instantiates without error
2. `embed("hello world")` returns a Float32Array of length 384 with norm = 1.0
3. Output matches the canonical reference (same vector across all targets — Wasm is deterministic)
4. Cold-start latency is under the per-target SLO (defined in `runtimes.yaml`)
5. No runtime warnings about missing Wasm features (SIMD, bulk memory, etc.)

Failure surfaces in `../results/v<X.Y.Z>.json` and gates publishing.

## Status

Pre-alpha. CI matrix pending — see `.github/workflows/build-engine.yml`.