# eval/benchmarks

Performance measurements of the shipped engine across deployment targets.

## What gets measured

- **Latency** — cold start (first call), warm steady-state, per-target
- **Throughput** — strings/sec under continuous load
- **Memory** — peak RSS, steady-state, allocations per call
- **Bundle size** — .wasm bytes, .bin bytes, total npm install footprint, gzipped over the wire

## Layout

```
benchmarks/
├── latency.js        cold + warm latency, single-call
├── throughput.js     strings/sec under sustained load
├── memory.js         RSS profiling
└── bundle-size.js    measures the published artifact sizes
```

Each script writes its output to `../results/v<X.Y.Z>.json` (merged with the rest by `scripts/run-eval.sh`).

## Per-target matrix

For each release, benchmarks run against:

- Node.js (current LTS, 18, 20, 22)
- Cloudflare Workers (via Miniflare)
- Deno (current stable)
- Bun (current stable)
- Browser (Chromium, headless via Playwright)

Per-target results live in `compatibility/` for portability tracking; aggregated perf summaries live here.

## Status

Pre-alpha — bench scripts pending. The current ~570 ms / call number in the README is from an unoptimized debug build; expect significant improvement from SIMD + cached weight unpacking.