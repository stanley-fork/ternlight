// eval/benchmarks/perf.js
//
// First-pass perf baseline for the shipped WASM artifact. Measures:
//   - Cold-start latency: require() + first embed() call (includes lazy model init)
//   - Per-query latency: p50, p95, p99, min, max over N samples of repeated queries
//   - Memory: RSS + heap_used after warmup
//   - Bundle size: .wasm + .bin from disk
//
// Output:
//   stdout — single JSON object (machine-readable; pipe to a file under results/)
//   stderr — human-readable summary table
//
// Usage:
//   node eval/benchmarks/perf.js > eval/benchmarks/results/<date>-<commit>-<feature>.json
//
// Deferred for now:
//   - WASM linear memory size (need wasm-bindgen plumbing; not exposed by default Node bindings)
//   - Throughput in qps (derivable from p50)
//   - Per-runtime portability (Cloudflare Workers, Deno, Bun) — see eval/compatibility/

const { performance } = require('node:perf_hooks');
const path = require('node:path');
const fs   = require('node:fs');
const { execSync } = require('node:child_process');
const os = require('node:os');

const ROOT       = path.resolve(__dirname, '..', '..');
const ENGINE_PKG = path.join(ROOT, 'engine', 'pkg');
const ASSETS     = path.join(ROOT, 'engine', 'assets');

// ── Hardcoded query set (v1) ───────────────────────────────────────────────
//
// ~15 representative strings spanning typical lengths. Each query runs N_REPS times
// to give a per-query distribution; total samples = queries.length × N_REPS.
// Real workload corpora (e.g., SciFact's 300 queries) belong upstream as a data
// dependency, not embedded here.

const QUERIES = [
    "reset my password",
    "how do I cancel my subscription",
    "what is the meaning of life",
    "fix npm install error EACCES",
    "best Python web framework for REST APIs",
    "why is my docker container crashing",
    "how to write unit tests in Rust",
    "TypeScript generics tutorial",
    "what does CORS error mean",
    "how to deploy a Next.js app to Vercel",
    "machine learning vs deep learning",
    "configure SSH for GitHub",
    "regex to match email addresses",
    "kubernetes pod stuck in CrashLoopBackOff",
    "how to set up a monorepo with pnpm workspaces",
];

const N_REPS = 10;   // each query repeated → samples = QUERIES.length × N_REPS

// ── 1) Cold start ──────────────────────────────────────────────────────────

const t_req_start = performance.now();
const { embed, config_summary } = require(path.join(ENGINE_PKG, 'tern_engine'));
const t_require_ms = performance.now() - t_req_start;

// First embed() call includes lazy model init (sha256 verify + layout walk +
// RuntimeWeights decode). Use a short throwaway string so we measure init cost
// roughly independent of query-length-driven inference cost.
const t_first_start = performance.now();
embed("warmup");
const t_first_ms = performance.now() - t_first_start;

const t_cold_total_ms = t_require_ms + t_first_ms;

// ── 2) Steady-state per-query latency ──────────────────────────────────────

// Warmup additional calls so JIT / caches are stable before measurement.
for (let i = 0; i < 5; i++) embed(QUERIES[i % QUERIES.length]);

const samples = [];
for (let rep = 0; rep < N_REPS; rep++) {
    for (const q of QUERIES) {
        const t = performance.now();
        embed(q);
        samples.push(performance.now() - t);
    }
}
samples.sort((a, b) => a - b);

function pct(arr, p) {
    const idx = Math.min(arr.length - 1, Math.floor(arr.length * p));
    return arr[idx];
}

const per_query_ms = {
    n:    samples.length,
    min:  +samples[0].toFixed(2),
    p50:  +pct(samples, 0.50).toFixed(2),
    p95:  +pct(samples, 0.95).toFixed(2),
    p99:  +pct(samples, 0.99).toFixed(2),
    max:  +samples[samples.length - 1].toFixed(2),
    mean: +(samples.reduce((s, v) => s + v, 0) / samples.length).toFixed(2),
};

// ── 3) Memory (post-warmup snapshot) ───────────────────────────────────────

const mem = process.memoryUsage();
const memory_mb = {
    rss:       +(mem.rss       / 1024 / 1024).toFixed(2),
    heap_used: +(mem.heapUsed  / 1024 / 1024).toFixed(2),
    heap_total:+(mem.heapTotal / 1024 / 1024).toFixed(2),
    external:  +(mem.external  / 1024 / 1024).toFixed(2),
};

// ── 4) Bundle sizes ────────────────────────────────────────────────────────

function fileSize(p) { try { return fs.statSync(p).size; } catch { return null; } }

const wasm_bytes = fileSize(path.join(ENGINE_PKG, 'tern_engine_bg.wasm'));
const bin_bytes  = fileSize(path.join(ASSETS, 'model.bin'));
const tok_bytes  = fileSize(path.join(ASSETS, 'tokenizer.json'));

const bundle_bytes = {
    wasm:      wasm_bytes,
    bin:       bin_bytes,
    tokenizer: tok_bytes,
    total:     (wasm_bytes || 0) + (bin_bytes || 0) + (tok_bytes || 0),
};

// ── 5) Provenance ──────────────────────────────────────────────────────────

let commit = 'unknown';
try { commit = execSync('git rev-parse --short HEAD', { cwd: ROOT, encoding: 'utf8' }).trim(); } catch {}

const host = `${os.platform()}-${os.arch()} Node ${process.version}  ${os.cpus()[0].model}`;

// ── Output ──────────────────────────────────────────────────────────────────

const out = {
    build:           config_summary(),
    commit,
    host,
    timestamp:       new Date().toISOString(),
    cold_start_ms: {
        require:         +t_require_ms.toFixed(2),
        first_inference: +t_first_ms.toFixed(2),
        total:           +t_cold_total_ms.toFixed(2),
    },
    per_query_ms,
    memory_mb,
    bundle_bytes,
};

// JSON to stdout — clean for piping into results/<...>.json
process.stdout.write(JSON.stringify(out, null, 2) + '\n');

// Human-readable summary to stderr — visible in the terminal even when stdout is redirected
const MB = b => (b / 1024 / 1024).toFixed(2) + ' MB';
const lines = [
    '',
    `  build:           ${out.build}`,
    `  commit:          ${out.commit}`,
    `  host:            ${out.host}`,
    '',
    `  cold start:      require=${out.cold_start_ms.require} ms  first=${out.cold_start_ms.first_inference} ms  total=${out.cold_start_ms.total} ms`,
    `  per-query (ms):  p50=${per_query_ms.p50}  p95=${per_query_ms.p95}  p99=${per_query_ms.p99}  min=${per_query_ms.min}  max=${per_query_ms.max}  n=${per_query_ms.n}`,
    `  memory:          rss=${memory_mb.rss} MB  heap=${memory_mb.heap_used} MB`,
    `  bundle:          total=${MB(bundle_bytes.total)}  (.wasm=${MB(bundle_bytes.wasm)}  .bin=${MB(bundle_bytes.bin)}  tokenizer=${MB(bundle_bytes.tokenizer)})`,
    '',
];
process.stderr.write(lines.join('\n'));
