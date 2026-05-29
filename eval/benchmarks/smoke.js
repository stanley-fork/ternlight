// eval/benchmarks/smoke.js
//
// Semantic similarity sanity test. Loads N sentence pairs from corpus/smoke-pairs.json,
// embeds both sentences in each pair via the WASM engine, computes cosine similarity,
// prints per-pair verdict against the expected band.
//
// Output is human-readable (not JSON). Run after every engine build before any
// quality/perf testing — catches "the model loaded but produces garbage" failure modes
// that aggregate metrics (Spearman, NDCG@10) would mask.
//
// Usage:
//   node eval/benchmarks/smoke.js
//
// Assumes engine/pkg/ exists (build with `wasm-pack build --target nodejs --features <emb_*>`).

const path = require('node:path');
const fs   = require('node:fs');

const ROOT       = path.resolve(__dirname, '..', '..');
const ENGINE_PKG = path.join(ROOT, 'engine', 'pkg', 'tern_engine');
const CORPUS     = path.join(__dirname, 'corpus', 'smoke-pairs.json');

const { embed, config_summary } = require(ENGINE_PKG);

function cosine(a, b) {
    if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function verdictFor(score, lo, hi) {
    if (lo === null || hi === null) return { tag: 'TBD ', color: '\x1b[33m' };   // yellow
    if (score >= lo && score <= hi)  return { tag: 'OK  ', color: '\x1b[32m' };   // green
    return                              { tag: 'FLAG', color: '\x1b[31m' };       // red
}

function fmtBand(lo, hi) {
    if (lo === null && hi === null) return '   (tbd)   ';
    return `[${lo.toFixed(2)}, ${hi.toFixed(2)}]`;
}

function truncate(s, n) {
    return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

// ── Run ────────────────────────────────────────────────────────────────────

console.log(`\nbuild:  ${config_summary()}\n`);

const pairs = JSON.parse(fs.readFileSync(CORPUS, 'utf8'));

const RESET = '\x1b[0m';
console.log(`  ${'label'.padEnd(28)}  ${'score'}  ${'expected'.padEnd(14)}  pair`);
console.log(`  ${'-'.repeat(28)}  ${'-----'}  ${'-'.repeat(14)}  ${'-'.repeat(60)}`);

let n_ok = 0, n_flag = 0, n_tbd = 0;
for (const p of pairs) {
    const va = embed(p.a);
    const vb = embed(p.b);
    const score = cosine(va, vb);
    const v = verdictFor(score, p.expected_low, p.expected_high);
    if (v.tag === 'OK  ') n_ok++;
    else if (v.tag === 'FLAG') n_flag++;
    else n_tbd++;

    const pair_str = `"${truncate(p.a, 26)}" ↔ "${truncate(p.b, 26)}"`;
    console.log(
        `  ${v.color}${v.tag}${RESET} ${p.label.padEnd(22)}  ${score.toFixed(3)}  ${fmtBand(p.expected_low, p.expected_high)}  ${pair_str}`
    );
}

console.log('');
console.log(`  ${n_ok} OK   ${n_flag} FLAG   ${n_tbd} TBD   /  ${pairs.length} total`);
console.log('');

// Notes on TBD pairs — print full descriptions so the reader can interpret the scores.
const tbd = pairs.filter(p => p.expected_low === null);
if (tbd.length > 0) {
    console.log('  TBD pair notes (no fixed expectation):');
    for (const p of tbd) {
        console.log(`    [${p.label}] ${p.notes}`);
    }
    console.log('');
}

process.exit(n_flag > 0 ? 1 : 0);
