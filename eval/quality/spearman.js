// eval/quality/spearman.js
//
// Distillation-fidelity quality metric. Loads a corpus of (text, teacher_emb)
// items produced by prep_spearman.py, runs the WASM engine to get student
// embeddings, builds M random sentence pairs, and computes Spearman rank
// correlation between student_cosine and teacher_cosine across the pairs.
//
// What the number means:
//   1.0  → student ranks pairs identically to teacher (perfect distillation)
//   0.7  → strong agreement; small reorderings
//   0.3  → weak; many pairs ranked differently
//   0.0  → no relationship; student lost the teacher's structure
//
// This is the "smoke test scaled up": same pair-cosine shape as smoke.js, but
// thousands of pairs and the "expected" comes from teacher data, not hand-set
// bands. A drop here vs. the Python QAT model's reported spearman would point
// to a packing/SIMD/scale bug in the WASM engine.
//
// Usage:
//   node eval/quality/spearman.js                                 # defaults: smoke_1k, 1000 pairs
//   node eval/quality/spearman.js --corpus spearman_smoke_1k.json --pairs 2000

const path = require('node:path');
const fs   = require('node:fs');

const ROOT       = path.resolve(__dirname, '..', '..');
const ENGINE_PKG = path.join(ROOT, 'engine', 'pkg', 'tern_engine');
const CORPUS_DIR = path.join(__dirname, 'corpus');

// ── Args ────────────────────────────────────────────────────────────────────

function arg(name, dflt) {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 ? process.argv[i + 1] : dflt;
}

const CORPUS_FILE = arg('corpus', 'spearman_smoke_1k.json');
const N_PAIRS     = parseInt(arg('pairs', '1000'), 10);
const SEED        = parseInt(arg('seed',  '42'),   10);

// ── Math helpers ────────────────────────────────────────────────────────────

function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// Mulberry32 — small, deterministic PRNG. Same seed → same pair selection across runs.
function mulberry32(seed) {
    let s = seed >>> 0;
    return function () {
        s = (s + 0x6D2B79F5) | 0;
        let t = s;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

// Average-rank assignment so ties don't bias Spearman. Float cosines almost
// never tie exactly, but the cost is trivial and it's the textbook formula.
function ranks(arr) {
    const n = arr.length;
    const idx = arr.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
    const r = new Array(n);
    let i = 0;
    while (i < n) {
        let j = i;
        while (j + 1 < n && idx[j + 1][0] === idx[i][0]) j++;
        const avg = (i + j) / 2 + 1; // 1-indexed average rank
        for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
        i = j + 1;
    }
    return r;
}

function pearson(x, y) {
    const n = x.length;
    let sx = 0, sy = 0;
    for (let i = 0; i < n; i++) { sx += x[i]; sy += y[i]; }
    const mx = sx / n, my = sy / n;
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        const ax = x[i] - mx, ay = y[i] - my;
        num += ax * ay;
        dx  += ax * ax;
        dy  += ay * ay;
    }
    return num / Math.sqrt(dx * dy);
}

function spearman(x, y) { return pearson(ranks(x), ranks(y)); }

// ── Run ─────────────────────────────────────────────────────────────────────

const { embed, config_summary } = require(ENGINE_PKG);

const corpusPath = path.join(CORPUS_DIR, CORPUS_FILE);
const items = JSON.parse(fs.readFileSync(corpusPath, 'utf8'));
const N = items.length;

console.log('');
console.log(`  build:        ${config_summary()}`);
console.log(`  corpus:       ${CORPUS_FILE}  (${N} sentences)`);

// Student embeddings via WASM
const tStart = Date.now();
const studentVecs = items.map(it => Array.from(embed(it.text)));
const teacherVecs = items.map(it => it.teacher_emb);
const embedMs = Date.now() - tStart;

// Sample N_PAIRS unique unordered (i, j) without replacement
const maxPairs = (N * (N - 1)) / 2;
const target   = Math.min(N_PAIRS, maxPairs);
const rng      = mulberry32(SEED);
const seen     = new Set();
const pairs    = [];
while (pairs.length < target) {
    const i = Math.floor(rng() * N);
    const j = Math.floor(rng() * N);
    if (i === j) continue;
    const key = i < j ? `${i},${j}` : `${j},${i}`;
    if (seen.has(key)) continue;
    seen.add(key);
    pairs.push([i, j]);
}

const studentCos = pairs.map(([i, j]) => cosine(studentVecs[i], studentVecs[j]));
const teacherCos = pairs.map(([i, j]) => cosine(teacherVecs[i], teacherVecs[j]));

const rho = spearman(studentCos, teacherCos);
const r   = pearson(studentCos, teacherCos);

console.log(`  pairs:        ${pairs.length}  (seed=${SEED})`);
console.log(`  embed time:   ${embedMs} ms  (${(embedMs / N).toFixed(1)} ms/sentence)`);
console.log('');
console.log(`  spearman(student vs teacher):  ${rho.toFixed(4)}`);
console.log(`  pearson (student vs teacher):  ${r.toFixed(4)}`);
console.log('');
