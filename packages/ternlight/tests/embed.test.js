// Basic smoke test that the package's public API actually works end-to-end.
// Run with: node --test packages/ternlight/tests/
//
// Requires that `pkg/` has been populated by scripts/build-engine.sh.

const test = require('node:test');
const assert = require('node:assert/strict');

const { embed, cosineSim, similar, engineInfo, TernError } = require('../src/index.js');

test('embed returns a 384-dim Float32Array', () => {
    const v = embed('hello world');
    assert.ok(v instanceof Float32Array, 'embed must return a Float32Array');
    assert.equal(v.length, 384, 'embedding dim must be 384');
});

test('embed output is L2-normalized', () => {
    const v = embed('the quick brown fox');
    let norm = 0;
    for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
    norm = Math.sqrt(norm);
    assert.ok(Math.abs(norm - 1) < 1e-3, `expected ||v|| ≈ 1, got ${norm}`);
});

test('cosineSim of a vector with itself is ~1', () => {
    const v = embed('forgot my password');
    const sim = cosineSim(v, v);
    assert.ok(Math.abs(sim - 1) < 1e-4, `expected self-similarity ≈ 1, got ${sim}`);
});

test('cosineSim ranks semantically similar pairs higher than unrelated', () => {
    const v1 = embed('how do I reset my password');
    const v2 = embed('forgot my password');
    const v3 = embed('chocolate cake recipe');
    const close = cosineSim(v1, v2);
    const far   = cosineSim(v1, v3);
    assert.ok(close > far, `expected ${close} > ${far}`);
});

test('similar returns top-K matches sorted by similarity descending', () => {
    const corpus = [
        'I forgot my password and need to reset it',
        'where is my package shipment tracking',
        'how to cancel a recurring subscription',
    ];
    const matches = similar('forgot password', corpus, { topK: 2 });
    assert.equal(matches.length, 2, 'should return topK results');
    assert.equal(
        matches[0].text,
        'I forgot my password and need to reset it',
        'most similar match should be the password-reset string',
    );
    assert.ok(
        matches[0].sim > matches[1].sim,
        'results should be sorted descending by sim',
    );
});

test('embed throws TernError on non-string input', () => {
    assert.throws(
        () => embed(42),
        (err) => err instanceof TernError && err.code === 'INVALID_INPUT',
    );
});

test('cosineSim throws TernError on dim mismatch', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([1, 2, 3, 4]);
    assert.throws(
        () => cosineSim(a, b),
        (err) => err instanceof TernError && err.code === 'DIM_MISMATCH',
    );
});

test('engineInfo returns a non-empty config string', () => {
    const info = engineInfo();
    assert.equal(typeof info, 'string');
    assert.ok(info.length > 0);
    assert.ok(info.includes('embedding_format='), 'should mention embedding_format');
});
