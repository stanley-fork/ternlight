# @tern/semantic

> On-device semantic embeddings for JavaScript. Embed text, compare similarity, find nearest neighbors — locally, no API keys, no backend.

## Install

```bash
npm install @tern/semantic
```

The package bundles the compiled Wasm engine (`engine.wasm`, ~600 KB after `wasm-opt`) and the trained model (`model.bin`, ~2.8 MB). No postinstall download. No network calls at runtime.

## Use

```js
import { embed, cosineSim, similar } from '@tern/semantic';

// 1. Embed a single string → 384-dim L2-normalized Float32Array
const v1 = await embed("how do I reset my password");
const v2 = await embed("forgot my password");

// 2. Compare two embeddings
cosineSim(v1, v2);                                  // → 0.78

// 3. Nearest-neighbor search over a corpus
const corpus = [
  "I forgot my password and need to reset it",
  "where is my package shipment tracking",
  "how to cancel a recurring subscription",
];
const matches = await similar("forgot password", corpus, { topK: 3 });
// → [{ text: "...", sim: 0.81 }, ...]
```

## API

### `embed(text: string): Promise<Float32Array>`

Returns a 384-dim L2-normalized embedding. Output is on the unit hypersphere — cosine similarity reduces to dot product.

### `cosineSim(a: Float32Array, b: Float32Array): number`

Cosine similarity between two L2-normalized vectors. Range `[-1, 1]`; for typical text embeddings, output is in `[0, 1]`.

### `similar(query: string, corpus: string[], opts?: { topK?: number }): Promise<{ text: string, sim: number }[]>`

Convenience: embed query, embed each corpus item, return top-K matches sorted descending by similarity.

For large corpora, embed the corpus once at startup and reuse the vectors:

```js
const corpusEmbeds = await Promise.all(corpus.map(embed));
// ... later:
function search(query) {
  const q = await embed(query);
  return corpusEmbeds
    .map((v, i) => ({ text: corpus[i], sim: cosineSim(q, v) }))
    .sort((a, b) => b.sim - a.sim)
    .slice(0, 3);
}
```

## Performance

Numbers from a M-series Mac, debug build:

- **Cold start:** ~600 ms (one-time, includes Wasm instantiation + tokenizer load)
- **Per-call latency:** ~570 ms (will improve significantly with SIMD + cached weight unpacking — see `eval/REPORT.md` for the latest)
- **Throughput:** ~1.8 strings/sec
- **Bundle:** ~3 MB total (engine.wasm + model.bin), gzipped ~2 MB on the wire

Performance work is active — track latest numbers in [`../../eval/REPORT.md`](../../eval/REPORT.md).

## Quality

The shipped model is distilled from `sentence-transformers/all-MiniLM-L6-v2`. Quality on standard benchmarks (full scorecard in [`../../eval/REPORT.md`](../../eval/REPORT.md)):

| Metric | Value | vs Teacher |
|---|---|---|
| STS-B AUC | 0.84 | -0.02 |
| Mean teacher cosine sim | 0.81 | n/a (this IS the alignment) |
| Recall@3 (general retrieval) | 0.75 | matches |

The model is small (~9M params, ~3 MB shipped). It's purpose-built for short-string similarity (queries, intents, FAQs) — not long-document understanding.

## Status

**v0.1, pre-alpha.** Not yet on npm. Tracking issues and roadmap at [https://github.com/wenshutang/tern-vec](https://github.com/wenshutang/tern-vec).