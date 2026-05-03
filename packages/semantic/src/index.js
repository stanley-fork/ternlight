// @tern/semantic — public API
//
// Thin wrapper around the Wasm engine exported from engine/. The engine handles
// tokenization + forward pass; this package adds JS-side ergonomics:
// L2-normalized arithmetic helpers, top-K search, and a stable public surface.
//
// The Wasm and model artifacts live alongside this file as static assets:
//   ./engine.wasm   — compiled engine (copied here by scripts/build-engine.sh)
//   ./model.bin     — trained model (fetched from GitHub Release at publish time)

// TODO: implement once the engine pkg/ output is wired into the publish pipeline.

export async function embed(text) {
  throw new Error("@tern/semantic: embed() not yet implemented — engine wiring pending");
}

export function cosineSim(a, b) {
  if (a.length !== b.length) throw new Error("vector length mismatch");
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

export async function similar(query, corpus, opts = {}) {
  const topK = opts.topK ?? 3;
  const q = await embed(query);
  const scored = await Promise.all(
    corpus.map(async (text) => ({ text, sim: cosineSim(q, await embed(text)) })),
  );
  return scored.sort((a, b) => b.sim - a.sim).slice(0, topK);
}