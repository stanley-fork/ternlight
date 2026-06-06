# Ternlight docs-search demo

Standalone single-page demo: semantic search over the [React docs](https://react.dev)
"Learn" section, running entirely in the browser via ternlight's WASM engine.

**~1,987 doc paragraphs · 7 MB wasm · ~4 second one-time indexing · sub-3 ms per query.**

## Files

```
examples/docs-search/
├── index.html                 single-page UI (indexing + search phases)
├── style.css                  editorial styles, no framework
├── main.js                    ESM module: load wasm, index, search
├── chunks.json                1,987 paragraph-level chunks of React docs (~1.2 MB)
├── scripts/
│   └── gen-corpus.js          one-shot offline prep script (committed for reproducibility)
├── wasm/                      web-target wasm-pack output (int4 ship build)
│   ├── tern_engine.js
│   ├── tern_engine_bg.wasm    ~7.25 MB (model + tokenizer + engine, all baked in)
│   └── *.d.ts
└── README.md                  this file
```

The `wasm/` directory is a web-target build
(`wasm-pack build --target web --features emb_int4`). The engine's
`model.bin` and `tokenizer.json` are embedded directly into the wasm via
Rust's `include_bytes!`, so the demo needs nothing beyond the files in
this directory.

## Run it locally

Wasm modules can't be loaded from `file://` URLs — you need a real HTTP
server. Any will do; pick one:

```bash
# Option 1 — Python (already on macOS)
cd examples/docs-search
python3 -m http.server 8000

# Option 2 — Node, one-liner
cd examples/docs-search
npx serve -p 8000
```

Open <http://localhost:8000> in a browser.

## What you should see

1. **Indexing phase** (~3-5 seconds) — a progress bar with a live throughput
   counter ("XXX emb/sec"). The engine is loading the corpus, embedding
   every paragraph against the ternlight model.
2. **Search appears** once indexing finishes — autofocused input.
3. **Type or click a suggested query.** Results appear within 2-3 ms per
   keystroke. The latency badge to the right of the input updates live.
4. **Footer** shows: engine config (int4), the indexing summary, network
   requests since indexing (should stay at 0 unless you click a result),
   and the wasm size.

## Try these queries

These are designed to show the semantic > keyword wins — each finds the
right doc paragraph even when the query doesn't share keywords with the
listing:

| Query | What it should match |
|---|---|
| "how do I run something when state changes" | useEffect docs |
| "share state between components" | Lifting State Up docs |
| "pass data deep without prop drilling" | Context docs |
| "why is my component re-rendering" | Render and Commit docs |
| "remember a value across renders" | useRef / useState docs |
| "skip a re-render" | React.memo / useMemo docs |

A naive keyword search would miss many of these — none of them share words
with the page they should match.

## Re-generate `chunks.json` from a different react.dev clone

The prep script in `scripts/gen-corpus.js` walks react.dev's
`src/content/learn/` directory, strips MDX/JSX components we can't render
meaningfully (`<Sandpack>`, `<YouWillLearn>`, etc.), splits prose into
paragraphs, and prepends the heading-path breadcrumb to each chunk.

```bash
node examples/docs-search/scripts/gen-corpus.js /path/to/react.dev
```

The script is deterministic — same input directory, same output JSON.

## Rebuild the wasm

If you change the engine code, rebuild the web-target output:

```bash
# Build to engine/pkg first (wasm-pack 0.14 + cargo 1.94 has an --out-dir
# bug we work around by moving the output)
mv engine/pkg engine/pkg-backup
(cd engine && wasm-pack build --target web --release --features emb_int4)
rm -rf examples/docs-search/wasm
mv engine/pkg examples/docs-search/wasm
mv engine/pkg-backup engine/pkg
```

This pattern is exactly what `scripts/build-engine.sh` does for the
ternlight npm package, just with `--target web` instead of `--target nodejs`.

## Deploying to Vercel

1. Connect Vercel to your GitHub repo
2. **Root Directory**: `examples/docs-search`
3. **Framework Preset**: Other
4. **Build Command**: (empty)
5. **Output Directory**: `.`

Vercel sets `Content-Type: application/wasm` correctly by default. Add a
`vercel.json` at the repo root if you want explicit cache headers for the
wasm file (recommended — see `vercel.json`).

## Attribution

The bundled corpus is derived from the React documentation source at
[github.com/reactjs/react.dev](https://github.com/reactjs/react.dev),
licensed CC BY 4.0. Only the prose paragraphs are included; code samples,
interactive demos (`<Sandpack>` blocks), and React-specific JSX components
are stripped during the offline prep step. Each result card links back to
the source page on react.dev.
