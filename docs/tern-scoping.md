# @tern — Product Design Document

## 1. Vision

@tern is the **SQLite of semantic matching** — a zero-config, entirely self-contained semantic embedding engine for Node.js developers. It produces dense vector representations of short natural language strings entirely on-device, with zero external dependencies, no network calls, and no GPU required.

Everything @tern offers — similarity scoring, intent classification, semantic search — is built on top of this single primitive: **a good embedding, produced locally, in milliseconds.**

---

## 2. The Market Gap

Developers building modern web applications, edge functions, and CLI tools need intelligent text processing. The current ecosystem forces a polarized choice:

| Option | Problem |
|---|---|
| External APIs (OpenAI, Cohere) | Network latency, privacy risk, per-call cost, single point of failure |
| ONNX / Transformers.js | 20–100MB+ of weights, C++ bindings, complex setup |
| Statistical models (FastText, CBOW) | No contextual understanding, no attention mechanism, poor on paraphrases |

There is no option that delivers true attention-based semantic understanding at the scale of a standard npm package. @tern fills that gap by pioneering 1.58-bit ternary transformer architecture compiled to WebAssembly.

---

## 3. Target Audience

**Edge / Serverless Developers**
Building on Cloudflare Workers, Vercel Edge, or AWS Lambda where strict package size limits and cold-start penalties are enforced. These environments often prohibit native dependencies entirely.

**Local-First App Builders**
Creating browser extensions, Electron apps, or offline-capable tools requiring semantic search without a backend. The corpus lives on-device; the embeddings stay on-device.

**DevTool Creators**
Building CLI tools for log triage, code deduplication, or document routing. Typically processing high volumes of short strings where API costs at scale are prohibitive.

**Privacy-Constrained Deployments**
Medical, legal, industrial, and enterprise contexts where data cannot leave the device by policy or regulation. @tern is often the *only* viable option here, not just the convenient one.

---

## 4. Core Use Cases

### Embedding Generation *(the primitive)*
Produce a dense semantic vector from any short string. The developer owns what happens downstream — store it in a vector DB, compare against precomputed embeddings, feed it into a classifier, or build an offline search index. This is the foundational capability that all other use cases are built on.

> *"You get BERT-quality semantic representations without a network call, a GPU, or a 100MB dependency."*

### Semantic Similarity Matching
Compare two strings for semantic closeness. Compute embeddings for both strings, take cosine similarity, return a score. The natural fit for FAQ matching, duplicate detection, and paraphrase identification.

> Example: "my screen is black" matches "the display is broken" — a pattern regex cannot solve.

### Intent Classification Against a Fixed Label Set
Given a developer-defined set of categories (10–100 intents), route an input string to the closest one. Pre-compute label embeddings at startup; at runtime, embed the input and find the nearest neighbor. The chatbot routing, form triage, and log classification pattern.

### Client-Side Semantic Search
Pre-embed a known corpus (FAQ articles, documentation, product listings) at build time. Store the index locally. At query time, embed the search string and return nearest neighbors. No search backend required. The strongest use case for static sites, browser extensions, and local-first apps.

### High-Volume Recurring Classification
For pipelines already doing semantic classification via API — content moderation pre-filters, product categorization, log stream triage — @tern replaces the API call for high-confidence cases. The pattern: @tern handles the obvious 80%, escalate the uncertain 20% to a heavier model. Dramatically reduces per-call cost and latency at volume.

### Air-Gapped and Regulated Environments
Medical records, legal documents, industrial sensor data, enterprise compliance systems — anywhere data cannot egress by policy. All five use cases above apply here; the difference is @tern isn't a preference, it's a requirement.

---

## 5. The Design Pattern: Regex + Semantic

@tern does not replace regex or strict string matching. It acts as a **probabilistic funnel** upstream of deterministic code:

- **Regex** matches *format*. Use it when you control the vocabulary (validating a UUID, extracting an email). Fails completely on human phrasing variation.
- **Semantic matching** matches *intent*. Use it when dealing with unpredictable natural language ("locked out" = "forgot password").

The recommended pattern:

```js
const score = await tern.similarity(userInput, knownIntent);
if (score > 0.85) {
  // snap back to deterministic routing
  handleIntent(knownIntent);
}
```

Use @tern to score messy input against known system states. Once confidence crosses a threshold, return to strict if/else logic.

---

## 6. The @tern Ecosystem

The project ships under the `@tern` scoped namespace. Each package is a purpose-built micro-tool sharing one underlying Wasm engine — only the trained model head changes.

| Package            | Use Case                                     | Task Type             |
| ------------------ | -------------------------------------------- | --------------------- |
| **@tern/semantic** | Embedding generation, similarity matching    | Encoder, no head      |
| **@tern/classify** | Edge intent routing for serverless functions | Classification head   |
| **@tern/extract**  | Lightweight micro-NER for CLI and CI/CD      | Token classification  |

**The branding:** "Tern" is both the seabird — small, fast, built for long distances — and a direct abbreviation of *ternary*, the 1.58-bit quantization scheme that makes the extreme size reduction possible. The name works as a technical signal and a product identity simultaneously.

---

## 7. The Model Tiers

All tiers share a single Wasm engine. The only difference is the `.bin` model file. Developers upgrade tiers by changing one line of config; no engine rewrite is required.

| Tier | d_model | Layers | Params | Packed Size | Total Package | Intended For |
|---|---|---|---|---|---|---|
| **nano** | 128 | 2 | ~3M | ~750KB | ~1.5MB | Absolute size-constrained edge; lowest accuracy |
| **micro** *(default)* | 256 | 2 | ~7M | ~1.75MB | ~3MB | Standard npm use case; best size/quality balance |
| **base** | 384 | 2 | ~12M | ~3MB | ~4.5MB | Higher-accuracy distillation; near budget ceiling |

**Micro is the anchor tier.** It ships first, fits the 3–5MB constraint, and covers the largest portion of the use case matrix. Nano and base can be released later without touching the engine.

---

## 8. Key Constraints

These are non-negotiable product requirements, not preferences:

| Constraint | Target |
|---|---|
| **Total package size** | < 5MB |
| **External NPM dependencies** | Zero |
| **Platform requirement** | CPU only — no GPU, no Python, no C++ build tools |
| **License** | 100% Open Source |
| **Runtime** | Native CPU via Node.js / WebAssembly (V8) |

---

## 9. Performance Targets

| Metric | Target | Notes |
|---|---|---|
| **Total package size** | ~3MB (micro tier) | Model + Wasm engine + JS wrapper |
| **Cold-start load time** | < 15ms | Synchronous read of model binary into Wasm memory |
| **Single inference latency** | 1–5ms | CPU native execution, no network |
| **RAM footprint** | < 10MB | Negligible impact on serverless memory caps |
| **Context window** | 64–128 tokens | Optimized for queries and short sentences |
