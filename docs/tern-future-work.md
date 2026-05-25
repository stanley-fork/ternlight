# @tern — Future Work & Optimization Opportunities

> This document captures known limitations, deferred optimizations, and follow-on research directions. It is not a roadmap — nothing here is committed. It exists so promising directions aren't lost and so active phase documents stay focused on current scope.

---

## 1. Distillation: Stronger Teacher Model

**Current approach:** `all-MiniLM-L6-v2` (22M params, 384-dim output) as the teacher. Chosen for the prototype because its output dimension is close to the student's (`d_model=256`), making the projection layer small and well-conditioned.

**The opportunity:** The teacher never ships — it is discarded after training. A stronger teacher produces higher-quality soft targets, which sets a higher ceiling for the student's learned representations. `mixedbread-ai/mxbai-embed-large-v1` (335M params, 1024-dim, state-of-the-art on MTEB) is the natural upgrade candidate.

The reason this is deferred rather than used from the start: the 256→1024 output projection is large and noisier to train through for a shallow 2-layer student. Once the distillation pipeline is proven stable with the simpler setup, swapping the teacher is a one-line config change (`teacher: "mixedbread-ai/mxbai-embed-large-v1"`) followed by a retrain. A direct quality comparison between the two teachers on the eval tasks will tell you whether the upgrade is worth it.

**Implementation complexity:** Trivial — change one config value, retrain. The output projection layer automatically resizes because it is config-driven (`d_model → output_dim`). Update `output_dim: 1024` in the config to match the new teacher.

---

## 2. Distillation: Intermediate Layer Alignment

**Current approach:** Output-only distillation — the student minimizes cosine distance between its final embedding and the teacher's final embedding. This is the correct baseline but leaves significant signal on the table.

**The opportunity:** Intermediate layer distillation, pioneered by DistilBERT and refined in TinyBERT, aligns the student's hidden states at each layer to corresponding layers of the teacher. The student learns not just *what* the teacher outputs, but *how* the teacher progressively abstracts meaning across depth.

For @tern's 2-layer student learning from a 6-layer teacher, the alignment would be:

```
Student layer 1 hidden state  →  Teacher layer 2 hidden state
Student layer 2 hidden state  →  Teacher layer 4 hidden state
Student final embedding        →  Teacher final embedding
```

Each alignment adds a loss term — typically cosine similarity or MSE after a learned linear projection to bridge the dimension mismatch (student d_model=256 → teacher d_model=384):

```python
loss = (
    cosine_loss(student_final, teacher_final)           # output alignment
  + λ1 * cosine_loss(proj1(student_layer1), teacher_layer2)  # intermediate
  + λ2 * cosine_loss(proj2(student_layer2), teacher_layer4)  # intermediate
)
```

The projection layers (`proj1`, `proj2`) are float32 linear layers used only during training and discarded at export — same as the output projection head.

**Why this matters for @tern specifically:** A 2-layer student has very little depth to work with. Without intermediate guidance, each layer has to figure out independently how to use its capacity. With intermediate alignment, the student gets an explicit signal about how layer 1 should behave (like teacher layer 2) and how layer 2 should behave (like teacher layer 4). This is significantly more informative than a single loss signal at the end.

**Implementation complexity:** Medium. Requires hooking into the teacher's intermediate activations during the training loop. The projection layers add a small number of float32 parameters that are training-only.

---

## 3. Distillation: Attention Matrix Transfer

**Current approach:** Distillation operates on hidden states (vectors per token). Attention matrices (the full `n_heads × seq_len × seq_len` attention weights) are not used.

**The opportunity:** TinyBERT showed that aligning attention weight matrices — not just hidden states — provides a strong structural signal. The student learns which token pairs the teacher attends to, not just what the resulting representations look like. This is particularly valuable for a shallow student because attention patterns encode syntactic and semantic relationships that would otherwise require depth to discover.

For @tern, where inputs are short (64–128 tokens) and the task is semantic matching rather than generation, attention transfer may yield disproportionate gains because the attention patterns over short strings are compact and learnable.

**Implementation complexity:** Medium-high. Requires aligning per-head attention matrices, which adds memory overhead during training. The alignment loss must be careful about head permutation — teacher head 3 doesn't necessarily correspond to student head 3.

---

## 4. Distillation: Hard Negative Mining

**Current approach:** Training corpus is a set of (text, teacher_embedding) pairs. The loss is over individual strings in isolation.

**The opportunity:** Contrastive training with hard negatives — pairs of strings that are semantically close but not identical — forces the model to develop finer-grained discrimination. For @tern's primary use case (FAQ matching, intent routing), the failure mode is false positives: "how do I reset my password" matching "how do I reset my username" above a confidence threshold.

Hard negatives can be mined automatically: find pairs where the teacher assigns moderate cosine similarity (e.g., 0.5–0.7) — close enough to be plausible, far enough to be distinguishable. Training the student to correctly order these pairs improves precision at the tail of the score distribution.

**Implementation complexity:** Medium. Requires offline mining of hard negative pairs from the training corpus using the teacher. The training loop becomes triplet or contrastive rather than purely regression-based.

---

## 5. Quantization: Higher-Precision Embedding Table

**Current approach:** The embedding table is post-training quantized to ternary at export (~1.95MB). This is a deliberate deviation from the BitNet b1.58 paper — see `tern-phase1-prototype.md` Section 5 for the full reasoning.

**Context from the paper:** BitNet b1.58 keeps embeddings at full float precision throughout training and inference, explicitly stating this is required for language model token sampling. @tern is an encoder, not a language model, so that rationale does not apply. However, the paper establishes that higher-precision embeddings are the principled choice when size budget allows — the "1.58-bit" claim excludes embeddings in the paper's own reported numbers.

**The opportunity:** The embedding table is the most sensitive parameter block to quantization — each row is looked up independently, with no averaging effect across a matrix multiply to compensate for weight errors. Moving from ternary to a higher precision could improve embedding quality, particularly for rare tokens at the tail of the vocabulary distribution.

**Size trade-off (micro tier, 30k vocab):**

```
Ternary (current):         30,522 × 256 × 2 bits  = ~1.95MB
Int8 embedding:            30,522 × 256 × 8 bits  = ~7.8MB   (too large at 30k vocab)
Float16 embedding:         30,522 × 256 × 16 bits = ~15.6MB  (way too large)
```

With 30k vocab, any precision above ternary breaks the size budget. **This optimization is only viable in combination with Section 8 (custom BPE vocabulary):**

```
10k vocab + ternary:   10,000 × 256 × 2 bits  = ~0.64MB  (better size, lower precision)
10k vocab + int8:      10,000 × 256 × 8 bits  = ~2.56MB  (fits budget, better quality)
10k vocab + float16:   10,000 × 256 × 16 bits = ~5.12MB  (borderline)
```

A custom 10k-token BPE vocabulary combined with int8 embeddings is the most promising path to matching the paper's higher-precision embedding approach while staying within budget.

**When to revisit:** After Milestone 5 eval is run on the post-training-quantized checkpoint. If the quality gap between the float32 training checkpoint and the ternary-exported model is significant, this optimization (combined with Section 8) becomes Phase 2 scope rather than future work.

**Implementation complexity:** High. Requires the custom BPE vocabulary (Section 8), changes to the export script, and updated Wasm engine unpacking logic to handle mixed-precision weight storage.

---

## 6. Wasm Engine: SIMD Optimisation

**Current approach:** Phase 2 implements baseline ternary inference without SIMD. Correctness is validated first.

**The opportunity:** WebAssembly SIMD (`simd128`) allows processing 128 bits at once — equivalent to 64 ternary weight comparisons per instruction cycle. For the inner loop of a BitLinear forward pass (iterating over weight rows), SIMD offers potential 4–8× throughput improvement on supported runtimes.

All modern targets support SIMD: Node.js (V8), Chrome/Firefox, Cloudflare Workers.

**Implementation path:** After Phase 2 baseline is validated, add a SIMD code path via Rust's `std::arch::wasm32` intrinsics. Keep the scalar path as a fallback for environments that don't declare SIMD support at instantiation time.

**Implementation complexity:** Medium. SIMD intrinsics in Rust are stable for wasm32. The main work is restructuring the weight storage layout to align to 16-byte boundaries for SIMD loads.

---

## 7. Wasm Engine: Batch Inference

**Current approach:** Single string in, single embedding out. Each call pays the full Wasm function call overhead.

**The opportunity:** For use cases like pre-embedding a corpus of 500 FAQ documents at startup, batch inference amortises the overhead. A `embed_batch(texts: &[&str]) -> Vec<Vec<f32>>` export that processes multiple strings in a single Wasm call would improve throughput significantly for offline indexing workflows.

With SIMD (Section 5), batch processing also enables cross-sequence parallelism — different sequences can be packed into SIMD lanes.

**Implementation complexity:** Low-medium. The single-string path is already written; batch is a loop over it with optimised memory reuse.

---

## 8. Vocabulary: Custom Domain BPE

**Current approach:** BERT's `bert-base-uncased` vocabulary (30,522 tokens). General English with some tech coverage.

**The opportunity:** A custom BPE vocabulary trained specifically on the @tern target corpus (Stack Overflow, GitHub issues, documentation, chatbot transcripts, error messages) would:

1. **Reduce vocabulary size** — a 10k-token BPE over this domain likely covers >98% of tokens encountered at inference, vs. a 30k general vocabulary with many irrelevant entries
2. **Improve tokenization of developer terms** — BERT's vocabulary often splits `camelCase`, `snake_case`, version strings, and error codes into fragments that don't carry semantic meaning. A domain-specific vocabulary would have whole-token entries for common patterns
3. **Reduce embedding table size** — 10k × 256 at ternary = ~640KB vs. the current 1.95MB

**The cost:** Requires retraining from scratch with the new vocabulary. The parity test in Phase 2 would also need to be re-run against the new tokenizer.

**Implementation complexity:** Medium for vocabulary training (SentencePiece or HuggingFace tokenizers trainer). High overall because it touches training, export, and the Wasm vocab asset.

---

## 9. Tokenizer: WASI Target for Server Environments

**Current decision:** `wasm32-unknown-unknown` — no WASI, maximum compatibility with edge/browser runtimes.

**The deferred opportunity:** For server-side Node.js deployments where the user is not constrained by edge runtime limitations, a `wasm32-wasip1` build would unlock OS threading support, enabling rayon-based batch parallelism and potentially a simpler Wasm build with fewer dependency constraints.

This would ship as an optional alternative binary — same model file, different engine build. The JS wrapper would detect the runtime and load the appropriate `.wasm` variant.

**When to revisit:** If batch inference (Section 6) becomes a primary use case and the SIMD path alone doesn't provide sufficient throughput for large offline indexing jobs.

---

## 10. Formal Benchmarking Against STS-B and MTEB

**Current approach:** Phase 1 eval uses a proxy STS task on a manually curated corpus.

**The opportunity:** Semantic Textual Similarity Benchmark (STS-B) and the Massive Text Embedding Benchmark (MTEB) are the standard evaluation suites for embedding models. Publishing @tern's scores on these benchmarks provides an objective, comparable quality signal for potential adopters — and forces honest comparison against MiniLM, TinyBERT, and similar lightweight models.

Given the size constraint, @tern should not be expected to match MiniLM on general STS-B. The relevant comparison is: does @tern at ~2.5MB deliver quality comparable to MiniLM at 80MB on the specific task types @tern targets (short-string similarity, intent routing)?

**Implementation complexity:** Low. STS-B is a standard dataset; running the eval is straightforward once the Wasm engine is operational.

---

## 11. Fine-Tuned Variants

**Current scope:** One general-purpose micro-tier model for English/tech text.

**Future variants worth considering:**

| Variant | Corpus Focus | Use Case |
|---|---|---|
| `@tern/semantic-code` | GitHub code, Stack Overflow | Code search, deduplication |
| `@tern/semantic-support` | Customer support transcripts | FAQ routing, ticket triage |
| `@tern/semantic-multilingual` | mBERT-distilled, top 10 languages | Non-English markets |

Each variant would use the same Wasm engine binary — only the `.bin` model file swaps. This is the architectural bet from the product doc: one engine, many models.

**Implementation complexity:** Medium per variant. Requires domain-specific training corpus curation and a separate distillation run.

---

## 12. Pre-Computed Index Format

**Current scope:** @tern produces embeddings. Indexing and nearest-neighbour search are the caller's responsibility.

**The opportunity:** An optional `@tern/index` package that provides a compact, pre-computed HNSW or flat cosine index over a static corpus. The developer embeds their corpus at build time, ships the index as a static asset, and calls `index.search(query, topK)` at runtime. No backend required.

This is the strongest use case for static sites, documentation search, and browser extensions — and is currently underserved by the ecosystem.

**Implementation complexity:** Medium-high. HNSW in Wasm is non-trivial; a flat cosine index over small corpora (<10k documents) is straightforward.

---

## 13. Roadmap: Improving Semantic Similarity Quality

**Context:** Spot-check testing of the Phase 2 Wasm engine revealed that the current model leans heavily on **lexical (token) overlap** when computing similarity. Pairs with shared keywords score high; pairs with the same meaning but different wording score lower than intuition expects:

```
"reset my password"     ↔ "I forgot my password"        → 0.80   (high overlap on "password")
"oauth token expired"   ↔ "refresh oauth access token"  → 0.66   (high overlap on "oauth/token")
"cancel my subscription" ↔ "how do I unsubscribe"       → 0.24   (zero overlap — score is low)
"webpack config not working" ↔ "troubleshoot webpack errors" → 0.40 (partial overlap)
```

Top-1 retrieval rankings remain correct (matches Phase 1 eval R@3 = 0.80). The issue is **absolute confidence scores** for paraphrases without shared tokens. This is intrinsic to a small (9.5M param) distilled model trained on a single dataset (MS MARCO, 150k samples) — not an engine bug.

This section synthesizes the levers available to improve this, in priority order. Individual sections above cover each lever in detail.

### Tier 1 — Highest impact / lowest effort (~1 day)

**Diversify the training corpus.** MS MARCO is almost entirely questions. The student never learned that `"unsubscribe"` ≈ `"cancel subscription"` because it never saw paraphrase pairs. The teacher knows — its embeddings for both are close — so distilling on diverse phrasings implicitly transfers that knowledge to the student. No loss-function changes needed.

| Source | What it adds | HuggingFace ID |
|---|---|---|
| Quora Question Pairs | Paraphrase pairs (directly addresses the failure mode) | `quora` |
| AllNLI | Statements (not questions) | `sentence-transformers/all-nli` |
| ParaNMT | Multi-version paraphrases | `sentence-transformers/parallel-sentences-talks` |
| Stack Overflow titles | Developer phrasings | `pacovaldez/stackoverflow-questions` |

A 50/30/20 split across these (replacing pure MS MARCO) at the same 150k–300k scale would substantially close the paraphrase gap. Estimated improvement: pairs like `"cancel sub" ↔ "unsubscribe"` move from ~0.24 to ~0.65+. Documented in the data section of `tern-distill-prototype/scaled-training.md`.

### Tier 2 — Medium impact / medium effort (~half day each)

**Tune the contrastive loss weight.** Current: `0.15`. Higher weights (0.3, 0.5) push different inputs further apart, sharpening the score distribution. Risk: at high weights, distillation alignment with the teacher degrades. Sweep needed.

**Scale to `d_model=384` (base tier).** More dimensions = more semantic capacity, particularly for concepts without token overlap. The Phase 1 doc names this as the marginal-quality fallback — we passed micro-tier eval but have headroom in the 5MB budget. Trade: ~1.8MB more package size. The Wasm engine reads dimensions from the `.bin` header, so no engine changes required.

### Tier 3 — High impact / high effort (multi-day projects)

These already have dedicated sections above:

- **Section 4 — Hard Negative Mining.** Directly targets the failure mode of "should be very similar but isn't." Mines pairs with moderate teacher cosine similarity (0.5–0.7) and trains the student to correctly order them.
- **Section 2 — Intermediate Layer Alignment.** DistilBERT-style — match teacher's hidden states at every layer, not just the final embedding. Strongest single quality lever for a shallow student.
- **Section 8 — Custom Domain BPE.** A 10k-token vocabulary trained on tech corpus would tokenize words like `webpack`, `oauth`, `javascript` as single tokens (currently subword fragments), helping the model learn them more cleanly.
- **Section 1 — Stronger Teacher.** Upgrade from `all-MiniLM-L6-v2` (22M) to `mxbai-embed-large-v1` (335M). Higher-quality soft targets → higher quality ceiling for the student.

### Important caveat: this is a model issue, not an engine issue

The Wasm engine is **architecture-ready** — it reads dimensions from the `.bin` header and runs whatever model is exported. Better models drop in without engine code changes. Improving semantic quality means re-training, re-exporting, and the engine just runs the new `.bin` file.

The Phase 1 eval results (Task 1 cosine_sim 0.81, Task 3 R@3 0.80/1.00 with ternary embedding) remain the bar — these spot-check observations don't change the engine's correctness, only highlight a quality lever for future training runs.

---

## 14. Projection Head Strategy: Aligning Student and Teacher Spaces

**Status (2026-05-02):** Option A1 (ship the f32 projection) is now implemented. The `.bin` format gained an `output_dim` field in the header and the projection weight + bias are written as f32 after the final layer norm. The engine reads them and applies an `f32_linear` step between `mean_pool` and `l2_normalize`. Output dimension is now 384.

**Context:** Phase 1 distillation used an asymmetric setup — student encoder body at `d_model=256`, with a learned linear projection `256 → 384` mapping the encoder output into the teacher's coordinate frame so the cosine/MSE loss could be computed. This projection head was full-precision float32 and shipped only at training time.

**What the Phase 2 regression test revealed:** When the Wasm engine exports only the encoder body (without the projection), STS-B AUC drops by ~0.10 and Spearman by ~0.11 vs. Phase 1 baselines. Recall@3 retrieval is unaffected. The interpretation: the projection layer is where the model learned the teacher's semantic geometry. The 256-dim encoder output retains structure but isn't aligned to the teacher's frame.

This is a fundamental design decision, not a bug — the engine is bit-correct end-to-end (validated to ~2e-7 vs. Python in `test_embed.js`). It surfaced three architectural options for future training runs.

### Option A — Ship the projection (quickest fix)

Export `model.proj` (256 × 384 float32 ≈ 400 KB) alongside the ternary weights. Add one matmul in `inference.rs` after `mean_pool`. Output dim becomes 384.

- **Pro:** Recovers most of the STS-B gap with no retraining. Single-day implementation.
- **Con:** Adds float32 weights to an otherwise ternary engine. Embedding dimension goes up, increasing storage cost for downstream indexes.

### Option B — Project teacher *down* to student dim (best of both)

Add a frozen or learned linear `teacher_384 → 256` reducer during training. Distill 256-to-256 directly. The encoder learns to produce embeddings already in the (reduced) teacher space — no projection needed at inference.

- **Pro:** Wasm engine stays pure-ternary, output dim matches the smaller storage footprint.
- **Con:** Requires retraining. The reducer compresses teacher information — quality ceiling may be lower than Option A's 384-dim alignment.

### Option C — Dimension-agnostic structural loss (cleanest)

Drop the projection entirely. Replace direct vector matching with **pairwise similarity matching** — the student learns to reproduce the teacher's *similarity geometry* over batches, not its raw vectors. Common in retrieval distillation (RankDistil, MarginMSE).

- **Pro:** Student dim becomes a free hyperparameter — 256, 128, even 64. No coordinate-frame coupling. Compresses well.
- **Con:** More complex loss formulation. Harder to debug. May converge slower than direct vector regression.

### Recommendation

For the **next training run**, pursue Option B paired with Section 13's corpus diversification. The combination — broader paraphrase coverage *plus* native 256-dim teacher alignment — directly addresses both the lexical-overlap weakness and the projection-layer dependency in one retrain. Reserve Option A as a same-week patch if a quality fix is needed before retraining.

### What this teaches about distillation more generally

When student and teacher have different hidden dimensions (the common case for compression), **the projection layer is not just glue — it's a learned semantic adapter**. Discarding it at inference discards a real piece of the model. The cleanest distillation architectures either (a) match dimensions on both sides via teacher-side reduction, or (b) avoid the dependency entirely with structural losses. This is worth designing for upfront in any future @tern variant (Section 11) rather than hitting it again at export time.

**Implementation complexity:** Low for Option A (engine-only change). Medium for Option B (requires modifying the training loop's loss computation). High for Option C (substantial loss redesign + tuning).

---

## 15. TypeScript Types and DX Polish

**Current scope:** JavaScript only, no type definitions.

**The opportunity:** Full TypeScript type definitions, JSDoc annotations, and an ergonomic API surface — particularly for the `classify(text, labels)` pattern where the label type and return type can be made strongly typed. This is table stakes for adoption in TypeScript-first projects.

**Implementation complexity:** Low. Pure documentation and `.d.ts` authoring.

---

## 16. `.bin` Format: Tighter Packing & Compression Options

**Current approach:** Wire format v1 stores ternary values at 2 bits each (4 per byte). Output projection stays fp32 by design (per the bitlinear-asymmetry postmortem). No transport-layer compression assumed at the format level — the `.bin` ships as-is. Total packed size for the ternary build: ~2.8 MB.

**The opportunity:** Several orthogonal byte-savings are available. None are urgent — the current 2.8 MB is small in absolute terms — but capturing them so the option isn't lost. All are lossless and reversible; users get identical model quality.

### 16.1 Tight ternary packing — 5 values per byte (1.585 bits/value)

The 2-bit-per-value encoding is 21% over the information-theoretic minimum for ternary data (`log₂(3) ≈ 1.585` bits). Five ternary values can fit in one byte using base-3 encoding:

```
byte = t₀·1 + t₁·3 + t₂·9 + t₃·27 + t₄·81
       where each tᵢ ∈ {0, 1, 2}  (mapped from {-1, 0, +1})

byte ∈ [0, 242]  (= 2 + 6 + 18 + 54 + 162)
       leaving codes 243..255 as "should never appear" / reserved for error signal
```

Decode: extract digits via `(byte / 3ᵏ) mod 3` for k = 0..4. Production implementation uses a precomputed ~1.2 KB lookup table (`u8 → [i8; 5]`) — one memory load + five reads per byte, competitive with bit-shifts.

**Storage win** for the current architecture (vocab=30522, d_model=256):

| Section | v1 (2-bit) | tight (1.585-bit) | Savings |
|---|---|---|---|
| Embedding packed | ~1.86 MB | ~1.51 MB | −358 KB (−19%) |
| BitLinear weights | ~400 KB | ~325 KB | −75 KB (−19%) |
| **Total `.bin`** | **~2.8 MB** | **~2.3 MB** | **−500 KB (−18%)** |

**Cost (engineering / format complexity):**
- Decoder is more code (LUT + base-3 logic + invalid-code handling)
- New unit tests for encode/decode parity
- Format-version migration (v1 → v2)
- WASM SIMD is awkward with base-3 (which loves powers of 2) — the LUT approach scalarizes well, but vectorized decode is more involved

**Cost (runtime):** negligible — see 16.4 below.

**When to revisit:** if bundle download size becomes a load-bearing UX constraint (mobile-first, very slow connections) AND the ship target ends up being ternary embedding (currently int8 per Phase 4 Stage A in `tern-training-pipeline.md`). Otherwise unjustified given the engineering tax.

### 16.2 Transport-layer compression (NOT a format change)

`Content-Encoding: gzip` (or brotli) at the HTTP layer compresses on-the-wire bytes without any engine code change. Modern CDNs do this automatically. Recovers ~10% of the `.bin` size (limited because most of the file is high-entropy packed ternary, low redundancy left to squeeze).

This is the right "first" answer for download-size optimization. It composes with tight packing if both are applied — gzip after tight packing recovers less (already higher entropy), but stacking gets to ~2.0 MB on the wire from the current 2.8 MB.

Note: belongs in deployment / CDN config, not in the engine repo. Captured here so the option isn't conflated with format work.

### 16.3 Other deferred byte-savings (not yet evaluated)

- **Sparse embedding rows.** If a meaningful fraction of vocab rows are all-zero (rare/unused tokens), a "row presence" bitmap + skipping zero rows could cut more bytes. Hasn't been measured; likely worth ~10–20% on the embedding if zero rows are common, ~0% if not. Worth measuring after the first real production training run.
- **fp16 output projection.** Currently fp32 (~385 KB, 13% of `.bin`). Half-precision would save ~190 KB at near-zero quality cost — easier engineering win than tight ternary packing, but only ~7% on the total `.bin`.
- **True 1.58-bit packing** (vs 5-in-8 = 1.6 bits). Would need a variable-rate encoding (Huffman-style) — too much complexity for the marginal extra savings over 16.1.

### 16.4 Performance notes — why tight packing wouldn't slow users down

Unpack work happens in two distinct places with very different cost profiles:

**Per query (embedding lookup):**
- Per token: load packed row (~52 B tight vs ~64 B v1) → unpack 256 ternary → multiply by per-row scale
- 30-token query: ~2 µs of unpack work either way
- Less than 0.05% of total inference time (~5–20 ms per query)

**Per engine load (BitLinear weight unpack):**
- Pre-expand all ternary weights to in-memory format once at init (NEVER unpack per-matmul — that would defeat the purpose of having packed weights to begin with)
- Total ~1.6 M ternary values for the current architecture → ~1.5 ms at engine init
- ~1% of cold-start time, dominated elsewhere by network + WASM instantiation

End-to-end: unpack cost is microseconds per query and milliseconds at load for any reasonable ternary encoding scheme. The bytes-on-disk question and the user-side runtime question are essentially decoupled.

**Cross-reference:** Section 5 (Higher-Precision Embedding Table) is the *opposite* direction — bigger files for better quality. Section 16 is tighter packing for smaller files at same quality. The two are orthogonal — both could be applied independently. Phase 4 Stage A (see `tern-training-pipeline.md`) settled int8 as the ship embedding format, partially superseding Section 5's framing.

**Implementation complexity:** Medium for 16.1 (decoder + LUT + format-version migration + parity tests). Low for 16.2 (CDN configuration only). Variable for 16.3 (sparse-rows measurement is cheap; fp16 projection is straightforward; true 1.58-bit is high).
