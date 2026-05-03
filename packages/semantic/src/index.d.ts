/**
 * Embed a string into a 384-dim L2-normalized vector.
 *
 * @param text - Input string. Will be tokenized to max 128 tokens internally.
 * @returns Float32Array of length 384, with L2 norm = 1.
 */
export function embed(text: string): Promise<Float32Array>;

/**
 * Cosine similarity between two L2-normalized vectors.
 *
 * For unit vectors, cosine sim equals the dot product. Range [-1, 1];
 * for typical text embeddings, values are in [0, 1].
 *
 * @throws if vector lengths don't match.
 */
export function cosineSim(a: Float32Array, b: Float32Array): number;

export interface SimilarOptions {
  /** Number of top matches to return. Defaults to 3. */
  topK?: number;
}

export interface SimilarResult {
  text: string;
  sim: number;
}

/**
 * Convenience: embed query + each corpus item, return top-K matches sorted
 * descending by similarity.
 *
 * For large corpora, prefer pre-embedding the corpus once with `embed()` and
 * caching the vectors instead of calling this on every query.
 */
export function similar(
  query: string,
  corpus: string[],
  opts?: SimilarOptions,
): Promise<SimilarResult[]>;