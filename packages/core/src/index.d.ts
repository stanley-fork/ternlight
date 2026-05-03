/**
 * Common error class for all @tern packages.
 *
 * @property code - Stable identifier for catching specific error categories.
 */
export class TernError extends Error {
  code?: string;
  constructor(message: string, code?: string);
}

/** A 384-dim L2-normalized text embedding. */
export type Embedding = Float32Array;

/** Result row from a nearest-neighbor search. */
export interface SimilarityResult {
  text: string;
  sim: number;
}