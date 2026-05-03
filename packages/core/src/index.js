// @tern/core — shared runtime utilities.
// Currently a placeholder. Members are added as downstream packages need them.

export class TernError extends Error {
  constructor(message, code) {
    super(message);
    this.name = "TernError";
    this.code = code;
  }
}