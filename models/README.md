# models

> Pointers to model release artifacts. Binaries themselves are NOT committed to this repo — they're distributed via HuggingFace and GitHub Releases, and bundled into the `ternlight` npm package at publish time.

## Variants

Three quantization variants of the same architecture (2-layer Transformer encoder, ~9.5M parameters, 384-dim L2-normalized output) are shipped. All share the same BERT WordPiece tokenizer (`tokenizer.json`, ~695 KB).

| File | Quantization | Bin size | Spearman vs teacher | Role |
|---|---|---:|---:|---|
| **`model-int4.bin`** ⭐ | 4-bit per-row PTQ on embedding; ternary BitLinear | **4.6 MB** | **0.835** | **Primary ship** |
| `model-embedding-int8.bin` | 8-bit per-row on embedding; ternary BitLinear | 8.3 MB | 0.841 | Higher-quality variant |
| `model-ternary.bin` | Ternary embedding; ternary BitLinear | 2.9 MB | 0.710 | Size-extreme variant |

Each `.bin` ships with a `.bin.json` sidecar carrying provenance — training run ID, source checkpoint, code commit, packing timestamp, SHA-256 — for reproducibility and integrity checks.

## Where to get them

| Channel | Use case |
|---|---|
| **[HuggingFace `wenshutang/ternlight`](https://huggingface.co/wenshutang/ternlight)** | Public download via `hf_hub_download` or the HF web UI. Primary distribution channel. |
| **GitHub Releases** | Versioned URLs with download stats. Consumed by the build pipeline that bundles the `.bin` into the `ternlight` npm package. |

## Why models aren't in git

- Binaries don't diff well and compound repo bloat over time as variants and re-trains accumulate
- npm consumers get the `.bin` bundled into the published package — they don't need the raw artifact
- HF + GitHub Releases both give versioned URLs and integrity hashes for free

## Pipeline

```
training/distill/runs/<run-name>/checkpoint_ep<N>.pt    (not in git)
        ↓ training/pack/pack.py
training/pack/out/model-<variant>.bin                   (not in git)
        ↓ scripts/release-model.sh
GitHub Release v<X.Y.Z>                                 (e.g. model-int4.bin asset)
        ↓ hf upload wenshutang/ternlight ...
huggingface.co/wenshutang/ternlight                     (public download)
        ↓ at npm publish time
packages/ternlight/pkg/                                 (bundled into the published package)
```
