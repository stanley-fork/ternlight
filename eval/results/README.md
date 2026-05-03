# eval/results

Per-release JSON outputs from `scripts/run-eval.sh`. Committed at release tag time so cross-version diffs surface regressions automatically.

## Filename convention

```
v0.1.0.json
v0.1.1.json
v0.2.0.json
```

One file per published release (npm + GitHub tag). The corresponding `REPORT.md` (in the parent `eval/` directory) reads from the latest version.

## File schema

```json
{
  "version": "0.1.0",
  "released": "2026-05-15",
  "commit": "abc123...",
  "engine_build": {
    "wasm_bytes": 612345,
    "wasm_gzipped": 285432,
    "model_bin_bytes": 2763952,
    "total_npm_install_bytes": 3500000
  },
  "quality": {
    "task1_teacher_alignment_mean_cos": 0.8137,
    "task2_stsb_auc": 0.8525,
    "task2_stsb_spearman": 0.7066,
    "task3_general_r3": 0.75,
    "task3_tech_r3": 1.00
  },
  "quantization_gap": {
    "task1_vs_float32": -0.012,
    "task2_auc_vs_float32": -0.020
  },
  "perf": {
    "node22_cold_ms": 612,
    "node22_warm_ms": 568,
    "node22_throughput_per_sec": 1.78,
    "node22_memory_peak_mb": 14.2
  },
  "compatibility": {
    "node18": "PASS",
    "node20": "PASS",
    "node22": "PASS",
    "browser_chromium": "PASS"
  },
  "comparison": {
    "transformers_js_minilm": {
      "stsb_auc": 0.85,
      "bundle_mb": 80,
      "warm_ms": 90
    }
  }
}
```

The full schema is documented in [../../docs/eval/methodology.md](../../docs/eval/methodology.md).

## Status

Pre-alpha. First release artifact will land as `v0.1.0.json` once the engine + packages publishing pipeline is wired up.