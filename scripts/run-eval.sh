#!/usr/bin/env bash
# Run all eval suites against the currently-built engine and regenerate REPORT.md.
#
# Usage:
#   bash scripts/run-eval.sh                          # run all suites
#   SUITE=regression bash scripts/run-eval.sh         # just regression
#   VERSION=0.1.0 bash scripts/run-eval.sh            # tag results with a specific version
set -euo pipefail

SUITE="${SUITE:-all}"
VERSION="${VERSION:-dev}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Running eval suite: $SUITE  (version=$VERSION)"

# TODO: implement once eval/ scripts land.
# Sketch:
#   if [[ "$SUITE" == "all" || "$SUITE" == "regression" ]]; then
#       node "$ROOT/eval/regression/regression_test.js" --output-json "$ROOT/eval/results/v${VERSION}.partial.regression.json"
#   fi
#   if [[ "$SUITE" == "all" || "$SUITE" == "benchmarks" ]]; then
#       node "$ROOT/eval/benchmarks/latency.js"     --output-json "$ROOT/eval/results/v${VERSION}.partial.latency.json"
#       node "$ROOT/eval/benchmarks/memory.js"      --output-json "$ROOT/eval/results/v${VERSION}.partial.memory.json"
#       node "$ROOT/eval/benchmarks/bundle-size.js" --output-json "$ROOT/eval/results/v${VERSION}.partial.size.json"
#   fi
#   ... merge partials into v${VERSION}.json
#   ... regenerate REPORT.md from the merged JSON

echo "(stub — implement once eval scripts land)"