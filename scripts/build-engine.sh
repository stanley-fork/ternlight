#!/usr/bin/env bash
# Build the Wasm engine and copy it into the semantic package.
#
# Usage:
#   bash scripts/build-engine.sh              # release build
#   PROFILE=debug bash scripts/build-engine.sh
set -euo pipefail

PROFILE="${PROFILE:-release}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "Building tern-engine ($PROFILE)..."
cd "$ROOT/engine"

if [[ "$PROFILE" == "release" ]]; then
    wasm-pack build --target nodejs --release
    if command -v wasm-opt >/dev/null 2>&1; then
        echo "Optimizing with wasm-opt -Oz..."
        wasm-opt -Oz pkg/tern_engine_bg.wasm -o pkg/tern_engine_bg.wasm
    else
        echo "WARNING: wasm-opt not found — skipping size optimization"
    fi
else
    wasm-pack build --target nodejs
fi

echo "Copying artifacts into packages/semantic..."
cp pkg/tern_engine_bg.wasm "$ROOT/packages/semantic/engine.wasm"
cp pkg/tern_engine.js      "$ROOT/packages/semantic/src/_engine.js"

WASM_BYTES=$(wc -c <"$ROOT/packages/semantic/engine.wasm")
echo "Done. engine.wasm = ${WASM_BYTES} bytes"