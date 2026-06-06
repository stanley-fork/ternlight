#!/usr/bin/env bash
# Build the Wasm engine and copy it into the ternlight package.
#
# This is the Stage 2 build (wasm-pack as the orchestrator). The ternlight
# package's `pkg/` directory becomes a snapshot of engine/pkg/, ready to be
# published as part of the npm tarball.
#
# Usage:
#   bash scripts/build-engine.sh                  # release build, default features
#   PROFILE=debug bash scripts/build-engine.sh    # debug build
#   FEATURE=emb_int8 bash scripts/build-engine.sh # override embedding format
#
# See docs/tern-bundling.md → Stage 3 for the longer-term plan of dropping
# down to wasm-bindgen-cli directly (more control, no wasm-pack opinions).

set -euo pipefail

PROFILE="${PROFILE:-release}"
FEATURE="${FEATURE:-emb_int4}"   # current ship target — see eval/results/quality.json
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENGINE_DIR="$ROOT/engine"
TARGET_DIR="$ROOT/packages/ternlight/pkg"

echo "Building tern-engine ($PROFILE, --features $FEATURE)..."
cd "$ENGINE_DIR"

if [[ "$PROFILE" == "release" ]]; then
    wasm-pack build --target nodejs --release --features "$FEATURE"
    if command -v wasm-opt >/dev/null 2>&1; then
        echo "Optimizing with wasm-opt -Oz..."
        wasm-opt -Oz pkg/tern_engine_bg.wasm -o pkg/tern_engine_bg.wasm
    else
        echo "WARNING: wasm-opt not found — skipping size optimization"
    fi
else
    wasm-pack build --target nodejs --features "$FEATURE"
fi

echo "Copying engine/pkg/ → packages/ternlight/pkg/ ..."
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR"
# Copy the files we actually want shipped — skip wasm-pack's auto package.json
# (we ship our own at packages/ternlight/package.json), skip the README it
# generates (ours lives one level up), and skip the .gitignore.
cp "$ENGINE_DIR/pkg/tern_engine.js"           "$TARGET_DIR/"
cp "$ENGINE_DIR/pkg/tern_engine_bg.wasm"      "$TARGET_DIR/"
cp "$ENGINE_DIR/pkg/tern_engine.d.ts"         "$TARGET_DIR/"
cp "$ENGINE_DIR/pkg/tern_engine_bg.wasm.d.ts" "$TARGET_DIR/"

WASM_BYTES=$(wc -c <"$TARGET_DIR/tern_engine_bg.wasm")
WASM_MB=$(awk "BEGIN {printf \"%.2f\", $WASM_BYTES / 1024 / 1024}")
echo "Done. packages/ternlight/pkg/tern_engine_bg.wasm = ${WASM_MB} MB"
