#!/usr/bin/env bash
# zoro-gravity 앱을 실행합니다. 더블클릭하거나 터미널에서 실행합니다.
# Apple Silicon 은 Metal 로 돌고, Intel 맥은 Metal 커널이 맞지 않아 CPU 로 돕니다.
set -e
cd "$(dirname "$0")/../.."
export CARGO_TARGET_DIR="$PWD/target-metal"
cd app
npm install
exec npx tauri dev --release --no-watch --features metal -- --no-default-features
