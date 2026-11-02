#!/usr/bin/env bash
# zoro-gravity 앱을 Metal 로 실행합니다. 더블클릭하거나 터미널에서 실행합니다.
set -e
cd "$(dirname "$0")/../.."
export ZOROGRAVITY_DEVICE=metal CARGO_TARGET_DIR="$PWD/target-metal"
cd app
npm install
exec npx tauri dev --release --no-watch --features metal -- --no-default-features
