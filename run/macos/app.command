#!/usr/bin/env bash
# zoro-gravity 앱을 실행합니다. 더블클릭하거나 터미널에서 실행합니다.
set -e
cd "$(dirname "$0")/../../app"
npm install
exec npm run tauri dev
