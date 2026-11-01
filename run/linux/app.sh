#!/usr/bin/env bash
# zoro-gravity 앱을 실행합니다. 없는 모델은 켜질 때 내려받고 설정 화면에 진행률을 표시합니다.
set -e
cd "$(dirname "$0")/../../app"
npm install
exec npm run tauri dev
