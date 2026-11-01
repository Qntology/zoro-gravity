#!/usr/bin/env bash
# 모델 관리 화면을 엽니다 (진행률 표시 다운로드 · 삭제). GPU SDK 는 필요 없습니다.
cd "$(dirname "$0")/../.."
exec cargo run --release -p zorogravity-models
