#!/usr/bin/env bash
# NVIDIA 는 CUDA 10.2 를 끝으로 macOS 지원을 중단해서 이 백엔드는 맥에서 빌드할 수 없습니다.
echo "[run] macOS 에는 CUDA 가 없습니다. 같은 폴더의 metal.command 를 쓰세요." >&2
exit 1
