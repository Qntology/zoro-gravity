#!/usr/bin/env bash
# AMD 는 macOS 용 ROCm 을 내지 않아서 이 백엔드는 맥에서 빌드할 수 없습니다.
echo "[run] macOS 에는 ROCm 이 없습니다. 같은 폴더의 metal.command 를 쓰세요." >&2
exit 1
