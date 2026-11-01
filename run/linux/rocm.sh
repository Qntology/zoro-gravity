#!/usr/bin/env bash
# AMD ROCm 으로 zorogravity-core 를 빌드해 실행합니다. ROCm(hipcc, libamdhip64)이 있어야 합니다.
#   ./rocm.sh [예제] [예제 인자...]
exec bash "$(dirname "$0")/../../scripts/run-unix.sh" rocm "$@"
