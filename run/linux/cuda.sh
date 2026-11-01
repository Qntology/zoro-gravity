#!/usr/bin/env bash
# NVIDIA CUDA 로 zorogravity-core 를 빌드해 실행합니다. CUDA Toolkit(nvcc)과 NVIDIA 드라이버가 있어야 합니다.
#   ./cuda.sh [예제] [예제 인자...]
exec bash "$(dirname "$0")/../../scripts/run-unix.sh" cuda "$@"
