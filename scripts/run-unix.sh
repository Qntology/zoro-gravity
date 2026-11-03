#!/usr/bin/env bash
# 백엔드 하나로 zorogravity-core 예제를 빌드해서 실행합니다. run/linux 와 run/macos 의 스크립트가 부릅니다.
#
#   bash scripts/run-unix.sh <cuda|rocm|metal|vulkan> [예제] [예제 인자...]
#
# 기본 예제는 판단 경로 전체를 도는 decide_probe 이고 모델 폴더(앱 데이터)에 zoro 가 있어야 합니다.
# 백엔드마다 빌드 폴더를 따로 써서 캐시가 섞이지 않습니다 (vulkan 은 기본 기능이라 target 을 함께 씁니다).
set -euo pipefail
cd "$(dirname "$0")/.."

BACKEND="$1"
shift
EXAMPLE="${1:-decide_probe}"
[ $# -gt 0 ] && shift

fail() {
    echo "[run] $*" >&2
    exit 1
}

find_rocm() {
    local d
    for d in "${ROCM_PATH:-}" "${HIP_PATH:-}" /opt/rocm /opt/rocm/core /opt/rocm/core-* /opt/rocm-*; do
        if [ -n "$d" ] && [ -x "$d/bin/hipcc" ]; then
            echo "$d"
            return 0
        fi
    done
    return 1
}

command -v cargo > /dev/null || fail "cargo 가 없습니다. https://rustup.rs 에서 Rust 를 설치하세요."

case "$BACKEND" in
    vulkan)
        TARGET=target
        # 맥의 포크는 Homebrew 또는 LunarG SDK 의 로더를 찾습니다.
        if [ "$(uname -s)" = Darwin ] && [ -z "${VULKAN_SDK:-}" ] \
            && [ ! -e /opt/homebrew/lib/libvulkan.dylib ] && [ ! -e /usr/local/lib/libvulkan.dylib ]; then
            fail "Vulkan 로더가 없습니다. brew install vulkan-loader molten-vk 로 설치하세요."
        fi
        ;;
    metal)
        TARGET=target-metal
        [ "$(uname -s)" = Darwin ] || fail "Metal 은 macOS 에서만 쓸 수 있습니다."
        ;;
    rocm)
        TARGET=target-rocm
        ROCM_PATH="$(find_rocm)" || fail "hipcc 가 없습니다. ROCm 을 설치하거나 ROCM_PATH 를 지정하세요."
        export ROCM_PATH
        export PATH="$ROCM_PATH/bin:$PATH"
        export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
        ;;
    cuda)
        TARGET=target-cuda
        # 커널을 빌드 때 nvcc 로 컴파일합니다. BLAS 라이브러리는 실행 때 찾습니다.
        CUDA_ROOT="${CUDA_PATH:-${CUDA_HOME:-/usr/local/cuda}}"
        export PATH="$CUDA_ROOT/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
        command -v nvcc > /dev/null || fail "nvcc 가 없습니다. CUDA Toolkit 을 설치하거나 CUDA_PATH 를 지정하세요."
        ;;
    *)
        fail "모르는 백엔드입니다: $BACKEND (cuda, rocm, metal, vulkan)"
        ;;
esac

export CARGO_TARGET_DIR="$TARGET" ZOROGRAVITY_DEVICE="$BACKEND"
echo "[run] 백엔드 $BACKEND · 예제 $EXAMPLE · 빌드 폴더 $TARGET"
cargo build -p zorogravity-core --release --no-default-features --features "$BACKEND" --example "$EXAMPLE"
exec "$TARGET/release/examples/$EXAMPLE" "$@"
