#!/usr/bin/env bash
# Vulkan 으로 zorogravity-core 를 빌드해 실행합니다. GPU 드라이버와 libvulkan 이 있어야 합니다.
#   ./vulkan.sh [예제] [예제 인자...]
exec bash "$(dirname "$0")/../../scripts/run-unix.sh" vulkan "$@"
