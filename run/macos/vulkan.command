#!/usr/bin/env bash
# Vulkan(MoltenVK)으로 zorogravity-core 를 빌드해 실행합니다. 더블클릭하거나 터미널에서 실행합니다.
# 먼저 설치하세요: brew install vulkan-loader molten-vk
#   ./vulkan.command [예제] [예제 인자...]
exec bash "$(dirname "$0")/../../scripts/run-unix.sh" vulkan "$@"
