#!/usr/bin/env bash
# Metal 로 zorogravity-core 를 빌드해 실행합니다. 더블클릭하거나 터미널에서 실행합니다.
# Apple Silicon 맥에서만 씁니다. 따로 설치할 것은 없습니다.
#   ./metal.command [예제] [예제 인자...]
exec bash "$(dirname "$0")/../../scripts/run-unix.sh" metal "$@"
