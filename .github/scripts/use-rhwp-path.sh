#!/usr/bin/env bash
# rhwp 저장소는 git 3.6 GB, 체크아웃 9 GB 라서 cargo 의 git 의존성으로 받으면 러너 디스크와 시간을 다 씁니다.
# tools/hwp-builder/Cargo.toml 이 고정한 커밋에서 빌드에 필요한 경로만 받아 path 의존성으로 바꿉니다.
# Cargo.lock 의 rhwp 출처 줄도 지워서 나머지 의존성은 --locked 로 그대로 검사합니다. CI 전용이고 커밋하지 않습니다.
# 저장소를 체크아웃한 폴더의 바로 위에서 실행합니다. rhwp 를 올린 뒤 빌드가 파일을 못 찾으면 아래 목록에 더하세요.
set -euo pipefail
repo="${1:-zoro-gravity}"
toml="$repo/tools/hwp-builder/Cargo.toml"
lock="$repo/Cargo.lock"

url="$(sed -nE 's/^rhwp = \{ git = "([^"]+)".*/\1/p' "$toml")"
rev="$(sed -nE 's/^rhwp = \{ git = "[^"]+", rev = "([0-9a-f]{40})".*/\1/p' "$toml")"
if [ -z "$url" ] || [ -z "$rev" ]; then
  echo "::error title=rhwp::$toml 에서 rhwp 의 git 주소와 고정 커밋을 찾지 못했습니다"
  exit 1
fi

git init -q rhwp
git -C rhwp remote add origin "$url"
git -C rhwp sparse-checkout set --no-cone --stdin <<'EOF'
/Cargo.toml
/Cargo.lock
/build.rs
/llms.txt
/src/
/crates/
/saved/
/tests/fixtures/fonts/
/mydocs/manual/
/mydocs/tech/agent_roadmap/atlas_r1_r200.md
/samples/hml/
/samples/3-09*
/samples/hwp3-sample16-hwp5.hwp
/samples/render-p35-font-native-bitmap.hwpx
EOF
git -C rhwp fetch -q --depth 1 --filter=blob:none origin "$rev"
git -C rhwp checkout -q FETCH_HEAD

sed -i.bak -E 's|^rhwp = \{ git = "[^"]+", rev = "[0-9a-f]{40}"|rhwp = { path = "../../../rhwp"|' "$toml"
sed -i.bak '/^source = "git+https:\/\/github\.com\/edwardkim\/rhwp?rev=/d' "$lock"
rm -f "$toml.bak" "$lock.bak"

grep -q '^rhwp = { path = "../../../rhwp"' "$toml" || { echo "::error title=rhwp::$toml 을 path 로 바꾸지 못했습니다"; exit 1; }
if grep -q 'edwardkim/rhwp' "$lock"; then
  echo "::error title=rhwp::$lock 에 rhwp 출처가 남았습니다"
  exit 1
fi
echo "::notice title=rhwp::$(git -C rhwp rev-parse --short HEAD) 를 필요한 경로만 받았습니다 ($(du -sh rhwp | cut -f1))"
