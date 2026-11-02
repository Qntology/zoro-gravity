# 실패 원인과 실행 증거를 어노테이션으로 남깁니다. 작업 로그는 인증 없이 받을 수 없지만 어노테이션은 API 로 읽을 수 있습니다.

# annot LOGFILE : 컴파일 · 빌드 스크립트 · 시험 오류의 앞부분과 로그 끝부분을 어노테이션 하나로
annot() {
  local f="$1" msg
  msg="$( { grep -E '\[DEVICE-OP\]' "$f" | cut -c1-120 | head -20; grep -E '^(error|warning: build failed)|panicked at|failed to run custom build|--- stderr|--- stdout|nvcc|fatal|^failures:|FAILED|exit code|STATUS_|not found' -A3 "$f" | cut -c1-300 | head -30; echo '--- last lines ---'; tail -25 "$f" | cut -c1-300; } | tr '\r' ' ' | sed 's/%/%25/g' | awk '{printf "%s%%0A", $0}' | cut -c1-3800 )"
  echo "::error title=log::$msg"
}

# notice_lines TITLE LOGFILE REGEX : 맞는 줄을 알림 하나로 (실제로 무엇이 돌았는지의 증거)
notice_lines() {
  local msg
  msg="$(grep -E "$3" "$2" | cut -c1-300 | head -40 | tr '\r' ' ' | sed 's/%/%25/g' | awk '{printf "%s%%0A", $0}' | cut -c1-3800)"
  echo "::notice title=$1::${msg:-<맞는 줄 없음>}"
}
