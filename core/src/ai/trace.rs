//! 모델 사용 기록. `ZOROGRAVITY_TRACE` 에 파일 경로를 주면 모델이 언제 무엇을 했는지 한 줄씩 남깁니다 (JSON Lines).
//! 시연 영상과 시간 분석에 씁니다. 설정하지 않으면 아무것도 쓰지 않습니다.
//!
//! 기본 기록에는 민원 내용이 없습니다. 판단의 사실은 식별 정보를 뺀 값이고 (`decision.rs`), 문서와 글쓰기는 좌표 · 개수 · 시간만 남깁니다.
//! `ZOROGRAVITY_TRACE_TEXT=1` 이면 읽은 글자와 모델이 쓴 글도 남깁니다. 합성 서류 시연 전용이고, 실제 민원에서는 켜면 안 됩니다.

use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

pub const TRACE_ENV: &str = "ZOROGRAVITY_TRACE";
pub const TRACE_TEXT_ENV: &str = "ZOROGRAVITY_TRACE_TEXT";

fn sink() -> Option<&'static Mutex<File>> {
    static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();
    SINK.get_or_init(|| File::create(std::env::var_os(TRACE_ENV)?).ok().map(Mutex::new)).as_ref()
}

/// 기록을 남기는 중인가. 기록할 값을 만드는 비용을 아낄 때 씁니다.
pub fn enabled() -> bool {
    sink().is_some()
}

/// 읽은 글자와 쓴 글까지 남기는가
pub fn with_text() -> bool {
    enabled() && std::env::var(TRACE_TEXT_ENV).is_ok_and(|v| v == "1")
}

fn line(at_ms: u128, model: &str, step: &str, ms: f64, detail: &Value) -> String {
    json!({ "t": at_ms as u64, "model": model, "step": step, "ms": (ms * 10.0).round() / 10.0, "detail": detail }).to_string()
}

/// 끝난 일 하나를 남깁니다. `t` 는 끝난 시각이고, `ms` 는 걸린 시간입니다.
pub fn record(model: &str, step: &str, ms: f64, detail: Value) {
    let Some(sink) = sink() else { return };
    let at = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let mut file = sink.lock().unwrap_or_else(|p| p.into_inner());
    // 기록 실패가 심사를 멈추면 안 됩니다.
    let _ = writeln!(file, "{}", line(at, model, step, ms, &detail));
    let _ = file.flush();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_is_one_json_object_with_the_end_time_and_the_duration() {
        let l = line(1_000, "gocr", "비전", 12.34, &json!({ "inputs": 3 }));
        let v: Value = serde_json::from_str(&l).unwrap();
        assert_eq!(v["t"], 1_000);
        assert_eq!(v["model"], "gocr");
        assert_eq!(v["ms"], 12.3);
        assert_eq!(v["detail"]["inputs"], 3);
        assert!(!l.contains('\n'));
    }

    #[test]
    fn nothing_is_recorded_unless_it_is_asked_for() {
        // 시험 실행에는 환경 변수가 없습니다.
        assert!(!enabled());
        assert!(!with_text());
        record("zoro", "판단", 1.0, json!({}));
    }
}
