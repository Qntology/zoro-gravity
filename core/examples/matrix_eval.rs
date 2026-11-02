//! N:N 시나리오를 층별로 돌립니다 (자동화 모드). 규칙 층 시험(`tests/matrix.rs`)과 같은 입력을 씁니다.
//!
//!   matrix_eval [--rules] [--readings <읽기 폴더>] [--only W1,W3] [출력.json]
//!
//! - 인자 없음: 서류 정답 값 + 실제 모델
//! - `--rules`: 모델 대역. GPU 가 필요 없습니다.
//! - `--readings`: 서류 값을 `case_read` 가 읽은 글자에서 뽑습니다. OCR · 추출 · 대장 대조까지 포함합니다.
//!
//! 자동 확정을 기대한 건이 멈추면 `놓침`, 기대하지 않은 건이 자동 확정되면 `사고` 입니다. 사고는 0 이어야 합니다.
//! 민원인 재입력(소명)은 시나리오에 적힌 차례대로 답합니다. 읽은 값이 틀리면 민원인이 화면에서 고친다고 봅니다.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde_json::{json, Value};
use zorogravity_core::ai::accel;
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::zoro::open_engine_models;
use zorogravity_core::decision::{DecisionModel, DecisionTask};
use zorogravity_core::extract::{extract, PageText};
use zorogravity_core::intake::{intake_detail, list_intake, submit_form, IntakeStatus, KioskStatus, SubmitError};
use zorogravity_core::lattice::Lattice;
use zorogravity_core::policy::AutomationPolicy;
use zorogravity_core::state_machine::Engine;

#[path = "../tests/support/scenario.rs"]
mod scenario;
use scenario::{category, kiosk_values, modes, paper_names, scenarios};

fn now() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 11, 2, 1, 0, 0).unwrap()
}

fn today() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 11, 2).unwrap()
}

/// 첫 선택지에 높은 확률을 주는 대역
struct Agrees;

impl DecisionModel for Agrees {
    fn probabilities(&self, task: &DecisionTask) -> Result<Vec<f32>> {
        let n = task.options.len();
        Ok((0..n).map(|i| if i == 0 { 0.98 } else { 0.02 / (n - 1).max(1) as f32 }).collect())
    }
}

fn plain(v: &Value) -> String {
    let s = v.as_str().map(str::to_string).unwrap_or_else(|| v.to_string());
    s.chars().filter(|c| !matches!(c, ' ' | ',')).collect()
}

/// 스캐너에 넣은 서류를 읽은 값으로 키오스크 입력을 바꿉니다.
fn from_readings(values: &mut Value, s: &Value, mode: &str, dir: &Path, db: &Lattice) -> Result<usize> {
    let id = s["id"].as_str().unwrap();
    let mut pages = Vec::new();
    for name in paper_names(s, mode) {
        let path = dir.join(id).join(format!("{name}.json"));
        let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
        pages.push(PageText::from_json(&format!("{id}-{name}"), &serde_json::from_str(&text)?));
    }
    let got = extract(&pages, db);
    let mut edited = Vec::new();
    if let Some(prefill) = got.prefill.as_object() {
        for (key, read) in prefill {
            if values.get(key).is_some_and(|truth| plain(truth) != plain(read)) {
                edited.push(json!(key));
            }
        }
    }
    let fixes = edited.len();
    values["scanned"] = Value::Array(got.docs);
    values["edited"] = Value::Array(edited);
    Ok(fixes)
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let rules = args.iter().position(|a| a == "--rules").map(|i| args.remove(i)).is_some();
    let readings: Option<PathBuf> = args.iter().position(|a| a == "--readings").map(|i| {
        args.remove(i);
        PathBuf::from(args.remove(i))
    });
    let only: Vec<String> = args
        .iter()
        .position(|a| a == "--only")
        .map(|i| {
            args.remove(i);
            args.remove(i).split(',').map(str::to_string).collect()
        })
        .unwrap_or_default();
    let out = args.first().cloned().unwrap_or_else(|| "out/matrix_model.json".to_string());

    let db = Lattice::embedded()?;
    let mut device = None;
    let mut engine = if rules {
        Engine::new(b"matrix-eval-salt", AutomationPolicy::default(), Some(Box::new(Agrees)))
    } else {
        let d = accel::select_device()?;
        println!("장치 {}", accel::backend_name(&d));
        let residency = Residency::shared(d.clone(), Budget::auto(&d));
        let (model, writer) = open_engine_models(&model_dir("zoro"), residency)?;
        device = Some(d);
        Engine::new(b"matrix-eval-salt", AutomationPolicy::default(), Some(Box::new(model))).with_writer(Box::new(writer))
    };
    engine.enable_automation("A-0001", "합성 시험용 근거 조항", now())?;

    let (mut rows, mut token) = (Vec::new(), 0u32);
    let (mut auto_expected, mut auto_done, mut accidents, mut wrong, mut fixes) = (0, 0, 0, 0, 0);
    let (mut dept_all, mut dept_ok) = (0, 0);
    let started = Instant::now();
    for s in scenarios() {
        let id = s["id"].as_str().unwrap().to_string();
        if !only.is_empty() && !only.iter().any(|o| id.starts_with(o.as_str())) {
            continue;
        }
        for mode in modes() {
            let expected = s["expect"]["by_mode"][&mode].as_str().unwrap();
            let excuses: Vec<&str> = s.get("excuses").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
            let (mut next, mut excuse) = (0, None);
            token += 1;
            let t = Instant::now();
            let mut fixed = 0;
            let result = loop {
                let mut values = kiosk_values(&s, &mode, excuse);
                if let Some(dir) = &readings {
                    fixed = from_readings(&mut values, &s, &mode, dir, &db)?;
                }
                match submit_form(&mut engine, &db, category(&s), &values, now(), today(), token) {
                    Err(SubmitError::NeedExcuse { .. }) if next < excuses.len() => {
                        excuse = Some(excuses[next]);
                        next += 1;
                    }
                    other => break other,
                }
            };
            fixes += fixed;
            let ms = t.elapsed().as_millis();
            let mut department = None;
            let (got, verdict, confidence, reason) = match result {
                Ok(r) => match r.status {
                    KioskStatus::Waiting => {
                        let d = intake_detail(&engine, &r.receipt_no)?;
                        department = d.summary.as_ref().and_then(|s| s.department.clone());
                        ("officer", d.candidate, d.confidence, d.reason)
                    }
                    KioskStatus::Finished { .. } => {
                        let code = list_intake(&engine).into_iter().find(|i| i.receipt_no == r.receipt_no).and_then(|i| match i.status {
                            IntakeStatus::Finalized { final_code, .. } => Some(final_code),
                            _ => None,
                        });
                        ("auto", code, None, String::new())
                    }
                },
                Err(e) => ("error", None, None, format!("{e:?}")),
            };
            if expected == "auto" {
                auto_expected += 1;
                auto_done += (got == "auto") as i32;
            } else if got == "auto" {
                accidents += 1;
            }
            let want_verdict = s["expect"]["verdict"].as_str().map(str::to_string);
            let mismatch = verdict != want_verdict;
            wrong += mismatch as i32;
            if let Some(want) = s["expect"]["department"].as_str() {
                if !rules {
                    dept_all += 1;
                    dept_ok += (department.as_deref() == Some(want)) as i32;
                }
            }
            let tag = match (expected, got) {
                ("auto", "officer") => "놓침",
                (e, "auto") if e != "auto" => "사고",
                (_, "error") => "오류",
                _ if mismatch => "후보",
                _ => "",
            };
            println!(
                "{id:<22} {mode:<8} 기대 {expected:<8} 실제 {got:<8} 확신도 {:<6} {ms:>6} ms {tag} {}",
                confidence.map_or("-".to_string(), |c| format!("{c:.3}")),
                if tag.is_empty() { "" } else { &reason }
            );
            rows.push(json!({ "id": id, "mode": mode, "expected": expected, "got": got, "verdict": verdict, "confidence": confidence, "ms": ms, "reason": reason, "department": department }));
        }
    }
    println!(
        "\n자동 확정 기대 {auto_expected}건 중 {auto_done}건, 사고 {accidents}건, 후보 불일치 {wrong}건, 민원인 수정 {fixes}칸, 부서 {dept_ok}/{dept_all}, 조합 {}개, {:.1}초",
        rows.len(),
        started.elapsed().as_secs_f64()
    );
    if let Some(dir) = Path::new(&out).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(
        &out,
        serde_json::to_string_pretty(&json!({ "auto_expected": auto_expected, "auto_done": auto_done, "accidents": accidents, "wrong": wrong, "fixes": fixes, "department": [dept_ok, dept_all], "rows": rows }))?,
    )?;
    if let Some(d) = device {
        accel::finish_process(&d, 0);
    }
    Ok(())
}
