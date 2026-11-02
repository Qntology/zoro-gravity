//! N:N 검증: 민원 종류 × 서류 상태 × 운영 모드. 규칙 층입니다 (모델은 대역을 씁니다).
//!
//! 서류는 정답 값을 그대로 넣습니다. OCR 은 `case_read`, 실제 모델은 `matrix_eval` 예제가 봅니다.
//! 시나리오는 `core/fixtures/scenarios.json` 에 있습니다. 표 출력: `cargo test --test matrix -- --nocapture`

use std::collections::BTreeMap;

use anyhow::Result;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde_json::{json, Value};
use zorogravity_core::case::Category;
use zorogravity_core::decision::{DecisionModel, DecisionTask};
use zorogravity_core::intake::{self, confirm_form, counter_verify, intake_detail, list_intake, officer_action, submit_form, CounterInput, DeskAction, DeskError, IntakeStatus, KioskStatus, SubmitError};
use zorogravity_core::lattice::Lattice;
use zorogravity_core::policy::AutomationPolicy;
use zorogravity_core::state_machine::gate3_substantive::Justification;
use zorogravity_core::state_machine::Engine;

#[path = "support/scenario.rs"]
mod scenario;
use scenario::{category, kiosk_values, modes, scan, scenarios};

fn now() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 11, 2, 1, 0, 0).unwrap()
}

fn today() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 11, 2).unwrap()
}

/// 첫 선택지에 높은 확률을 주는 대역. 규칙 층에서는 모델이 동의한다고 가정하고 정책과 경로만 봅니다.
struct Agrees;

impl DecisionModel for Agrees {
    fn probabilities(&self, task: &DecisionTask) -> Result<Vec<f32>> {
        let n = task.options.len();
        Ok((0..n).map(|i| if i == 0 { 0.98 } else { 0.02 / (n - 1).max(1) as f32 }).collect())
    }
}

fn engine(auto: bool) -> Engine {
    let mut e = Engine::new(b"matrix-salt", AutomationPolicy::default(), Some(Box::new(Agrees)));
    if auto {
        e.enable_automation("A-0001", "합성 시험용 근거 조항", now()).unwrap();
    }
    e
}

/// 민원인 한 사람이 키오스크에서 끝까지 낸 결과. `excuses` 는 소명 재요청에 차례로 답합니다.
fn submit(e: &mut Engine, db: &Lattice, s: &Value, mode: &str, token: u32) -> (String, Option<String>) {
    let excuses: Vec<&str> = s.get("excuses").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
    let mut next = 0;
    let mut excuse = None;
    loop {
        match submit_form(e, db, category(s), &kiosk_values(s, mode, excuse), now(), today(), token) {
            Ok(r) => {
                let outcome = match r.status {
                    KioskStatus::Waiting => "officer",
                    KioskStatus::Finished { .. } => "auto",
                };
                return (outcome.to_string(), Some(r.receipt_no));
            }
            Err(SubmitError::NeedExcuse { message, .. }) => {
                assert!(next < excuses.len(), "{}: 소명을 더 요구합니다: {message}", s["id"]);
                excuse = Some(excuses[next]);
                next += 1;
            }
            Err(SubmitError::Retry(m)) => return (format!("retry: {m}"), None),
            Err(other) => panic!("{} {mode}: {other:?}", s["id"]),
        }
    }
}

fn finished_code(e: &Engine, receipt: &str) -> Option<String> {
    list_intake(e).into_iter().find(|i| i.receipt_no == receipt).and_then(|i| match i.status {
        IntakeStatus::Finalized { final_code, .. } => Some(final_code),
        _ => None,
    })
}

#[test]
fn every_scenario_ends_where_it_should_for_every_paper_state_and_operating_mode() {
    let db = Lattice::embedded().unwrap();
    let mut wrong = Vec::new();
    let mut table = String::from("시나리오 | 서류 | 보조 모드 | 자동화 모드 | 후보\n");
    let mut count = 0;
    for s in scenarios() {
        for mode in modes() {
            let expected = s["expect"]["by_mode"][&mode].as_str().unwrap();
            for auto in [false, true] {
                count += 1;
                let mut e = engine(auto);
                let (got, receipt) = submit(&mut e, &db, &s, &mode, count);
                // 보조 모드에서는 자동 확정이 없습니다.
                let want = if !auto && expected == "auto" { "officer" } else { expected };
                if got != want {
                    wrong.push(format!("{} {mode} auto={auto}: 기대 {want}, 실제 {got}", s["id"].as_str().unwrap()));
                }
                let candidate = match receipt.as_deref() {
                    Some(r) if got == "officer" => intake_detail(&e, r).unwrap().candidate,
                    Some(r) => finished_code(&e, r),
                    None => None,
                };
                let want_verdict = s["expect"]["verdict"].as_str().map(str::to_string);
                if candidate != want_verdict {
                    wrong.push(format!("{} {mode} auto={auto}: 후보 기대 {want_verdict:?}, 실제 {candidate:?}", s["id"].as_str().unwrap()));
                }
                if auto {
                    table.push_str(&format!("{} | {mode} | {} | {got} | {}\n", s["id"].as_str().unwrap(), "officer", candidate.as_deref().unwrap_or("-")));
                }
            }
        }
    }
    println!("{table}\n조합 {count}개");
    assert!(wrong.is_empty(), "틀린 조합 {}개:\n{}", wrong.len(), wrong.join("\n"));
}

/// 다른 민원의 서류를 낸 경우 (N:N). 필요 없는 서류가 섞이면 자동 확정하지 않고 공무원에게 갑니다.
#[test]
fn papers_from_another_category_never_finish_automatically() {
    let db = Lattice::embedded().unwrap();
    let clean: Vec<Value> = scenarios().into_iter().filter(|s| s["id"].as_str().unwrap().ends_with("clean")).collect();
    assert_eq!(clean.len(), 4);
    let mut token = 1000;
    let mut count = 0;
    for s in &clean {
        for other in &clean {
            if s["category"] == other["category"] {
                continue;
            }
            let mut v = kiosk_values(s, "id_only", None);
            let extra: Vec<Value> = other["docs"].as_object().unwrap().iter().filter(|(_, d)| d.get("as").is_none()).map(|(k, d)| scan(other, k, d)).collect();
            v["scanned"].as_array_mut().unwrap().extend(extra);
            let mut e = engine(true);
            token += 1;
            count += 1;
            let r = submit_form(&mut e, &db, category(s), &v, now(), today(), token).expect("접수는 됩니다");
            assert_eq!(r.status, KioskStatus::Waiting, "{} + {}", s["id"], other["id"]);
            let d = intake_detail(&e, &r.receipt_no).unwrap();
            assert!(d.reason.contains("필요 없는 서류"), "{}", d.reason);
        }
    }
    println!("교차 조합 {count}개");
    assert_eq!(count, 12);
}

fn full_counter(detail: &intake::IntakeDetail, s: &Value) -> CounterInput {
    let mut input = CounterInput {
        docs: detail.docs.iter().filter(|d| !d.done).map(|d| d.kind.clone()).collect(),
        identity_checked: true,
        pending_checked: true,
        ..CounterInput::default()
    };
    if let Some(c) = s.get("officer").and_then(|o| o.get("counter")) {
        input.contract_rent = c["contract_rent"].as_i64();
        input.latest_transfer = c["latest_transfer"].as_i64();
        input.printed_account = c["printed_account"].as_str().map(str::to_string);
        input.landlord = c["landlord"].as_array().map(|a| (a[0].as_str().unwrap().to_string(), a[1].as_str().unwrap().to_string()));
    }
    if let Some(amount) = s.get("officer").and_then(|o| o.get("reduce_to")).and_then(Value::as_i64) {
        input.approved_amount = Some(amount);
    }
    input
}

/// 공무원에게 간 건은 어떤 서류 상태에서도 끝낼 수 있어야 합니다 (막다른 길 없음). 함정 행위는 막혀야 합니다.
#[test]
fn every_case_that_reaches_the_officer_can_be_finished_and_traps_are_blocked() {
    let db = Lattice::embedded().unwrap();
    let mut token = 5000;
    let mut count = 0;
    let mut traps: BTreeMap<String, usize> = BTreeMap::new();
    for s in scenarios() {
        for mode in modes() {
            token += 1;
            let mut e = engine(false);
            let (got, receipt) = submit(&mut e, &db, &s, &mode, token);
            assert_eq!(got, "officer", "{} {mode}", s["id"]);
            let receipt = receipt.unwrap();
            let id = s["id"].as_str().unwrap();
            count += 1;

            if category(&s) == Category::Grievance {
                let c = intake::Checklist { identity_checked: true, contact_checked: true, body_read: true, department_checked: true };
                intake::transfer_grievance(&mut e, &receipt, "A-1", "건설관리과", c, now()).unwrap();
                continue;
            }

            let detail = intake_detail(&e, &receipt).unwrap();
            assert!(!detail.reason.trim().is_empty(), "{id} {mode}: 공무원이 볼 사유가 없습니다");
            let trap = s.get("officer").and_then(|o| o.get("trap")).and_then(Value::as_str);
            let ends_supplement = s.get("officer").and_then(|o| o.get("ends")).and_then(Value::as_str) == Some("supplement");
            match trap {
                Some("rubber_stamp") => {
                    let r = officer_action(&mut e, &receipt, "A-1", DeskAction::ApproveHighRisk(Justification::ProveFalsePositive { evidence_scans: 0 }), now());
                    assert!(matches!(r, Err(DeskError::Blocked(_))), "{id} {mode}: 입증 없는 승인 {r:?}");
                    *traps.entry("입증 없는 승인 차단".into()).or_default() += 1;
                }
                Some("unproven_edit") => {
                    let mut input = full_counter(&detail, &s);
                    input.approved_amount = Some(1_000_000);
                    let r = counter_verify(&mut e, &db, &receipt, "A-1", &input, now());
                    assert!(matches!(r, Err(DeskError::Blocked(_))), "{id} {mode}: 무증빙 수정 {r:?}");
                    assert!(e.store.active(&receipt).is_some(), "막히면 세션이 남습니다");
                    *traps.entry("무증빙 금액 수정 차단".into()).or_default() += 1;
                }
                Some("force_pass") => {
                    let r = officer_action(&mut e, &receipt, "A-1", DeskAction::ForcePass, now());
                    assert!(matches!(r, Err(DeskError::Blocked(_))), "{id} {mode}: 강제 진입 {r:?}");
                    *traps.entry("강제 진입 차단".into()).or_default() += 1;
                }
                Some("instant_reject") => {
                    let archive = officer_action(&mut e, &receipt, "A-1", DeskAction::Reject, now()).unwrap();
                    assert_eq!(finished_code(&e, &receipt).as_deref(), Some("SUPPLEMENT_7DAYS"), "{id} {mode}: 즉시 반려 {archive}");
                    *traps.entry("즉시 반려를 보완요구로 전환".into()).or_default() += 1;
                    continue;
                }
                Some("unmask") => {
                    let allowed = intake::request_unmask(&mut e, &receipt, "A-1", now()).unwrap();
                    assert!(!allowed, "{id} {mode}: 낮은 위험도에서 실명 열람");
                    *traps.entry("실명 열람 거부".into()).or_default() += 1;
                }
                _ => {}
            }

            if ends_supplement {
                officer_action(&mut e, &receipt, "A-1", DeskAction::Supplement, now()).unwrap();
                assert_eq!(finished_code(&e, &receipt).as_deref(), Some("SUPPLEMENT_7DAYS"), "{id} {mode}");
                continue;
            }

            let input = full_counter(&detail, &s);
            let archive = match confirm_form(&mut e, &db, &receipt, "A-1", &input, now()) {
                Ok(a) => a,
                // 값이 바뀌어 후보가 달라졌으면 공무원이 다시 보고 확정합니다.
                Err(DeskError::Changed { .. }) => confirm_form(&mut e, &db, &receipt, "A-1", &CounterInput::default(), now()).unwrap_or_else(|err| panic!("{id} {mode}: 재확정 {err}")),
                Err(err) => panic!("{id} {mode}: 확정 실패 {err}"),
            };
            assert!(!archive.is_empty());
            assert!(e.store.active(&receipt).is_none(), "{id} {mode}: 세션이 남았습니다");
        }
    }
    println!("공무원 처리 {count}건, 차단 {traps:?}");
    assert!(traps.len() >= 5, "{traps:?}");
}

/// 민원인 재입력 경로: 서명이 약하면 다시 받고, 소명 재요청에는 내부 정보가 없습니다.
#[test]
fn the_kiosk_asks_again_without_leaking_anything() {
    let db = Lattice::embedded().unwrap();
    let s = scenarios().into_iter().find(|s| s["id"] == "W2-mother-gap").unwrap();
    let mut e = engine(true);
    let mut weak = kiosk_values(&s, "id_only", Some("cash"));
    weak["signature"] = json!({ "stroke_count": 1, "bbox_area_ratio": 0.03, "single_line": true });
    assert!(matches!(submit_form(&mut e, &db, Category::Welfare, &weak, now(), today(), 1), Err(SubmitError::Retry(_))));

    let first = submit_form(&mut e, &db, Category::Welfare, &kiosk_values(&s, "id_only", None), now(), today(), 2);
    let Err(SubmitError::NeedExcuse { message, locked }) = first else { panic!("소명을 물어야 합니다") };
    assert!(message.contains("200000원"), "{message}");
    assert!(locked.is_empty());

    let second = submit_form(&mut e, &db, Category::Welfare, &kiosk_values(&s, "id_only", Some("typo")), now(), today(), 3);
    let Err(SubmitError::NeedExcuse { message, locked }) = second else { panic!("오타 주장은 잠겨야 합니다") };
    assert_eq!(locked, vec!["typo".to_string()]);
    for word in ["모친", "관계망", "위반", "G3_", "박영희"] {
        assert!(!message.contains(word), "{word}: {message}");
    }
    assert_eq!(e.store.active_count(), 0, "소명을 받기 전에는 세션이 생기지 않습니다");
}

/// 본인 확인 전에는 대장 값이 민원인 화면으로 나가지 않습니다.
#[test]
fn without_identity_the_kiosk_never_reveals_registry_values() {
    let db = Lattice::embedded().unwrap();
    let s = scenarios().into_iter().find(|s| s["id"] == "W2-mother-gap").unwrap();
    let mut e = engine(true);
    let r = submit_form(&mut e, &db, Category::Welfare, &kiosk_values(&s, "none", None), now(), today(), 9).expect("소명을 묻지 않고 창구로 넘깁니다");
    assert_eq!(r.status, KioskStatus::Waiting);
}
