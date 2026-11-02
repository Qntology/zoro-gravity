//! 양식 입력 건의 흐름: 공무원 확인 대기 → 창구 확인 → 공무원 확정.
//! 스캔 건의 규칙은 `governance.rs` 와 단위 테스트가 지킵니다. 민원 종류 × 서류 상태 전체는 `matrix.rs` 가 봅니다.

use anyhow::Result;
use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use zorogravity_core::case::{BankFacts, CaseFile, CaseSource, Category, LeaseFacts, Person, ScannedDoc, Signature};
use zorogravity_core::intake::{confirm_form, counter_verify, intake_detail, submit_form, CounterInput, DeskError, SubmitError};
use zorogravity_core::lattice::Lattice;
use zorogravity_core::policy::AutomationPolicy;
use zorogravity_core::state_machine::{CaseOutcome, Engine, OfficerDecision};
use zorogravity_core::verdict::Verdict;

const WELFARE_KINDS: [&str; 4] = ["신청서", "임대차계약서", "이체내역서", "금융정보동의서"];

fn now() -> DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 11, 2, 1, 0, 0).unwrap()
}

fn engine() -> Engine {
    Engine::new(b"form-route-test", AutomationPolicy::default(), None)
}

fn db() -> Lattice {
    Lattice::embedded().unwrap()
}

fn kinds(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| s.to_string()).collect()
}

/// 창구에서 필요한 확인을 모두 마친 입력
fn counter(docs: &[&str]) -> CounterInput {
    CounterInput { docs: kinds(docs), identity_checked: true, pending_checked: true, ..CounterInput::default() }
}

/// 양식으로 접수한 청년월세 신청. 금액과 계좌가 모두 맞고 서명이 유효합니다.
fn welfare_form(id: &str) -> CaseFile {
    CaseFile {
        case_id: id.into(),
        category: Category::Welfare,
        source: CaseSource::KioskForm,
        docs: Vec::new(),
        applicant: Person { name: "김철수".into(), birth_prefix: "990101".into(), identifier: String::new() },
        landlord: Some(Person { name: "박영희".into(), birth_prefix: "640312".into(), identifier: String::new() }),
        lease: Some(LeaseFacts { applied_rent: 500_000, contract_rent: 500_000, latest_transfer: 500_000 }),
        bank: Some(BankFacts { handwritten_account: "110-123-991023".into(), printed_account: "110-123-991023".into() }),
        signature: Some(Signature { stroke_count: 8, bbox_area_ratio: 0.42, single_line: false }),
        poa_written_on: None,
        today: NaiveDate::from_ymd_opt(2026, 11, 2).unwrap(),
        excuse: None,
        relations: Vec::new(),
        supplement_count: 0,
        grievance_text: None,
        contact: None,
        intake: Default::default(),
    }
}

#[test]
fn a_consistent_form_waits_for_the_officer_and_is_not_finalized_by_the_system() -> Result<()> {
    let mut e = engine();
    let out = e.run(welfare_form("F-1"), now())?;
    assert!(matches!(out, CaseOutcome::OfficerReview { .. }), "{out:?}");
    assert!(e.store.active("F-1").is_some());
    assert!(e.store.archive().is_empty(), "양식 입력만으로는 확정되지 않습니다");
    Ok(())
}

#[test]
fn a_rent_difference_between_declared_values_goes_to_the_counter_and_is_not_corrected() -> Result<()> {
    let mut e = engine();
    let mut case = welfare_form("F-2");
    case.lease = Some(LeaseFacts { applied_rent: 600_000, contract_rent: 500_000, latest_transfer: 500_000 });
    let out = e.run(case, now())?;
    let CaseOutcome::OfficerReview { reason, .. } = out else { panic!("창구 확인으로 가야 합니다: {out:?}") };
    assert!(reason.contains("신청 금액과 계약 월세가 다릅니다"), "{reason}");
    assert!(!reason.contains("정정"), "신고값은 정정하지 않습니다: {reason}");
    Ok(())
}

#[test]
fn an_account_difference_goes_to_the_counter_and_is_not_locked() -> Result<()> {
    let mut e = engine();
    let mut case = welfare_form("F-3");
    case.bank = Some(BankFacts { handwritten_account: "110-123-991023".into(), printed_account: "110-999-000111".into() });
    let out = e.run(case, now())?;
    let CaseOutcome::OfficerReview { reason, .. } = out else { panic!("창구 확인으로 가야 합니다: {out:?}") };
    assert!(reason.contains("계좌"), "{reason}");
    Ok(())
}

#[test]
fn an_invalid_signature_asks_the_kiosk_to_sign_again() -> Result<()> {
    let mut e = engine();
    let mut case = welfare_form("F-4");
    case.signature = Some(Signature { stroke_count: 1, bbox_area_ratio: 0.038, single_line: true });
    let out = e.run(case, now())?;
    assert!(matches!(out, CaseOutcome::KioskRetry { .. }), "{out:?}");
    assert_eq!(e.store.active_count(), 0, "서명 재요청은 세션을 만들지 않습니다");
    Ok(())
}

#[test]
fn confirming_before_the_counter_check_is_blocked_and_the_session_stays() -> Result<()> {
    let mut e = engine();
    e.run(welfare_form("F-5"), now())?;
    let out = e.confirm("F-5", "A-1", OfficerDecision::Confirm, now())?;
    assert!(matches!(out, CaseOutcome::Blocked { verdict: Verdict::G1InterceptForceHold, .. }), "{out:?}");
    assert!(e.store.active("F-5").is_some());
    Ok(())
}

#[test]
fn a_partial_counter_check_still_blocks_and_names_the_rest() -> Result<()> {
    let mut e = engine();
    e.run(welfare_form("F-6"), now())?;
    let out = confirm_form(&mut e, &db(), "F-6", "A-1", &counter(&WELFARE_KINDS[..3]), now());
    let Err(DeskError::Blocked(reason)) = out else { panic!("일부만 확인하면 막혀야 합니다: {out:?}") };
    assert!(reason.contains("금융정보동의서"), "{reason}");
    assert!(e.store.active("F-6").is_some(), "막히면 세션이 남습니다");
    Ok(())
}

#[test]
fn after_the_counter_check_the_officer_can_confirm_and_it_is_archived() -> Result<()> {
    let mut e = engine();
    e.run(welfare_form("F-7"), now())?;
    confirm_form(&mut e, &db(), "F-7", "A-1", &counter(&WELFARE_KINDS), now()).map_err(|e| anyhow::anyhow!("{e}"))?;
    assert_eq!(e.store.archive().len(), 1);
    assert!(e.store.active("F-7").is_none(), "확정되면 세션을 지웁니다");
    Ok(())
}

#[test]
fn the_counter_check_accepts_only_documents_this_case_needs() -> Result<()> {
    let mut e = engine();
    e.run(welfare_form("F-8"), now())?;
    let out = counter_verify(&mut e, &db(), "F-8", "A-1", &counter(&["건강보험 고지서"]), now());
    assert!(matches!(out, Err(DeskError::Blocked(_))), "{out:?}");
    assert!(e.store.active("F-8").is_some());
    Ok(())
}

#[test]
fn the_counter_check_is_refused_for_scanned_cases() -> Result<()> {
    let mut e = engine();
    let mut case = welfare_form("S-1");
    case.source = CaseSource::Scanned;
    case.docs = WELFARE_KINDS
        .iter()
        .enumerate()
        .map(|(i, k)| ScannedDoc { kind: k.to_string(), content_hash: format!("h{i}"), pages: 1 })
        .collect();
    e.run(case, now())?;
    let out = counter_verify(&mut e, &db(), "S-1", "A-1", &counter(&["신청서"]), now());
    assert!(matches!(out, Err(DeskError::Blocked(_)) | Err(DeskError::NotFound(_))), "{out:?}");
    Ok(())
}

// ---- 키오스크 입력(JSON) → 접수 → 공무원 확정 ----

use serde_json::json;
use zorogravity_core::case::Category as Cat;

fn today() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 11, 2).unwrap()
}

fn sign() -> serde_json::Value {
    json!({ "stroke_count": 8, "bbox_area_ratio": 0.42, "single_line": false })
}

fn welfare_json() -> serde_json::Value {
    json!({
        "applicant_name": "이서연", "applicant_birth": "000315",
        "applied_rent": "400000", "handwritten_account": "110-123-100315",
        "excuse": "none", "consent": true, "signature": sign()
    })
}

#[test]
fn a_welfare_form_gets_a_w_receipt_and_waits_for_the_officer() -> Result<()> {
    let mut e = engine();
    let r = submit_form(&mut e, &db(), Cat::Welfare, &welfare_json(), now(), today(), 1).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    assert!(r.receipt_no.starts_with("W20261102-"), "{}", r.receipt_no);
    let d = intake_detail(&e, &r.receipt_no)?;
    assert!(d.lines.iter().any(|(k, v)| k == "신청 월세" && v == "400000원"));
    assert_eq!(d.candidate.as_deref(), Some("G3_ELIGIBLE_NORMAL"));
    assert_eq!(d.docs.len(), 4);
    assert!(d.docs.iter().all(|x| x.done), "{:?}", d.docs);
    Ok(())
}

#[test]
fn a_weak_signature_returns_a_retry_and_no_session() {
    let mut e = engine();
    let mut v = welfare_json();
    v["signature"] = json!({ "stroke_count": 1, "bbox_area_ratio": 0.03, "single_line": true });
    assert!(matches!(submit_form(&mut e, &db(), Cat::Welfare, &v, now(), today(), 2), Err(SubmitError::Retry(_))));
    assert_eq!(e.store.active_count(), 0);
}

#[test]
fn a_bad_form_value_is_reported_with_its_label() {
    let mut e = engine();
    let mut v = welfare_json();
    v["applied_rent"] = json!("abc");
    match submit_form(&mut e, &db(), Cat::Welfare, &v, now(), today(), 3) {
        Err(SubmitError::Form(err)) => assert!(err.to_string().contains("신청 월세"), "{err}"),
        other => panic!("입력 오류여야 합니다: {other:?}"),
    }
}

#[test]
fn the_officer_needs_identity_before_confirming_a_case_without_an_id_scan() -> Result<()> {
    let mut e = engine();
    let r = submit_form(&mut e, &db(), Cat::Welfare, &welfare_json(), now(), today(), 4).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let id = r.receipt_no.as_str();
    match confirm_form(&mut e, &db(), id, "A-1", &CounterInput::default(), now()) {
        Err(DeskError::ChecklistIncomplete(m)) => assert!(m.iter().any(|x| x.contains("본인 확인")), "{m:?}"),
        other => panic!("체크리스트 오류여야 합니다: {other:?}"),
    }
    assert_eq!(e.store.active_count(), 1);
    let archive = confirm_form(&mut e, &db(), id, "A-1", &counter(&[]), now()).map_err(|e| anyhow::anyhow!("{e}"))?;
    assert!(!archive.is_empty());
    assert!(e.store.active(id).is_none());
    assert_eq!(e.store.archive().len(), 1);
    Ok(())
}

#[test]
fn a_higher_settlement_needs_evidence_for_the_difference() -> Result<()> {
    let mut e = engine();
    let v = json!({ "applicant_name": "조아라", "applicant_birth": "830303", "org_name": "소상공인상생회", "program": "골목상권 활성화 사업", "claimed_total": "1000000" });
    let r = submit_form(&mut e, &db(), Cat::Subsidy, &v, now(), today(), 5).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let id = r.receipt_no.as_str();
    let d = intake_detail(&e, id)?;
    let verified = d.verified_sum.unwrap();
    assert!(verified < 1_000_000, "{verified}");
    let mut input = counter(&[]);
    input.approved_amount = Some(1_000_000);
    let short = counter_verify(&mut e, &db(), id, "A-1", &input, now());
    assert!(matches!(short, Err(DeskError::Blocked(ref m)) if m.contains("증빙")), "{short:?}");
    input.evidence_sum = Some(1_000_000 - verified);
    assert!(confirm_form(&mut e, &db(), id, "A-1", &input, now()).is_ok());
    assert_eq!(e.store.archive().len(), 1);
    Ok(())
}

#[test]
fn an_expired_power_of_attorney_reaches_the_officer_as_a_rejection_candidate() -> Result<()> {
    let mut e = engine();
    let v = json!({
        "principal_name": "최순자", "principal_birth": "450505", "applicant_name": "정현우", "applicant_birth": "750909",
        "relation": "자녀", "document": "인감증명서", "purpose": "부동산 매매", "poa_date": "2026-04-01", "signature": sign()
    });
    let r = submit_form(&mut e, &db(), Cat::Cert, &v, now(), today(), 6).map_err(|e| anyhow::anyhow!("{e:?}"))?;
    let d = intake_detail(&e, &r.receipt_no)?;
    assert_eq!(d.candidate.as_deref(), Some("G1_EXPIRED_MANDATORY_REJECT"), "{:?}", d.candidate);
    Ok(())
}
