//! 엔드투엔드 검증: 명세의 대표 사례를 `Engine` 으로 돌려 확정 경로를 확인합니다.
//! 보조 모드에서는 자동 확정이 없고, 자동화 모드에서도 불리한 처분은 공무원 확인으로 남는지 봅니다.

use anyhow::Result;
use chrono::{NaiveDate, TimeZone, Utc};

use zorogravity_core::decision::{DecisionModel, DecisionTask};
use zorogravity_core::drafting::TextWriter;
use zorogravity_core::audit::AuditCategory;
use zorogravity_core::case::{
    BankFacts, CaseFile, Category, Excuse, LeaseFacts, Link, Node, Person, Relation, ScannedDoc, Signature,
};
use zorogravity_core::policy::{AutomationPolicy, OperatingMode};
use zorogravity_core::privacy::node_hash;
use zorogravity_core::state_machine::gate3_substantive::Justification;
use zorogravity_core::state_machine::{CaseOutcome, Engine, OfficerDecision};
use zorogravity_core::verdict::Verdict;

const SALT: &[u8] = b"test-salt-not-for-production";
const APPLICANT_ID: &str = "9901011234567";
const LANDLORD_ID: &str = "6403122345678";
const MOTHER_ID: &str = "6403129999999";

fn now() -> chrono::DateTime<Utc> {
    Utc.with_ymd_and_hms(2026, 10, 31, 9, 0, 0).unwrap()
}

fn today() -> NaiveDate {
    NaiveDate::from_ymd_opt(2026, 10, 31).unwrap()
}

fn doc(kind: &str, hash: &str) -> ScannedDoc {
    ScannedDoc { kind: kind.into(), content_hash: hash.into(), pages: 1 }
}

fn person(name: &str, birth: &str, id: &str) -> Person {
    Person { name: name.into(), birth_prefix: birth.into(), identifier: id.into() }
}

fn node(id: &str, name: &str) -> Node {
    Node { hash: node_hash(SALT, id), name: name.into() }
}

/// 서류·서명·계좌·월세가 모두 맞는 청년월세 신청
fn clean_welfare(id: &str) -> CaseFile {
    CaseFile {
        case_id: id.into(),
        category: Category::Welfare,
        source: Default::default(),
        docs: vec![
            doc("신청서", "h1"),
            doc("임대차계약서", "h2"),
            doc("이체내역서", "h3"),
            doc("금융정보동의서", "h4"),
        ],
        applicant: person("김철수", "990101", APPLICANT_ID),
        landlord: Some(person("박영희", "640312", LANDLORD_ID)),
        lease: Some(LeaseFacts { applied_rent: 500_000, contract_rent: 500_000, latest_transfer: 500_000 }),
        bank: Some(BankFacts {
            handwritten_account: "110-123-991023".into(),
            printed_account: "110-123-991023".into(),
        }),
        signature: Some(Signature { stroke_count: 8, bbox_area_ratio: 0.42, single_line: false }),
        poa_written_on: None,
        today: today(),
        excuse: None,
        relations: vec![],
        supplement_count: 0,
        grievance_text: None,
        contact: None,
        intake: Default::default(),
    }
}

/// 임대인이 모친 본인인 위장 임차 (2촌 이내 혈족)
fn family_violation(id: &str) -> CaseFile {
    let mut c = clean_welfare(id);
    c.relations = vec![Relation {
        a: node(APPLICANT_ID, "김철수"),
        b: node(LANDLORD_ID, "박영희"),
        link: Link::Family { degree: 1 },
    }];
    c
}

/// 임대인과 이름만 같은 모친이 있는 동명이인 상황 (명세 Case 3-A)
fn homonym(id: &str) -> CaseFile {
    let mut c = clean_welfare(id);
    c.relations = vec![Relation {
        a: node(APPLICANT_ID, "김철수"),
        b: node(MOTHER_ID, "박영희"),
        link: Link::Family { degree: 1 },
    }];
    c
}

/// 지정한 선택지에 확률을 몰아 주는 시험용 Zoro-1 모델. 그 선택지가 없으면 같은 성격의 선택지에, 그것도 없으면 균등하게 줍니다.
struct Prefers {
    verdict: Verdict,
    confidence: f32,
}

impl DecisionModel for Prefers {
    fn probabilities(&self, task: &DecisionTask) -> anyhow::Result<Vec<f32>> {
        let n = task.options.len();
        let same = task.options.iter().position(|v| *v == self.verdict);
        let Some(idx) = same.or_else(|| task.options.iter().position(|v| v.class() == self.verdict.class())) else {
            return Ok(vec![1.0 / n as f32; n]);
        };
        let rest = if n > 1 { (1.0 - self.confidence) / (n - 1) as f32 } else { 0.0 };
        Ok((0..n).map(|i| if i == idx { self.confidence } else { rest }).collect())
    }
}

fn preferring(verdict: Verdict) -> Prefers {
    Prefers { verdict, confidence: 0.99 }
}

fn auto_engine(model: Option<Prefers>) -> Result<Engine> {
    let model: Option<Box<dyn DecisionModel>> = model.map(|m| Box::new(m) as Box<dyn DecisionModel>);
    let mut e = Engine::new(SALT, AutomationPolicy::default(), model);
    e.enable_automation("officer-17", "지침 제2조 (테스트용 법령 근거)", now())?;
    Ok(e)
}

fn archive_of(e: &Engine, idx: usize) -> &zorogravity_core::store::ArchiveRow {
    &e.store.archive()[idx]
}

#[test]
fn assist_mode_never_finalizes_on_its_own() -> Result<()> {
    let model = preferring(Verdict::G3EligibleNormal);
    let mut e = Engine::new(SALT, AutomationPolicy::default(), Some(Box::new(model)));
    assert_eq!(e.policy.mode(), OperatingMode::Assist);

    let out = e.run(clean_welfare("c-assist"), now())?;
    assert!(matches!(out, CaseOutcome::OfficerReview { .. }), "보조 모드는 공무원 확정을 기다립니다: {out:?}");
    assert_eq!(e.store.archive().len(), 0);
    assert_eq!(e.store.active_count(), 1);
    Ok(())
}

#[test]
fn auto_mode_finalizes_a_clean_case_only_with_model_confidence() -> Result<()> {
    let model = preferring(Verdict::G3EligibleNormal);
    let mut e = auto_engine(Some(model))?;

    let out = e.run(clean_welfare("c-auto"), now())?;
    assert!(matches!(out, CaseOutcome::AutoFinalized { .. }), "{out:?}");
    assert_eq!(e.store.archive().len(), 1);
    assert_eq!(e.store.active_count(), 0, "확정된 민원의 원문 세션은 남지 않습니다");
    assert!(e.store.audit().events().iter().any(|ev| ev.category == AuditCategory::AutoFinalized));
    assert!(e.store.audit().events().iter().any(|ev| ev.category == AuditCategory::AutomationChanged));
    Ok(())
}

#[test]
fn auto_mode_without_a_model_still_waits_for_the_officer() -> Result<()> {
    // 규칙은 통과해도 확신도(모델)가 없으면 자동 확정하지 않습니다.
    let mut e = auto_engine(None)?;
    let out = e.run(clean_welfare("c-nomodel"), now())?;
    assert!(matches!(out, CaseOutcome::OfficerReview { .. }), "{out:?}");
    assert_eq!(e.store.archive().len(), 0);
    Ok(())
}

#[test]
fn auto_mode_never_finalizes_a_violation_even_with_full_confidence() -> Result<()> {
    // 모델이 위반에 확률 1.0 을 줘도, 불리한 처분은 자동 확정되지 않아야 합니다.
    let model = Prefers { verdict: Verdict::G3ViolationHighRisk, confidence: 1.0 };
    let mut e = auto_engine(Some(model))?;

    let out = e.run(family_violation("c-violation"), now())?;
    match out {
        CaseOutcome::OfficerReview { judgement: Some(j), .. } => {
            assert_eq!(j.verdict, Verdict::G3ViolationHighRisk);
            assert!(j.confidence.unwrap_or(0.0) > 0.99, "모델은 강하게 동의했습니다");
        }
        other => panic!("불리한 처분은 공무원 확인으로 남아야 합니다: {other:?}"),
    }
    assert_eq!(e.store.archive().len(), 0);
    Ok(())
}

#[test]
fn model_cannot_invent_an_adverse_decision_without_evidence() -> Result<()> {
    // 관계망 증거가 없는데 모델이 위반이라고 답하면, 규칙 판정(적격)이 남고 확신도는 없어야 합니다.
    let model = preferring(Verdict::G3ViolationHighRisk);
    let mut e = auto_engine(Some(model))?;
    let out = e.run(clean_welfare("c-no-evidence"), now())?;
    match out {
        CaseOutcome::OfficerReview { judgement: Some(j), .. } => {
            assert_eq!(j.verdict, Verdict::G3EligibleNormal);
            assert_eq!(j.confidence, None, "모델이 다른 답을 냈으므로 자동 확정 근거가 없습니다");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(e.store.archive().len(), 0);
    Ok(())
}

#[test]
fn homonym_is_cleared_and_can_be_finalized_when_automation_is_on() -> Result<()> {
    let model = preferring(Verdict::G3EligibleHomonymClear);
    let mut e = auto_engine(Some(model))?;
    let out = e.run(homonym("c-homonym"), now())?;
    assert!(matches!(out, CaseOutcome::AutoFinalized { .. }), "{out:?}");
    assert!(e.store.audit().events().iter().any(|ev| ev.detail.contains("동명이인")));
    Ok(())
}

#[test]
fn invalid_signature_goes_back_to_the_kiosk_and_keeps_nothing() -> Result<()> {
    let mut c = clean_welfare("c-sign");
    c.signature = Some(Signature { stroke_count: 1, bbox_area_ratio: 0.038, single_line: true });
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    let out = e.run(c, now())?;
    assert!(matches!(out, CaseOutcome::KioskRetry { verdict: Verdict::G1SignInvalidRetry, .. }), "{out:?}");
    assert_eq!(e.store.active_count(), 0);
    Ok(())
}

#[test]
fn typo_claim_against_the_printed_contract_is_rejected_at_the_kiosk() -> Result<()> {
    // 명세 Case 2-B: 신청서·계약서 50만, 이체 30만, 민원인이 "신청서 오타" 를 누름
    let mut c = clean_welfare("c-typo");
    c.lease = Some(LeaseFacts { applied_rent: 500_000, contract_rent: 500_000, latest_transfer: 300_000 });
    c.excuse = Some(Excuse::TypoInApplication);
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    let out = e.run(c, now())?;
    assert!(matches!(out, CaseOutcome::KioskRetry { verdict: Verdict::G2RejectContradictionLock, .. }), "{out:?}");
    assert!(e.store.audit().events().iter().any(|ev| ev.category == AuditCategory::ContradictionBlocked));
    Ok(())
}

#[test]
fn bank_account_is_locked_to_the_printed_value_without_leaking_it_to_the_log() -> Result<()> {
    // 명세 Case 2-A: 수기 ...8 · 인쇄 ...3
    let mut c = clean_welfare("c-bank");
    c.bank = Some(BankFacts {
        handwritten_account: "110-123-991028".into(),
        printed_account: "110-123-991023".into(),
    });
    let model = preferring(Verdict::G3EligibleNormal);
    let mut e = auto_engine(Some(model))?;
    let out = e.run(c, now())?;
    assert!(matches!(out, CaseOutcome::AutoFinalized { .. }), "{out:?}");
    let locked = e
        .store
        .audit()
        .events()
        .iter()
        .find(|ev| ev.category == AuditCategory::FactLocked)
        .expect("계좌 고정 기록이 있어야 합니다");
    assert!(!locked.detail.contains("991023") && !locked.detail.contains("991028"), "감사 로그에 계좌번호가 없어야 합니다");
    Ok(())
}

#[test]
fn expired_power_of_attorney_is_an_adverse_candidate_that_the_officer_confirms() -> Result<()> {
    // 명세 Case 1-C: 2025-12-10 작성, 325일 경과
    let mut c = clean_welfare("c-poa");
    c.category = Category::Cert;
    c.docs = vec![doc("위임장", "p1"), doc("위임자신분증사본", "p2"), doc("대리인신분증", "p3")];
    c.landlord = None;
    c.lease = None;
    c.bank = None;
    c.poa_written_on = Some(NaiveDate::from_ymd_opt(2025, 12, 10).unwrap());
    let mut e = auto_engine(None)?;

    let out = e.run(c, now())?;
    assert!(matches!(
        out,
        CaseOutcome::OfficerReview { judgement: Some(ref j), .. } if j.verdict == Verdict::G1ExpiredMandatoryReject
    ), "{out:?}");

    let done = e.confirm("c-poa", "officer-17", OfficerDecision::Confirm, now())?;
    assert!(matches!(done, CaseOutcome::Finalized { .. }), "{done:?}");
    let row = archive_of(&e, 0);
    assert_eq!(row.final_code, "G1_EXPIRED_MANDATORY_REJECT");
    assert!(row.citizen_doc.contains("재작성"), "민원인 문서에는 재작성 안내가 있어야 합니다");
    assert!(!row.citizen_doc.contains("LatticeDB") && !row.citizen_doc.contains("위장"));
    Ok(())
}

#[test]
fn officer_cannot_force_pass_missing_documents() -> Result<()> {
    // 명세 Case 1-D: 이체내역서가 없는데 강제 진입
    let mut c = clean_welfare("c-force");
    c.docs.retain(|d| d.kind != "이체내역서");
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    let out = e.run(c, now())?;
    assert!(matches!(out, CaseOutcome::OfficerReview { .. }), "{out:?}");

    let blocked = e.confirm("c-force", "officer-17", OfficerDecision::ForcePassToGate2, now())?;
    assert!(matches!(blocked, CaseOutcome::Blocked { verdict: Verdict::G1InterceptForceHold, .. }), "{blocked:?}");
    assert_eq!(e.store.active_count(), 1, "차단돼도 세션은 남습니다");
    assert!(e.store.audit().events().iter().any(|ev| ev.category == AuditCategory::IllegalActionIntercepted));

    let issued = e.confirm("c-force", "officer-17", OfficerDecision::IssueSupplement, now())?;
    assert!(matches!(issued, CaseOutcome::Finalized { .. }), "{issued:?}");
    let row = archive_of(&e, 0);
    assert_eq!(row.final_code, "SUPPLEMENT_7DAYS");
    assert!(row.citizen_doc.contains("2026-11-07"), "보완 기한은 발급일 + 7일");
    Ok(())
}

#[test]
fn rubber_stamping_a_high_risk_case_needs_a_real_justification() -> Result<()> {
    // 명세 Case 3-D
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    let out = e.run(family_violation("c-stamp"), now())?;
    assert!(matches!(out, CaseOutcome::OfficerReview { .. }), "{out:?}");

    let blocked = e.confirm(
        "c-stamp",
        "officer-17",
        OfficerDecision::ApproveHighRisk(Justification::ProveFalsePositive { evidence_scans: 0 }),
        now(),
    )?;
    assert!(matches!(blocked, CaseOutcome::Blocked { verdict: Verdict::G3InterceptRubberStamp, .. }), "{blocked:?}");

    let no_cosign = e.confirm(
        "c-stamp",
        "officer-17",
        OfficerDecision::ApproveHighRisk(Justification::ApplyLawException {
            team_lead_cosign: false,
            rule_ref: "조례 제5조".into(),
        }),
        now(),
    )?;
    assert!(matches!(no_cosign, CaseOutcome::Blocked { .. }), "{no_cosign:?}");

    let approved = e.confirm(
        "c-stamp",
        "officer-17",
        OfficerDecision::ApproveHighRisk(Justification::ApplyLawException {
            team_lead_cosign: true,
            rule_ref: "조례 제5조".into(),
        }),
        now(),
    )?;
    assert!(matches!(approved, CaseOutcome::Finalized { .. }), "{approved:?}");
    assert!(archive_of(&e, 0).special_audit_flag, "예외 승인은 상급 감사실 직보 플래그가 붙습니다");
    Ok(())
}

#[test]
fn cancel_mistake_reverts_a_high_risk_approval_to_rejection() -> Result<()> {
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    e.run(family_violation("c-revert"), now())?;
    let out = e.confirm(
        "c-revert",
        "officer-17",
        OfficerDecision::ApproveHighRisk(Justification::CancelMistake),
        now(),
    )?;
    assert!(matches!(out, CaseOutcome::Finalized { .. }), "{out:?}");
    assert!(archive_of(&e, 0).final_code.starts_with("REVERTED_TO_REJECT"));
    Ok(())
}

#[test]
fn third_party_unmask_follows_the_seventy_percent_rule_and_is_logged() -> Result<()> {
    // 모델 없음 → 위험도 0 → 열람 거부
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    e.run(family_violation("c-unmask-1"), now())?;
    assert!(!e.request_unmask("c-unmask-1", "officer-17", now())?);

    // 위험도 96.4% → 열람 허용
    let model = preferring(Verdict::G3ViolationHighRisk);
    let mut e2 = Engine::new(SALT, AutomationPolicy::default(), Some(Box::new(model)));
    e2.run(family_violation("c-unmask-2"), now())?;
    assert!(e2.request_unmask("c-unmask-2", "officer-17", now())?);

    let kinds: Vec<AuditCategory> = e2.store.audit().events().iter().map(|ev| ev.category).collect();
    assert!(kinds.contains(&AuditCategory::PrivacyUnmaskGranted));
    assert!(!e2.store.audit().events().iter().any(|ev| ev.detail.contains("박영희")), "열람 대상 이름은 로그에 남기지 않습니다");
    Ok(())
}

#[test]
fn grievances_always_go_to_an_officer() -> Result<()> {
    let mut c = clean_welfare("c-grievance");
    c.category = Category::Grievance;
    c.docs = vec![];
    let model = preferring(Verdict::G3EligibleNormal);
    let mut e = auto_engine(Some(model))?;
    let out = e.run(c, now())?;
    assert!(matches!(out, CaseOutcome::OfficerReview { .. }), "{out:?}");
    Ok(())
}

/// 몇 번 물었는지 세는 대역
struct Counts(std::sync::Arc<std::sync::atomic::AtomicUsize>);

impl DecisionModel for Counts {
    fn probabilities(&self, task: &DecisionTask) -> anyhow::Result<Vec<f32>> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let n = task.options.len();
        Ok(vec![1.0 / n as f32; n])
    }
}

#[test]
fn the_category_is_never_asked_of_the_model() -> Result<()> {
    let asked = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut e = Engine::new(SALT, AutomationPolicy::default(), Some(Box::new(Counts(asked.clone()))));
    e.run(grievance("c-asked", NOISE), now())?;
    assert_eq!(asked.load(std::sync::atomic::Ordering::SeqCst), 0, "고충민원은 Gate 0 에서 끝나고, Gate 0 은 모델에 묻지 않습니다");
    Ok(())
}

#[test]
fn enabling_automation_requires_a_legal_basis_and_is_audited() -> Result<()> {
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    assert!(e.enable_automation("officer-17", "  ", now()).is_err());
    assert_eq!(e.policy.mode(), OperatingMode::Assist);
    e.enable_automation("officer-17", "지침 제2조", now())?;
    assert_eq!(e.policy.mode(), OperatingMode::Auto);
    e.disable_automation("officer-17", now());
    assert_eq!(e.policy.mode(), OperatingMode::Assist);
    assert_eq!(
        e.store.audit().events().iter().filter(|ev| ev.category == AuditCategory::AutomationChanged).count(),
        2
    );
    Ok(())
}

#[test]
fn same_input_gives_the_same_path() -> Result<()> {
    let run_once = || -> Result<String> {
        let model = preferring(Verdict::G3EligibleNormal);
        let mut e = auto_engine(Some(model))?;
        match e.run(clean_welfare("c-repeat"), now())? {
            CaseOutcome::AutoFinalized { post_audit, .. } => Ok(format!("auto:{post_audit}")),
            other => Ok(format!("{other:?}")),
        }
    };
    assert_eq!(run_once()?, run_once()?);
    Ok(())
}

/// 정해 둔 글을 쓰는 대역. 지시문에 따라 답을 고르고, 받은 입력을 모두 기록합니다.
struct Pen {
    summary: &'static str,
    excuse: &'static str,
    opinion: &'static str,
    asked: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl TextWriter for Pen {
    fn complete(&self, system: &str, user: &str, _max_tokens: usize) -> Result<String> {
        self.asked.lock().unwrap().push(user.to_string());
        if system.contains("표준 코드") {
            Ok(self.excuse.to_string())
        } else if system.contains("심사 의견") {
            Ok(self.opinion.to_string())
        } else {
            Ok(self.summary.to_string())
        }
    }
}

fn pen(summary: &'static str, excuse: &'static str, opinion: &'static str) -> Box<Pen> {
    Box::new(Pen { summary, excuse, opinion, asked: Default::default() })
}

/// 일부러 실패하는 대역 (가중치를 올리지 못한 경우 등)
struct BrokenPen;

impl TextWriter for BrokenPen {
    fn complete(&self, _: &str, _: &str, _: usize) -> Result<String> {
        anyhow::bail!("장치 메모리 부족")
    }
}

fn grievance(id: &str, text: &str) -> CaseFile {
    let mut c = clean_welfare(id);
    c.category = Category::Grievance;
    c.docs = vec![doc("진정서", "g1")];
    c.grievance_text = Some(text.into());
    c
}

const NOISE: &str = "매일 아침 일곱 시부터 저녁 여덟 시까지 공사 소음이 이어집니다. 방음벽을 설치해 주세요. 연락처 010-1234-5678";

#[test]
fn a_grievance_gets_a_summary_for_the_officer_and_nothing_is_finalized() -> Result<()> {
    let writer = pen("요지: 일곱 시부터 저녁 여덟 시까지 공사 소음\n요구: 방음벽 설치\n부서: 건설관리과", "", "");
    let asked = writer.asked.clone();
    let mut e = auto_engine(Some(preferring(Verdict::Cat05Grievance)))?.with_writer(writer);
    let out = e.run(grievance("c-g", NOISE), now())?;
    let CaseOutcome::OfficerReview { reason, .. } = out else { panic!("고충민원은 공무원이 처리합니다: {out:?}") };
    assert!(reason.contains("건설관리과"), "{reason}");
    assert_eq!(e.store.archive().len(), 0, "자동화 모드에서도 초안은 처분이 아닙니다");
    let draft = e.store.active("c-g").unwrap().draft.as_ref().expect("초안");
    assert!(draft.verified);
    assert_eq!(draft.demand, "방음벽 설치");
    assert!(!asked.lock().unwrap().iter().any(|t| t.contains("1234-5678")), "전화번호는 모델에 가지 않습니다");
    assert!(!e.store.audit().events().iter().any(|ev| format!("{ev:?}").contains("공사 소음")), "감사 로그에 민원 내용을 남기지 않습니다");
    Ok(())
}

#[test]
fn an_invented_number_is_flagged_instead_of_being_passed_along() -> Result<()> {
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None).with_writer(pen("요지: 매일 7시~18시 소음\n요구: 방음벽\n부서: 건설관리과", "", ""));
    let out = e.run(grievance("c-g2", NOISE), now())?;
    let CaseOutcome::OfficerReview { reason, .. } = out else { panic!("{out:?}") };
    assert!(reason.contains("확인할 내용"), "{reason}");
    let draft = e.store.active("c-g2").unwrap().draft.as_ref().unwrap();
    assert!(!draft.verified);
    assert!(draft.problems.iter().any(|p| p.contains("18")), "{:?}", draft.problems);
    Ok(())
}

#[test]
fn a_failing_writer_does_not_stop_the_case() -> Result<()> {
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None).with_writer(Box::new(BrokenPen));
    let out = e.run(grievance("c-g3", NOISE), now())?;
    assert!(matches!(out, CaseOutcome::OfficerReview { .. }), "{out:?}");
    assert!(e.store.active("c-g3").unwrap().draft.is_none());
    Ok(())
}

#[test]
fn a_free_text_excuse_is_normalized_into_a_receipt_request() -> Result<()> {
    // 명세 Case 2-C: 보일러 수리비를 월세에서 뺐다는 자유 서술
    let free_text = || {
        let mut c = clean_welfare("c-free");
        c.lease = Some(LeaseFacts { applied_rent: 500_000, contract_rent: 500_000, latest_transfer: 300_000 });
        c.excuse = Some(Excuse::FreeText("지난달에 보일러가 고장 나서 수리비 20만 원을 제가 먼저 내고 월세에서 깠어요".into()));
        c
    };
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None).with_writer(pen("", "CLAIM_REPAIR_DEDUCTION_NO_RECEIPT", ""));
    assert!(matches!(e.run(free_text(), now())?, CaseOutcome::KioskRetry { verdict: Verdict::G2RequireRepairReceipt, .. }));

    // 분류하지 못하면 예전처럼 공무원에게 갑니다.
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None).with_writer(pen("", "CLAIM_OTHER", ""));
    assert!(matches!(e.run(free_text(), now())?, CaseOutcome::OfficerReview { .. }));
    let mut e = Engine::new(SALT, AutomationPolicy::default(), None);
    assert!(matches!(e.run(free_text(), now())?, CaseOutcome::OfficerReview { .. }), "모델이 없어도 같습니다");
    Ok(())
}

#[test]
fn the_internal_document_takes_only_an_opinion_whose_numbers_check_out() -> Result<()> {
    let run = |opinion: &'static str| -> Result<String> {
        let mut e = auto_engine(Some(preferring(Verdict::G3EligibleNormal)))?.with_writer(pen("", "", opinion));
        assert!(matches!(e.run(clean_welfare("c-op"), now())?, CaseOutcome::AutoFinalized { .. }));
        Ok(archive_of(&e, 0).officer_doc.clone())
    };
    assert!(run("계약 월세 500,000원이 그대로 이체되어 적격입니다.")?.contains("심사 의견 초안 (모델 작성, 공무원 확인 전): 계약 월세 500,000원"));
    assert!(!run("계약 월세 300,000원이 이체되었습니다.")?.contains("심사 의견"), "숫자가 어긋나면 넣지 않습니다");
    Ok(())
}

#[test]
fn drafts_never_change_the_decision() -> Result<()> {
    // 쓰는 모델이 위반 건을 적격이라고 써도 처분은 규칙이 정합니다.
    let mut e = auto_engine(Some(Prefers { verdict: Verdict::G3ViolationHighRisk, confidence: 1.0 }))?.with_writer(pen("", "", "문제없이 적격입니다."));
    let out = e.run(family_violation("c-draft"), now())?;
    let CaseOutcome::OfficerReview { judgement: Some(j), .. } = out else { panic!("{out:?}") };
    assert_eq!(j.verdict, Verdict::G3ViolationHighRisk);
    assert_eq!(e.store.archive().len(), 0);
    Ok(())
}
