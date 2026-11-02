//! 키오스크 접수와 공무원 창구 처리.
//!
//! 접수: 입력 검사(`forms`) → 증빙 대조(`evidence`) → 심사(`Engine::run`).
//! 민원인에게 돌려주는 것은 접수 번호와 교부 문서뿐입니다. 요약 · 부서 후보 · 처분 후보는 공무원 콘솔에만 남습니다.
//! 창구 처리: 공무원이 원본을 확인하면 값을 반영해 다시 심사하고, 후보 처분을 확인한 뒤 확정합니다.

use std::fmt;

use anyhow::anyhow;
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use serde_json::Value;

use crate::audit::{Actor, AuditCategory, AuditEvent};
use crate::case::{BankFacts, CaseFile, CaseSource, Category, DocEvidence, Evidence, Identity, IntakeFacts, LeaseFacts, Link, Person, Receipt as PaidReceipt};
use crate::drafting::DEPARTMENTS;
use crate::evidence;
use crate::forms;
use crate::lattice::Lattice;
use crate::privacy;
use crate::state_machine::gate0_router;
use crate::state_machine::gate2_consistency::{check_manual_amount, rent_gap, review_receipts};
use crate::state_machine::gate3_substantive::Justification;
use crate::state_machine::{CaseOutcome, Engine, OfficerDecision};
use crate::store::TransferRow;
use crate::verdict::{OutcomeClass, Verdict};

/// 고충민원 본문 길이 (글자 수, 앞뒤 공백 제외)
pub const BODY_MIN_CHARS: usize = 10;
pub const BODY_MAX_CHARS: usize = 4_000;
/// 성명 길이 상한 (글자 수)
pub const NAME_MAX_CHARS: usize = 20;

/// 키오스크에서 받는 고충민원 입력
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct GrievanceInput {
    pub name: String,
    /// 생년월일 앞 6자리 (YYMMDD)
    pub birth_prefix: String,
    pub body: String,
    /// 회신용 연락처. 비워 두어도 됩니다.
    pub contact: Option<String>,
}

/// 입력 오류. 메시지는 민원인 화면에 그대로 보여 줍니다. 내부 정보는 넣지 않습니다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormError {
    NameRequired,
    NameInvalid,
    BirthPrefixInvalid,
    BodyTooShort,
    BodyTooLong,
    ContactInvalid,
    /// 양식 입력 검사 메시지 (고충민원 밖의 민원). 민원인 화면에 그대로 보입니다.
    Invalid(String),
}

impl fmt::Display for FormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::NameRequired => "성함을 적어 주세요.".to_string(),
            Self::NameInvalid => format!("성함은 숫자 없이 {}자 이내로 적어 주세요.", NAME_MAX_CHARS),
            Self::BirthPrefixInvalid => "생년월일 앞 6자리를 확인해 주세요. 예: 990101".to_string(),
            Self::BodyTooShort => format!("내용을 {}자 이상 적어 주세요.", BODY_MIN_CHARS),
            Self::BodyTooLong => format!("내용은 {}자 이내로 적어 주세요.", BODY_MAX_CHARS),
            Self::ContactInvalid => "연락처를 확인해 주세요. 예: 010-1234-5678".to_string(),
            Self::Invalid(msg) => msg.clone(),
        };
        f.write_str(&text)
    }
}

impl std::error::Error for FormError {}

/// YYMMDD 를 날짜로 바꿉니다. 세기는 오늘보다 미래가 되지 않도록 고릅니다.
pub(crate) fn birth_date(prefix: &str, today: NaiveDate) -> Option<NaiveDate> {
    let p = prefix.trim();
    if p.len() != 6 || !p.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let yy: i32 = p[0..2].parse().ok()?;
    let mm: u32 = p[2..4].parse().ok()?;
    let dd: u32 = p[4..6].parse().ok()?;
    let this_century = today.year() - today.year() % 100;
    let year = if yy <= today.year() % 100 { this_century + yy } else { this_century - 100 + yy };
    let date = NaiveDate::from_ymd_opt(year, mm, dd)?;
    (date <= today).then_some(date)
}

/// 연락처는 숫자 9~11자리이고 0으로 시작해야 합니다. 하이픈과 공백은 무시합니다.
fn contact_ok(contact: &str) -> bool {
    let digits: String = contact.chars().filter(|c| !matches!(c, '-' | ' ')).collect();
    (9..=11).contains(&digits.len()) && digits.starts_with('0') && digits.bytes().all(|b| b.is_ascii_digit())
}

/// 입력을 검사합니다. 오늘 날짜는 생년월일의 세기를 정하고 미래 날짜를 거르는 데 씁니다.
pub fn validate_grievance(input: &GrievanceInput, today: NaiveDate) -> Result<(), FormError> {
    let name = input.name.trim();
    if name.is_empty() {
        return Err(FormError::NameRequired);
    }
    if name.chars().count() > NAME_MAX_CHARS || name.chars().any(|c| c.is_ascii_digit()) {
        return Err(FormError::NameInvalid);
    }
    if birth_date(&input.birth_prefix, today).is_none() {
        return Err(FormError::BirthPrefixInvalid);
    }
    let len = input.body.trim().chars().count();
    if len < BODY_MIN_CHARS {
        return Err(FormError::BodyTooShort);
    }
    if len > BODY_MAX_CHARS {
        return Err(FormError::BodyTooLong);
    }
    if let Some(contact) = input.contact.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
        if !contact_ok(contact) {
            return Err(FormError::ContactInvalid);
        }
    }
    Ok(())
}

/// 검사를 통과한 입력으로 고충민원 한 건을 만듭니다.
/// 주민등록번호는 키오스크에서 받지 않습니다 (`identifier` 는 비워 둡니다).
pub fn grievance_case(input: GrievanceInput, case_id: &str, today: NaiveDate) -> Result<CaseFile, FormError> {
    validate_grievance(&input, today)?;
    let contact = input.contact.map(|c| c.trim().to_string()).filter(|c| !c.is_empty());
    Ok(CaseFile {
        case_id: case_id.into(),
        category: Category::Grievance,
        source: CaseSource::KioskForm,
        docs: vec![],
        applicant: Person { name: input.name.trim().into(), birth_prefix: input.birth_prefix.trim().into(), identifier: String::new() },
        landlord: None,
        lease: None,
        bank: None,
        signature: None,
        poa_written_on: None,
        today,
        excuse: None,
        relations: vec![],
        supplement_count: 0,
        grievance_text: Some(input.body.trim().into()),
        contact,
        intake: IntakeFacts::default(),
    })
}

/// 접수 번호: 민원 종류 글자 + 날짜 8자리 + `-` + 16진수 6자리. 번호는 비밀이 아니고, 겹치면 다시 뽑습니다.
pub fn receipt_no(prefix: char, today: NaiveDate, token: u32) -> String {
    format!("{prefix}{:04}{:02}{:02}-{:06X}", today.year(), today.month(), today.day(), token & 0x00FF_FFFF)
}

/// 민원인 화면이 받는 상태
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KioskStatus {
    /// 담당 주무관이 확인합니다
    Waiting,
    /// 처리가 끝났습니다. 교부 문서를 함께 보여 줍니다.
    Finished { notice: String },
}

/// 민원인에게 돌려주는 것: 접수 번호와 상태
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub receipt_no: String,
    pub status: KioskStatus,
}

#[derive(Debug)]
pub enum SubmitError {
    /// 민원인이 고쳐서 다시 낼 수 있는 입력 오류
    Form(FormError),
    /// 서명을 다시 받습니다. 민원인 화면에 그대로 보입니다.
    Retry(String),
    /// 이체액과 계약 월세가 달라 소명을 받습니다. `locked` 는 고를 수 없는 소명입니다.
    NeedExcuse { message: String, locked: Vec<String> },
    /// 시스템 오류. 민원인 화면에는 일반 안내만 보입니다.
    Internal(anyhow::Error),
}

fn free_receipt(engine: &Engine, prefix: char, today: NaiveDate, mut token: u32) -> String {
    let mut receipt = receipt_no(prefix, today, token);
    while engine.store.active(&receipt).is_some() || engine.store.archive().iter().any(|a| a.case_id == receipt) {
        token = token.wrapping_add(1);
        receipt = receipt_no(prefix, today, token);
    }
    receipt
}

fn kiosk_status(engine: &Engine, receipt: &str, outcome: CaseOutcome) -> Result<KioskStatus, SubmitError> {
    match outcome {
        CaseOutcome::OfficerReview { .. } => Ok(KioskStatus::Waiting),
        CaseOutcome::AutoFinalized { .. } | CaseOutcome::Finalized { .. } => {
            let notice = engine.store.archive().iter().find(|a| a.case_id == receipt).map(|a| a.citizen_doc.clone()).unwrap_or_default();
            Ok(KioskStatus::Finished { notice })
        }
        other => Err(SubmitError::Internal(anyhow!("접수가 처리 경로로 가지 않았습니다: {other:?}"))),
    }
}

/// 고충민원 접수: 검사 → 감사 기록 → 파이프라인 (요약 초안 후 공무원 확인 대기) → 접수 번호.
/// `now` 는 감사 기록 시각이고, `today` 는 접수 번호와 심사 기준일입니다. 앱은 현지 날짜를 넘깁니다.
/// `token` 은 접수 번호에 쓸 난수입니다. 이미 있는 번호와 겹치면 하나씩 올려 다시 씁니다.
pub fn submit_grievance(engine: &mut Engine, input: GrievanceInput, now: DateTime<Utc>, today: NaiveDate, token: u32) -> Result<Receipt, SubmitError> {
    let receipt = free_receipt(engine, 'G', today, token);
    let case = grievance_case(input, &receipt, today).map_err(SubmitError::Form)?;
    engine.store.record(AuditEvent::new(now, Actor::CitizenKiosk, AuditCategory::DecisionRecorded, &receipt, None, "고충민원 키오스크 접수"));
    let outcome = engine.run(case, now).map_err(SubmitError::Internal)?;
    let status = kiosk_status(engine, &receipt, outcome)?;
    Ok(Receipt { receipt_no: receipt, status })
}

/// 키오스크 접수: 입력 검사 → 증빙 대조 → 감사 기록 → 심사 → 접수 번호.
/// 서명이 부족하면 `Retry`, 차액 소명이 필요하면 `NeedExcuse` 로 돌아갑니다. 그 밖의 결과는 공무원 확인 대기나 자동 확정입니다.
#[allow(clippy::too_many_arguments)]
pub fn submit_form(
    engine: &mut Engine,
    db: &Lattice,
    category: Category,
    values: &Value,
    now: DateTime<Utc>,
    today: NaiveDate,
    token: u32,
) -> Result<Receipt, SubmitError> {
    let draft = forms::build(category, values, today).map_err(SubmitError::Form)?;
    let hash = |id: &str| engine.node_hash(id);
    let mut case = evidence::enrich(draft, db, &hash);
    // 차액을 민원인에게 알려도 되는 경우는 본인 확인이 끝났을 때뿐입니다. 그 전에는 창구에서 묻습니다.
    if case.intake.identity != Identity::Declared && case.excuse.is_none() {
        if let Some(gap) = case.lease.as_ref().and_then(rent_gap).filter(|g| *g != 0) {
            return Err(SubmitError::NeedExcuse { message: format!("이체한 금액과 계약 월세가 {}원 다릅니다. 어떤 경우인가요?", gap.abs()), locked: Vec::new() });
        }
    }
    let receipt = free_receipt(engine, forms::prefix(category), today, token);
    case.case_id = receipt.clone();
    let label = forms::label(category);
    engine.store.record(AuditEvent::new(now, Actor::CitizenKiosk, AuditCategory::DecisionRecorded, &receipt, None, format!("{label} 키오스크 접수")));
    if category == Category::Welfare {
        engine.store.record(AuditEvent::new(now, Actor::CitizenKiosk, AuditCategory::DecisionRecorded, &receipt, None, "금융정보 조회 동의 확인"));
    }
    match engine.run(case, now).map_err(SubmitError::Internal)? {
        CaseOutcome::KioskRetry { verdict: Verdict::G2RejectContradictionLock, reason } => Err(SubmitError::NeedExcuse { message: reason, locked: vec!["typo".to_string()] }),
        CaseOutcome::KioskRetry { reason, .. } => Err(SubmitError::Retry(reason)),
        outcome => {
            let status = kiosk_status(engine, &receipt, outcome)?;
            Ok(Receipt { receipt_no: receipt, status })
        }
    }
}

/// 공무원 콘솔의 창구 체크리스트 (고충민원). 모두 확인해야 이첩할 수 있습니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Checklist {
    /// 신분증으로 본인을 확인했습니다 (창구)
    pub identity_checked: bool,
    /// 회신 방법을 확인했습니다 (연락처를 다시 물었거나, 없음을 확인)
    pub contact_checked: bool,
    /// 민원 원문을 읽었습니다 (요약이 있으면 숫자와 날짜를 원문과 맞췄습니다)
    pub body_read: bool,
    /// 담당 부서를 민원 내용에 맞게 골랐습니다
    pub department_checked: bool,
}

impl Checklist {
    /// 확인하지 않은 항목의 이름. 비어 있으면 모두 확인된 것입니다.
    pub fn missing(&self) -> Vec<&'static str> {
        [
            (self.identity_checked, "본인 확인 (신분증)"),
            (self.contact_checked, "회신 방법 확인"),
            (self.body_read, "민원 원문 확인"),
            (self.department_checked, "담당 부서 확인"),
        ]
        .into_iter()
        .filter(|(ok, _)| !ok)
        .map(|(_, name)| name)
        .collect()
    }
}

/// 접수 목록에 보이는 상태
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntakeStatus {
    /// 공무원 확인 대기. `suggested` 는 요약이 제안한 부서이고, 없을 수 있습니다.
    Awaiting { received_at: DateTime<Utc>, suggested: Option<String> },
    /// 담당 부서로 이첩됨. 민원인 원문과 이름은 세션과 함께 지워집니다.
    Transferred(TransferRow),
    /// 확정됨. `auto` 면 자동화 모드가 확정했습니다.
    Finalized { archive_id: String, final_code: String, auto: bool },
}

/// 접수 목록의 한 줄
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntakeItem {
    pub receipt_no: String,
    /// 확인 대기 건만 이름이 남습니다
    pub name: Option<String>,
    pub category: Option<Category>,
    pub status: IntakeStatus,
}

/// 요약 초안 (공무원 화면 전용)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryView {
    pub gist: String,
    pub demand: String,
    pub department: Option<String>,
    pub verified: bool,
    pub problems: Vec<String>,
}

/// 서류별 확인 방법 (공무원 화면 전용)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocStatus {
    pub kind: String,
    /// 스캔 · 대장 · 키오스크 서명 · 창구 · 미확인
    pub how: &'static str,
    pub done: bool,
}

/// 접수 상세 (공무원 화면 전용). 민원인이 적은 내용 전부를 담습니다.
#[derive(Debug, Clone, PartialEq)]
pub struct IntakeDetail {
    pub item: IntakeItem,
    pub birth_prefix: String,
    pub contact: Option<String>,
    pub body: String,
    /// 양식 입력 값 (항목 이름, 값). 고충민원은 비어 있습니다.
    pub lines: Vec<(String, String)>,
    /// 공무원에게 보이는 확인 사유
    pub reason: String,
    /// 규칙이 낸 후보 처분 코드
    pub candidate: Option<String>,
    /// 후보 처분의 성격: "favorable", "adverse", "procedural"
    pub candidate_class: Option<&'static str>,
    pub confidence: Option<f32>,
    /// 필수 서류별 확인 상태
    pub docs: Vec<DocStatus>,
    /// 창구에서 확인할 일
    pub pending: Vec<String>,
    /// 본인 확인이 끝났는지
    pub identity_done: bool,
    /// 관계망 연결 (마스킹). 불리한 후보일 때만 보입니다.
    pub relations: Vec<String>,
    /// 신청 금액 (지원금 정산)
    pub claimed_total: Option<i64>,
    /// 인정 합계 (지원금 정산)
    pub verified_sum: Option<i64>,
    /// 창구에서 값을 넣어야 하는 항목: "lease", "bank", "landlord"
    pub needs: Vec<&'static str>,
    pub summary: Option<SummaryView>,
}

#[derive(Debug)]
pub enum DeskError {
    NotFound(String),
    OfficerIdRequired,
    DepartmentUnknown(String),
    ChecklistIncomplete(Vec<String>),
    /// 규칙이 처리를 막았습니다. 사유를 그대로 보입니다.
    Blocked(String),
    /// 창구 확인을 반영했더니 후보 처분이 바뀌었습니다. 다시 보고 결정합니다.
    Changed { from: Option<String>, to: Option<String> },
    Internal(anyhow::Error),
}

impl fmt::Display for DeskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "확인 대기 목록에 없는 접수 번호입니다: {id}"),
            Self::OfficerIdRequired => f.write_str("담당자 사번을 적어 주세요."),
            Self::DepartmentUnknown(name) => write!(f, "배정할 수 없는 부서입니다: {name}"),
            Self::ChecklistIncomplete(missing) => write!(f, "체크리스트를 모두 확인해 주세요: {}", missing.join(", ")),
            Self::Blocked(reason) => f.write_str(reason),
            Self::Changed { from, to } => write!(
                f,
                "창구 확인을 반영하니 후보 처분이 바뀌었습니다: {} → {}. 다시 확인해 주세요",
                from.as_deref().unwrap_or("없음"),
                to.as_deref().unwrap_or("없음")
            ),
            Self::Internal(e) => write!(f, "{e:#}"),
        }
    }
}

impl std::error::Error for DeskError {}

/// 접수 목록. 확인 대기는 접수 순서대로, 처리된 건은 최근 순서대로 보여 줍니다.
pub fn list_intake(engine: &Engine) -> Vec<IntakeItem> {
    let mut awaiting: Vec<(DateTime<Utc>, IntakeItem)> = engine
        .store
        .active_sessions()
        .map(|s| {
            let received_at = s.received_at;
            let item = IntakeItem {
                receipt_no: s.case.case_id.clone(),
                name: Some(s.case.applicant.name.clone()),
                category: Some(s.case.category),
                status: IntakeStatus::Awaiting { received_at, suggested: s.draft.as_ref().and_then(|d| d.department.clone()) },
            };
            (received_at, item)
        })
        .collect();
    awaiting.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.receipt_no.cmp(&b.1.receipt_no)));

    let transferred = engine.store.transferred().iter().rev().map(|t| IntakeItem {
        receipt_no: t.receipt_no.clone(),
        name: None,
        category: Some(Category::Grievance),
        status: IntakeStatus::Transferred(t.clone()),
    });
    let finalized = engine.store.archive().iter().rev().map(|a| IntakeItem {
        receipt_no: a.case_id.clone(),
        name: None,
        category: None,
        status: IntakeStatus::Finalized {
            archive_id: a.archive_id.clone(),
            final_code: a.final_code.clone(),
            auto: engine.store.audit().events().iter().any(|e| e.case_id == a.case_id && e.category == AuditCategory::AutoFinalized),
        },
    });
    awaiting.into_iter().map(|(_, item)| item).chain(transferred).chain(finalized).collect()
}

fn how_label(how: Evidence) -> &'static str {
    match how {
        Evidence::Scanned => "스캔",
        Evidence::Registry => "대장",
        Evidence::KioskSigned => "키오스크 서명",
        Evidence::Counter => "창구",
    }
}

fn link_label(link: Link) -> String {
    match link {
        Link::Family { degree } => format!("{degree}촌"),
        Link::Household { year } => format!("{year} 동일 세대"),
        Link::SharedAccount => "같은 계좌".to_string(),
        Link::Spouse => "배우자".to_string(),
        Link::OfficerOf => "임원".to_string(),
        Link::RepresentedBy => "대표".to_string(),
    }
}

/// 공무원이 창구에서 값을 넣어야 하는 항목
fn needs_of(case: &CaseFile) -> Vec<&'static str> {
    let mut needs = Vec::new();
    if case.category == Category::Welfare {
        if case.lease.is_none() {
            needs.push("lease");
        }
        if case.bank.is_none() {
            needs.push("bank");
        }
        if case.landlord.is_none() {
            needs.push("landlord");
        }
    }
    needs
}

/// 접수 상세. 확인 대기 건만 있습니다 (처리하면 세션과 함께 원문이 지워집니다).
pub fn intake_detail(engine: &Engine, receipt_no: &str) -> Result<IntakeDetail, DeskError> {
    let s = engine.store.active(receipt_no).ok_or_else(|| DeskError::NotFound(receipt_no.into()))?;
    let case = &s.case;
    let docs = gate0_router::mandatory_docs(case.category)
        .iter()
        .map(|kind| {
            let found = case.intake.evidence.iter().find(|e| e.kind == *kind).map(|e| how_label(e.how));
            let scanned = case.docs.iter().any(|d| d.kind == *kind);
            let how = found.or(scanned.then_some("스캔"));
            DocStatus { kind: kind.to_string(), how: how.unwrap_or("미확인"), done: how.is_some() }
        })
        .collect();
    let adverse = s.judgement.as_ref().is_some_and(|j| j.verdict.class() == OutcomeClass::Adverse);
    let relations = if adverse {
        case.relations
            .iter()
            .map(|r| format!("{} — {} — {}", privacy::mask_name(&r.a.name), link_label(r.link), privacy::mask_name(&r.b.name)))
            .collect()
    } else {
        Vec::new()
    };
    let subsidy = case.intake.subsidy.as_ref();
    Ok(IntakeDetail {
        item: IntakeItem {
            receipt_no: receipt_no.into(),
            name: Some(case.applicant.name.clone()),
            category: Some(case.category),
            status: IntakeStatus::Awaiting { received_at: s.received_at, suggested: s.draft.as_ref().and_then(|d| d.department.clone()) },
        },
        birth_prefix: case.applicant.birth_prefix.clone(),
        contact: case.contact.clone(),
        body: case.grievance_text.clone().unwrap_or_default(),
        lines: case.intake.form_lines.clone(),
        reason: s.reason.clone(),
        candidate: s.judgement.as_ref().map(|j| j.verdict.code().to_string()),
        candidate_class: s.judgement.as_ref().map(|j| match j.verdict.class() {
            OutcomeClass::Favorable => "favorable",
            OutcomeClass::Adverse => "adverse",
            OutcomeClass::Procedural => "procedural",
        }),
        confidence: s.judgement.as_ref().and_then(|j| j.confidence),
        docs,
        pending: case.intake.pending.clone(),
        identity_done: case.intake.identity != Identity::Declared,
        relations,
        claimed_total: subsidy.map(|f| f.claimed_total),
        verified_sum: subsidy.map(|f| review_receipts(f).verified_sum),
        needs: needs_of(case),
        summary: s.draft.as_ref().map(|d| SummaryView {
            gist: d.gist.clone(),
            demand: d.demand.clone(),
            department: d.department.clone(),
            verified: d.verified,
            problems: d.problems.clone(),
        }),
    })
}

/// 고충민원 이첩: 담당자 사번, 부서, 체크리스트를 모두 확인한 뒤 이첩 기록을 남기고 세션을 지웁니다.
/// 검사는 모두 상태를 바꾸기 전에 합니다. 이미 이첩된 접수 번호는 목록에 없으므로 다시 이첩할 수 없습니다.
pub fn transfer_grievance(
    engine: &mut Engine,
    receipt_no: &str,
    officer_id: &str,
    department: &str,
    checklist: Checklist,
    now: DateTime<Utc>,
) -> Result<TransferRow, DeskError> {
    let officer_id = officer_id.trim();
    if officer_id.is_empty() {
        return Err(DeskError::OfficerIdRequired);
    }
    if engine.store.active(receipt_no).is_none() {
        return Err(DeskError::NotFound(receipt_no.into()));
    }
    if !DEPARTMENTS.iter().any(|(name, _)| *name == department) {
        return Err(DeskError::DepartmentUnknown(department.into()));
    }
    let missing = checklist.missing();
    if !missing.is_empty() {
        return Err(DeskError::ChecklistIncomplete(missing.into_iter().map(str::to_string).collect()));
    }
    let row = TransferRow { receipt_no: receipt_no.into(), department: department.into(), officer_id: officer_id.into(), at: now };
    let event = AuditEvent::new(now, Actor::OfficerConsole, AuditCategory::OfficerConfirmed, receipt_no, Some(officer_id), format!("고충민원 이첩: {department}"));
    engine.store.close_transferred(receipt_no, row.clone(), vec![event]).map_err(DeskError::Internal)?;
    Ok(row)
}

/// 공무원이 창구에서 확인한 내용
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CounterInput {
    /// 원본을 확인한 서류
    pub docs: Vec<String>,
    /// 신분증을 확인했습니다
    pub identity_checked: bool,
    /// 확인할 일(`pending`)을 모두 봤습니다
    pub pending_checked: bool,
    /// 계약서 원본의 월세
    pub contract_rent: Option<i64>,
    /// 이체내역 원본의 금액
    pub latest_transfer: Option<i64>,
    /// 통장 원본의 계좌번호
    pub printed_account: Option<String>,
    /// 계약서 원본의 임대인 (성명, 생년월일 앞 6자리)
    pub landlord: Option<(String, String)>,
    /// 인정 금액 (지원금 정산). 인정 합계보다 높이려면 그만큼 증빙이 있어야 합니다.
    pub approved_amount: Option<i64>,
    /// 창구에서 새로 확인한 증빙 금액 (지원금 정산)
    pub evidence_sum: Option<i64>,
}

/// 창구 확인을 반영하고 다시 심사합니다. 값이 바뀌면 후보 처분도 다시 계산됩니다.
/// 검사는 세션을 꺼내기 전에 모두 합니다. 막히면 세션은 그대로입니다.
pub fn counter_verify(engine: &mut Engine, db: &Lattice, receipt_no: &str, officer_id: &str, input: &CounterInput, now: DateTime<Utc>) -> Result<CaseOutcome, DeskError> {
    let officer_id = officer_id.trim();
    if officer_id.is_empty() {
        return Err(DeskError::OfficerIdRequired);
    }
    let (category, needs_identity, has_pending, subsidy) = {
        let s = engine.store.active(receipt_no).ok_or_else(|| DeskError::NotFound(receipt_no.into()))?;
        if s.case.source != CaseSource::KioskForm {
            return Err(DeskError::Blocked("창구 확인은 키오스크 접수 건에만 씁니다".to_string()));
        }
        (s.case.category, s.case.intake.identity == Identity::Declared, !s.case.intake.pending.is_empty(), s.case.intake.subsidy.clone())
    };
    let required = gate0_router::mandatory_docs(category);
    if let Some(bad) = input.docs.iter().find(|k| !required.contains(&k.as_str())) {
        return Err(DeskError::Blocked(format!("이 민원에 필요 없는 서류입니다: {bad}")));
    }
    let mut missing = Vec::new();
    if needs_identity && !input.identity_checked {
        missing.push("본인 확인 (신분증)".to_string());
    }
    if has_pending && !input.pending_checked {
        missing.push("확인할 일".to_string());
    }
    if !missing.is_empty() {
        return Err(DeskError::ChecklistIncomplete(missing));
    }
    // 증빙 없는 금액 상향을 막습니다 (명세 Case 2-D).
    if let (Some(facts), Some(amount)) = (&subsidy, input.approved_amount) {
        let verified = review_receipts(facts).verified_sum;
        let evidence = input.evidence_sum.unwrap_or(0);
        if check_manual_amount(verified, amount, evidence).is_err() {
            let detail = format!("증빙 없는 금액 수정 차단: 인정 합계 {verified}원, 입력 {amount}원, 추가 증빙 {evidence}원");
            engine.store.record(AuditEvent::new(now, Actor::ZoroGuardrail, AuditCategory::IllegalActionIntercepted, receipt_no, Some(officer_id), detail));
            return Err(DeskError::Blocked(format!(
                "증빙 없이 금액을 올릴 수 없습니다. 인정 합계 {verified}원, 입력 {amount}원입니다. {}원만큼 증빙을 확인하거나 감액해 주세요",
                amount - verified
            )));
        }
    }

    let session = engine.store.take(receipt_no).ok_or_else(|| DeskError::NotFound(receipt_no.into()))?;
    let received_at = session.received_at;
    let mut case = session.case;
    apply_counter(&mut case, db, input, engine);
    case.intake.counter_done = true;
    case.intake.received_at = Some(received_at);
    let detail = if input.docs.is_empty() { "창구 확인 반영".to_string() } else { format!("창구 원본 확인: {}", input.docs.join(", ")) };
    engine.store.record(AuditEvent::new(now, Actor::OfficerConsole, AuditCategory::DecisionRecorded, receipt_no, Some(officer_id), detail));
    engine.run(case, now).map_err(DeskError::Internal)
}

fn apply_counter(case: &mut CaseFile, db: &Lattice, input: &CounterInput, engine: &Engine) {
    for kind in &input.docs {
        if !case.intake.evidence.iter().any(|e| &e.kind == kind) {
            case.intake.evidence.push(DocEvidence { kind: kind.clone(), how: Evidence::Counter });
        }
    }
    if input.identity_checked {
        case.intake.identity = Identity::Counter;
    }
    if input.pending_checked {
        case.intake.pending.clear();
    }
    if let (Some(contract), Some(transfer)) = (input.contract_rent, input.latest_transfer) {
        let applied = case.lease.as_ref().map(|l| l.applied_rent).unwrap_or(contract);
        case.lease = Some(LeaseFacts { applied_rent: applied, contract_rent: contract, latest_transfer: transfer });
    }
    if let Some(printed) = &input.printed_account {
        let handwritten = case.bank.as_ref().map(|b| b.handwritten_account.clone()).unwrap_or_else(|| printed.clone());
        case.bank = Some(BankFacts { handwritten_account: handwritten, printed_account: printed.clone() });
    }
    if let Some((name, birth)) = &input.landlord {
        let mut person = Person { name: name.clone(), birth_prefix: birth.clone(), identifier: String::new() };
        if let [one] = db.persons_named(name, birth).as_slice() {
            person.identifier = one.id.clone();
            let hash = |id: &str| engine.node_hash(id);
            let mut start = vec![one.id.as_str()];
            if !case.applicant.identifier.is_empty() {
                start.push(case.applicant.identifier.as_str());
            }
            case.relations = db.relations_around(&start, crate::state_machine::gate3_substantive::MAX_HOPS, &hash);
        }
        case.landlord = Some(person);
    }
    if let Some(facts) = case.intake.subsidy.as_mut() {
        if let Some(evidence) = input.evidence_sum.filter(|e| *e > 0) {
            facts.receipts.push(PaidReceipt {
                approval: format!("창구확인-{}", facts.receipts.len() + 1),
                vendor: "창구 확인 증빙".to_string(),
                date: case.today.to_string(),
                time: String::new(),
                amount: evidence,
                item: "창구에서 확인한 증빙".to_string(),
            });
        }
        if let Some(amount) = input.approved_amount {
            facts.claimed_total = amount;
        }
    }
}

/// 창구 확인을 반영하고 확정합니다. 다시 심사해서 후보가 바뀌면 멈추고 공무원에게 다시 보여 줍니다.
/// 불리한 후보는 반려로 확정됩니다. 절차 후보(증빙 부족)는 감액이나 보완요구를 골라야 합니다.
pub fn confirm_form(engine: &mut Engine, db: &Lattice, receipt_no: &str, officer_id: &str, input: &CounterInput, now: DateTime<Utc>) -> Result<String, DeskError> {
    let before = engine.store.active(receipt_no).and_then(|s| s.judgement.as_ref().map(|j| j.verdict.code().to_string()));
    let outcome = counter_verify(engine, db, receipt_no, officer_id, input, now)?;
    let after = match outcome {
        CaseOutcome::AutoFinalized { archive_id, .. } | CaseOutcome::Finalized { archive_id } => return Ok(archive_id),
        CaseOutcome::OfficerReview { judgement, .. } => judgement.map(|j| j.verdict.code().to_string()),
        other => return Err(DeskError::Internal(anyhow!("창구 확인 뒤 예상하지 못한 결과: {other:?}"))),
    };
    // 공무원이 금액을 정한 경우는 후보가 바뀔 것을 알고 고른 것입니다. 정상 후보가 되면 다시 묻지 않습니다.
    let amount_set = input.approved_amount.is_some() && after.as_deref() == Some(Verdict::G3EligibleNormal.code());
    if before != after && !amount_set {
        return Err(DeskError::Changed { from: before, to: after });
    }
    finish_decision(engine, receipt_no, officer_id, OfficerDecision::Confirm, now)
}

/// 공무원의 다른 결정 (반려 · 보완요구 · 고위험 승인 시도)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeskAction {
    Reject,
    Supplement,
    ForcePass,
    ApproveHighRisk(Justification),
}

/// 반려 · 보완요구 · 강제 진입 · 고위험 승인. 규칙이 막으면 사유를 돌려주고 세션은 그대로입니다.
pub fn officer_action(engine: &mut Engine, receipt_no: &str, officer_id: &str, action: DeskAction, now: DateTime<Utc>) -> Result<String, DeskError> {
    let officer_id = officer_id.trim();
    if officer_id.is_empty() {
        return Err(DeskError::OfficerIdRequired);
    }
    let decision = match action {
        DeskAction::Reject => OfficerDecision::Reject,
        DeskAction::Supplement => OfficerDecision::IssueSupplement,
        DeskAction::ForcePass => OfficerDecision::ForcePassToGate2,
        DeskAction::ApproveHighRisk(j) => OfficerDecision::ApproveHighRisk(j),
    };
    finish_decision(engine, receipt_no, officer_id, decision, now)
}

fn finish_decision(engine: &mut Engine, receipt_no: &str, officer_id: &str, decision: OfficerDecision, now: DateTime<Utc>) -> Result<String, DeskError> {
    match engine.confirm(receipt_no, officer_id, decision, now).map_err(DeskError::Internal)? {
        CaseOutcome::Finalized { archive_id } | CaseOutcome::AutoFinalized { archive_id, .. } => Ok(archive_id),
        CaseOutcome::Blocked { reason, .. } | CaseOutcome::OfficerReview { reason, .. } => Err(DeskError::Blocked(reason)),
        other => Err(DeskError::Internal(anyhow!("확정되지 않았습니다: {other:?}"))),
    }
}

/// 제3자 실명 열람 요청. 위험도가 낮으면 거부하고 기록을 남깁니다 (명세 Case 3-C).
pub fn request_unmask(engine: &mut Engine, receipt_no: &str, officer_id: &str, now: DateTime<Utc>) -> Result<bool, DeskError> {
    let officer_id = officer_id.trim();
    if officer_id.is_empty() {
        return Err(DeskError::OfficerIdRequired);
    }
    engine.request_unmask(receipt_no, officer_id, now).map_err(DeskError::Internal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::AutomationPolicy;
    use chrono::TimeZone;

    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, 2).unwrap()
    }

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 11, 2, 1, 0, 0).unwrap()
    }

    fn ok_input() -> GrievanceInput {
        GrievanceInput {
            name: "김철수".into(),
            birth_prefix: "990101".into(),
            body: "매일 아침 일곱 시부터 공사 소음이 이어집니다. 방음벽을 설치해 주세요.".into(),
            contact: Some("010-1234-5678".into()),
        }
    }

    fn engine() -> Engine {
        Engine::new(b"intake-test-salt", AutomationPolicy::default(), None)
    }

    #[test]
    fn a_complete_form_becomes_a_grievance_without_an_identity_number() {
        let c = grievance_case(ok_input(), "G-test", today()).unwrap();
        assert_eq!(c.category, Category::Grievance);
        assert!(c.docs.is_empty());
        assert_eq!(c.applicant.identifier, "");
        assert_eq!(c.contact.as_deref(), Some("010-1234-5678"));
        assert!(c.grievance_text.as_deref().unwrap().starts_with("매일 아침"));
    }

    #[test]
    fn name_is_required_and_has_no_digits() {
        let mut i = ok_input();
        i.name = "  ".into();
        assert_eq!(validate_grievance(&i, today()), Err(FormError::NameRequired));
        i.name = "김철수2".into();
        assert_eq!(validate_grievance(&i, today()), Err(FormError::NameInvalid));
        i.name = "가".repeat(NAME_MAX_CHARS + 1);
        assert_eq!(validate_grievance(&i, today()), Err(FormError::NameInvalid));
    }

    #[test]
    fn birth_prefix_must_be_a_real_past_date() {
        let mut i = ok_input();
        for bad in ["991301", "990231", "abcdef", "99010", "9901011", "261103"] {
            i.birth_prefix = bad.into();
            assert_eq!(validate_grievance(&i, today()), Err(FormError::BirthPrefixInvalid), "{bad}");
        }
        for good in ["990101", "261102", "301010"] {
            i.birth_prefix = good.into();
            assert_eq!(validate_grievance(&i, today()), Ok(()), "{good}");
        }
    }

    #[test]
    fn body_length_counts_characters_after_trimming() {
        let mut i = ok_input();
        i.body = "  가나다라마바사아  ".into();
        assert_eq!(validate_grievance(&i, today()), Err(FormError::BodyTooShort));
        i.body = "가".repeat(BODY_MIN_CHARS);
        assert_eq!(validate_grievance(&i, today()), Ok(()));
        i.body = "가".repeat(BODY_MAX_CHARS);
        assert_eq!(validate_grievance(&i, today()), Ok(()));
        i.body = "가".repeat(BODY_MAX_CHARS + 1);
        assert_eq!(validate_grievance(&i, today()), Err(FormError::BodyTooLong));
    }

    #[test]
    fn contact_is_optional_but_must_look_like_a_phone_number() {
        let mut i = ok_input();
        i.contact = None;
        assert_eq!(validate_grievance(&i, today()), Ok(()));
        i.contact = Some("   ".into());
        assert_eq!(validate_grievance(&i, today()), Ok(()));
        assert_eq!(grievance_case(i.clone(), "G-x", today()).unwrap().contact, None);
        i.contact = Some("02-123-4567".into());
        assert_eq!(validate_grievance(&i, today()), Ok(()));
        i.contact = Some("1234-5678".into());
        assert_eq!(validate_grievance(&i, today()), Err(FormError::ContactInvalid));
        i.contact = Some("010-12ab-5678".into());
        assert_eq!(validate_grievance(&i, today()), Err(FormError::ContactInvalid));
    }

    #[test]
    fn receipt_number_has_the_date_and_six_hex_digits() {
        assert_eq!(receipt_no('G', today(), 0x2A), "G20261102-00002A");
        assert_eq!(receipt_no('G', today(), 0xFFFF_FFFF), "G20261102-FFFFFF");
    }

    #[test]
    fn submission_holds_the_case_for_the_officer_and_returns_only_the_receipt() {
        let mut e = engine();
        let r = submit_grievance(&mut e, ok_input(), now(), today(), 0x1234).unwrap();
        assert_eq!(r.receipt_no, "G20261102-001234");
        assert_eq!(r.status, KioskStatus::Waiting);
        let session = e.store.active(&r.receipt_no).expect("공무원 콘솔로 넘어간 세션");
        assert!(session.reason.contains("공무원이 처리합니다"), "{}", session.reason);
        assert_eq!(e.store.archive().len(), 0, "접수만으로는 확정되지 않습니다");
    }

    #[test]
    fn the_receipt_date_is_the_local_date_the_caller_gives() {
        let mut e = engine();
        // 한국 시각 2026-11-02 01:00 은 UTC 로 2026-11-01 16:00 입니다.
        let late_utc = Utc.with_ymd_and_hms(2026, 11, 1, 16, 0, 0).unwrap();
        let r = submit_grievance(&mut e, ok_input(), late_utc, today(), 9).unwrap();
        assert!(r.receipt_no.starts_with("G20261102-"), "{}", r.receipt_no);
    }

    #[test]
    fn invalid_input_creates_no_session() {
        let mut e = engine();
        let mut i = ok_input();
        i.body = "짧음".into();
        let err = submit_grievance(&mut e, i, now(), today(), 1).unwrap_err();
        assert!(matches!(err, SubmitError::Form(FormError::BodyTooShort)));
        assert_eq!(e.store.active_count(), 0);
    }

    #[test]
    fn the_same_token_never_gives_the_same_receipt_twice() {
        let mut e = engine();
        let a = submit_grievance(&mut e, ok_input(), now(), today(), 7).unwrap();
        let b = submit_grievance(&mut e, ok_input(), now(), today(), 7).unwrap();
        assert_ne!(a.receipt_no, b.receipt_no);
        assert_eq!(e.store.active_count(), 2);
    }

    fn later(minutes: i64) -> DateTime<Utc> {
        now() + chrono::Duration::minutes(minutes)
    }

    fn full_checklist() -> Checklist {
        Checklist { identity_checked: true, contact_checked: true, body_read: true, department_checked: true }
    }

    #[test]
    fn the_queue_lists_awaiting_cases_in_arrival_order() {
        let mut e = engine();
        let second = submit_grievance(&mut e, ok_input(), later(5), today(), 2).unwrap();
        let first = submit_grievance(&mut e, ok_input(), later(1), today(), 1).unwrap();
        let items = list_intake(&e);
        let ids: Vec<&str> = items.iter().map(|i| i.receipt_no.as_str()).collect();
        assert_eq!(ids, vec![first.receipt_no.as_str(), second.receipt_no.as_str()]);
        assert!(items.iter().all(|i| matches!(i.status, IntakeStatus::Awaiting { .. })));
        assert_eq!(items[0].name.as_deref(), Some("김철수"));
    }

    #[test]
    fn detail_has_the_full_text_for_the_officer() {
        let mut e = engine();
        let r = submit_grievance(&mut e, ok_input(), now(), today(), 3).unwrap();
        let d = intake_detail(&e, &r.receipt_no).unwrap();
        assert_eq!(d.birth_prefix, "990101");
        assert_eq!(d.contact.as_deref(), Some("010-1234-5678"));
        assert!(d.body.starts_with("매일 아침"));
        assert!(d.reason.contains("공무원이 처리합니다"), "{}", d.reason);
        assert!(d.summary.is_none(), "요약 모델이 없으면 초안도 없습니다");
        assert!(matches!(intake_detail(&e, "G-none"), Err(DeskError::NotFound(_))));
    }

    #[test]
    fn checklist_names_every_unchecked_item() {
        assert_eq!(Checklist::default().missing().len(), 4);
        assert!(full_checklist().missing().is_empty());
        let partial = Checklist { body_read: true, ..Checklist::default() };
        assert_eq!(partial.missing(), vec!["본인 확인 (신분증)", "회신 방법 확인", "담당 부서 확인"]);
    }

    #[test]
    fn transfer_checks_officer_department_and_checklist_before_changing_anything() {
        let mut e = engine();
        let r = submit_grievance(&mut e, ok_input(), now(), today(), 4).unwrap();
        let id = r.receipt_no.as_str();
        assert!(matches!(transfer_grievance(&mut e, id, "  ", "건설관리과", full_checklist(), now()), Err(DeskError::OfficerIdRequired)));
        assert!(matches!(transfer_grievance(&mut e, id, "A-1", "없는과", full_checklist(), now()), Err(DeskError::DepartmentUnknown(_))));
        let partial = Checklist { identity_checked: true, ..Checklist::default() };
        match transfer_grievance(&mut e, id, "A-1", "건설관리과", partial, now()) {
            Err(DeskError::ChecklistIncomplete(missing)) => assert_eq!(missing.len(), 3),
            other => panic!("체크리스트 오류가 나야 합니다: {other:?}"),
        }
        assert!(matches!(transfer_grievance(&mut e, "G-none", "A-1", "건설관리과", full_checklist(), now()), Err(DeskError::NotFound(_))));
        assert_eq!(e.store.active_count(), 1, "실패한 이첩은 세션을 건드리지 않습니다");
    }

    #[test]
    fn transfer_moves_the_case_out_of_the_queue_and_records_the_officer() {
        let mut e = engine();
        let r = submit_grievance(&mut e, ok_input(), now(), today(), 5).unwrap();
        let row = transfer_grievance(&mut e, &r.receipt_no, " A-1 ", "건설관리과", full_checklist(), later(3)).unwrap();
        assert_eq!(row.officer_id, "A-1");
        assert_eq!(e.store.active_count(), 0, "원문 세션은 지워집니다");
        let items = list_intake(&e);
        assert_eq!(items.len(), 1);
        match &items[0].status {
            IntakeStatus::Transferred(t) => {
                assert_eq!(t.department, "건설관리과");
                assert_eq!(t.at, later(3));
            }
            other => panic!("이첩 완료로 보여야 합니다: {other:?}"),
        }
        assert!(items[0].name.is_none(), "이첩 뒤에는 이름이 남지 않습니다");
        let event = e.store.audit().events().iter().find(|ev| ev.case_id == r.receipt_no && ev.category == AuditCategory::OfficerConfirmed).expect("이첩 감사 기록");
        assert_eq!(event.officer_id.as_deref(), Some("A-1"));
        assert_eq!(event.detail, "고충민원 이첩: 건설관리과");
        assert!(matches!(intake_detail(&e, &r.receipt_no), Err(DeskError::NotFound(_))));
        assert!(matches!(transfer_grievance(&mut e, &r.receipt_no, "A-1", "건설관리과", full_checklist(), now()), Err(DeskError::NotFound(_))), "두 번 이첩할 수 없습니다");
    }
}
