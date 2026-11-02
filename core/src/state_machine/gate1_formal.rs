//! Gate 1: 형식심사 — 필수 서류, 전자서명, 위임장 유효기간 (명세 Part 2 §2).
//!
//! 이 게이트의 검사는 모두 규칙입니다. 모델 확률로 통과시키지 않습니다.

use std::collections::HashSet;

use chrono::{Days, NaiveDate};

use super::gate0_router::mandatory_docs;
use crate::case::{CaseFile, CaseSource, Category, Signature};
use crate::verdict::Verdict;

/// 전자서명 기준 (명세 Case 1-B): 획수 2회 이상, 서명란 면적 15% 이상, 한 줄 직선이 아님
pub const SIGN_MIN_STROKES: u32 = 2;
pub const SIGN_MIN_AREA: f32 = 0.15;
/// 위임장 유효기간 (명세 Case 1-C): 작성일로부터 180일
pub const POA_MAX_DAYS: i64 = 180;
/// 보완요구 기한 (민원처리법 제22조 조건부 접수 기준)
pub const SUPPLEMENT_DAYS: u64 = 7;

pub fn signature_valid(s: &Signature) -> bool {
    s.stroke_count >= SIGN_MIN_STROKES && s.bbox_area_ratio >= SIGN_MIN_AREA && !s.single_line
}

fn requires_signature(category: Category) -> bool {
    matches!(category, Category::Welfare | Category::Cert)
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FormalReport {
    /// 필요한데 스캔되지 않은 서류
    pub missing: Vec<String>,
    /// 스캔은 됐지만 이 민원에 필요 없는 서류 (예: 건강보험 고지서)
    pub unexpected: Vec<String>,
    /// 같은 내용 해시로 두 번 스캔된 서류 (한 장으로 합칩니다)
    pub duplicates: Vec<String>,
}

impl FormalReport {
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty()
    }
}

pub fn check_documents(case: &CaseFile) -> FormalReport {
    let required = mandatory_docs(case.category);
    let mut seen = HashSet::new();
    let mut duplicates = Vec::new();
    let mut kinds: Vec<&str> = Vec::new();
    for d in &case.docs {
        if !seen.insert(d.content_hash.as_str()) {
            duplicates.push(d.kind.clone());
            continue;
        }
        kinds.push(d.kind.as_str());
    }
    // 대장이나 창구에서 확인한 서류는 스캔된 것과 같이 셉니다.
    let missing = required
        .iter()
        .filter(|r| !kinds.contains(r) && !case.intake.evidence.iter().any(|e| e.kind == **r))
        .map(|r| r.to_string())
        .collect();
    let unexpected = kinds
        .iter()
        .filter(|k| !required.contains(*k))
        .map(|k| k.to_string())
        .collect();
    FormalReport { missing, unexpected, duplicates }
}

/// 위임장 작성일로부터 경과 일수 (명세 예: 2025-12-10 → 2026-10-31 = 325일)
pub fn poa_elapsed_days(written: NaiveDate, today: NaiveDate) -> i64 {
    (today - written).num_days()
}

pub fn poa_expired(written: NaiveDate, today: NaiveDate) -> bool {
    poa_elapsed_days(written, today) > POA_MAX_DAYS
}

/// 보완요구 기한 (발급일 + 7일)
pub fn supplement_deadline(today: NaiveDate) -> NaiveDate {
    today.checked_add_days(Days::new(SUPPLEMENT_DAYS)).unwrap_or(today)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Gate1 {
    /// 형식 요건을 모두 충족
    Pass(FormalReport),
    /// 키오스크에서 서명을 다시 받습니다. 처분이 아닙니다.
    SignatureRetry { verdict: Verdict, reason: String },
    /// 효력이 없음 (위임장 유효기간 도과). 처분 후보이므로 공무원이 확인합니다.
    Reject { verdict: Verdict, reason: String },
    /// 서류가 빠지거나 맞지 않음. 보완요구 또는 재투입을 공무원이 정합니다.
    Hold { report: FormalReport },
}

pub fn evaluate(case: &CaseFile) -> Gate1 {
    // 1) 위임장 유효기간: 효력 상실은 바로 결론입니다 (명세 Case 1-C)
    if case.category == Category::Cert {
        if let Some(written) = case.poa_written_on {
            if poa_expired(written, case.today) {
                return Gate1::Reject {
                    verdict: Verdict::G1ExpiredMandatoryReject,
                    reason: format!(
                        "위임장 작성일 {written} 로부터 {}일 경과 (법정 {}일 초과)",
                        poa_elapsed_days(written, case.today),
                        POA_MAX_DAYS
                    ),
                };
            }
        }
    }

    // 2) 전자서명: 서명란이 필요한 민원은 유효한 서명이 있어야 합니다 (명세 Case 1-B)
    if requires_signature(case.category) && !case.intake.stamp_requested {
        match &case.signature {
            Some(s) if signature_valid(s) => {}
            Some(_) => {
                return Gate1::SignatureRetry {
                    verdict: Verdict::G1SignInvalidRetry,
                    reason: "서명 획수나 면적이 기준에 못 미칩니다. 성함을 다시 적어 주세요".into(),
                };
            }
            None => {
                return Gate1::SignatureRetry {
                    verdict: Verdict::G1SignInvalidRetry,
                    reason: "서명란이 비어 있습니다".into(),
                };
            }
        }
    }

    // 3) 필수 서류. 양식 입력 건은 서류를 창구에서 확인하므로, 빠진 서류는 보류하지 않고 창구 확인 대기로 넘깁니다.
    //    확정 전에 창구 확인이 끝나야 하는 것은 `pipeline::confirm` 이 막습니다.
    let report = check_documents(case);
    if report.is_complete() || case.source == CaseSource::KioskForm {
        Gate1::Pass(report)
    } else {
        Gate1::Hold { report }
    }
}

/// 누락 서류를 둔 채 Gate 2 로 강제 진입하려는 시도를 막습니다 (명세 Case 1-D). 서류가 완전할 때만 통과합니다.
pub fn force_pass(report: &FormalReport) -> Result<(), Verdict> {
    if report.is_complete() {
        Ok(())
    } else {
        Err(Verdict::G1InterceptForceHold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{Person, ScannedDoc};

    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 10, 31).unwrap()
    }

    fn doc(kind: &str, hash: &str) -> ScannedDoc {
        ScannedDoc { kind: kind.into(), content_hash: hash.into(), pages: 1 }
    }

    fn case(category: Category, docs: Vec<ScannedDoc>) -> CaseFile {
        CaseFile {
            case_id: "t".into(),
            category,
            source: Default::default(),
            docs,
            applicant: Person { name: "김철수".into(), birth_prefix: "990101".into(), identifier: "x".into() },
            landlord: None,
            lease: None,
            bank: None,
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

    #[test]
    fn spec_signature_case_is_rejected() {
        // 명세 Case 1-B: 1획, 면적 3.8%, 직선
        let s = Signature { stroke_count: 1, bbox_area_ratio: 0.038, single_line: true };
        assert!(!signature_valid(&s));
        let ok = Signature { stroke_count: 8, bbox_area_ratio: 0.42, single_line: false };
        assert!(signature_valid(&ok));
    }

    #[test]
    fn spec_poa_case_is_expired_for_325_days() {
        // 명세 Case 1-C: 2025-12-10 작성, 2026-10-31 기준 325일
        let written = NaiveDate::from_ymd_opt(2025, 12, 10).unwrap();
        assert_eq!(poa_elapsed_days(written, today()), 325);
        assert!(poa_expired(written, today()));
        let mut c = case(Category::Cert, vec![doc("위임장", "a"), doc("위임자신분증사본", "b"), doc("대리인신분증", "c")]);
        c.poa_written_on = Some(written);
        assert!(matches!(evaluate(&c), Gate1::Reject { verdict: Verdict::G1ExpiredMandatoryReject, .. }));
    }

    #[test]
    fn duplicate_scans_merge_and_wrong_documents_hold_the_case() {
        // 명세 Case 1-A: 신청서 중복 스캔, 이체내역 자리에 건강보험 고지서
        let docs = vec![
            doc("신청서", "h1"),
            doc("신청서", "h1"),
            doc("건강보험고지서", "h9"),
        ];
        let c = case(Category::Welfare, docs);
        let report = check_documents(&c);
        assert_eq!(report.duplicates, vec!["신청서".to_string()]);
        assert!(report.missing.contains(&"임대차계약서".to_string()));
        assert!(report.missing.contains(&"이체내역서".to_string()));
        assert!(report.unexpected.contains(&"건강보험고지서".to_string()));
        assert!(matches!(evaluate(&c), Gate1::Hold { .. }));
    }

    #[test]
    fn force_pass_is_blocked_while_documents_are_missing() {
        let report = FormalReport { missing: vec!["이체내역서".into()], ..Default::default() };
        assert_eq!(force_pass(&report), Err(Verdict::G1InterceptForceHold));
        assert_eq!(force_pass(&FormalReport::default()), Ok(()));
    }

    #[test]
    fn invalid_signature_asks_the_kiosk_to_retry() {
        let mut c = case(Category::Welfare, vec![]);
        c.signature = Some(Signature { stroke_count: 1, bbox_area_ratio: 0.038, single_line: true });
        assert!(matches!(evaluate(&c), Gate1::SignatureRetry { verdict: Verdict::G1SignInvalidRetry, .. }));
    }

    #[test]
    fn supplement_deadline_is_seven_days_out() {
        assert_eq!(supplement_deadline(today()), NaiveDate::from_ymd_opt(2026, 11, 7).unwrap());
    }
}
