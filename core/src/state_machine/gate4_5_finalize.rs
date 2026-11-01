//! Gate 4: 문서 생성 규칙 · Gate 5: 확정 기록 (명세 Part 5).
//!
//! - 양식 선택: 간단한 서류가 빠졌고 보완 기회가 없었다면 반려 양식을 막고 보완요구로 바꿉니다 (Case 4-A).
//! - 민원인 교부용 문서: 내부 감사 용어와 제3자 실명이 섞이면 발급을 막습니다 (Case 4-B).
//! - 확정 수치: 문서에서 바뀐 값은 확정 값으로 되돌립니다 (Case 4-C).
//!
//! 여기서는 본문 텍스트까지 만듭니다. HWP 바이너리는 `tools/hwp-builder` (rhwp) 가 만듭니다.

use std::collections::BTreeMap;

use chrono::{Months, NaiveDate};

use crate::privacy;
use crate::store::ArchiveRow;
use crate::verdict::Verdict;

/// 보존 기간 (명세: 법정 보존 5년)
pub const RETENTION_YEARS: u32 = 5;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    /// 보완요구 (7일 유예)
    SupplementRequest,
    /// 반려 통지
    RejectNotice,
    /// 승인·지급 통지
    ApprovalNotice,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormChoice {
    pub form: Form,
    /// 요청한 양식이 절차 규칙 때문에 바뀌었는지
    pub overridden: bool,
}

/// 양식 선택. 간단한 서류가 빠졌고 보완 기회가 없었다면 반려 대신 보완요구로 바꿉니다 (명세 Case 4-A).
pub fn select_form(requested: Form, supplement_count: u32, simple_doc_missing: bool) -> FormChoice {
    if simple_doc_missing && supplement_count == 0 && requested == Form::RejectNotice {
        return FormChoice { form: Form::SupplementRequest, overridden: true };
    }
    FormChoice { form: requested, overridden: false }
}

/// 민원인 교부용 문서의 누출 검사 (명세 Case 4-B). 내부 용어나 `extra_terms`(제3자 실명 등)가 있으면 막습니다.
/// 찾은 용어 목록은 감사 로그에 원문으로 남기지 않도록 개수만 씁니다.
pub fn check_citizen_text(text: &str, extra_terms: &[String]) -> Result<(), (Verdict, Vec<String>)> {
    let mut leaks: Vec<String> = privacy::find_leaks(text, privacy::INTERNAL_TERMS)
        .into_iter()
        .map(str::to_string)
        .collect();
    leaks.extend(privacy::find_leaks(text, extra_terms).into_iter().map(str::to_string));
    if leaks.is_empty() {
        Ok(())
    } else {
        Err((Verdict::G4BlockInternalLeak, leaks))
    }
}

/// 확정 수치를 문서 값과 맞춥니다 (명세 Case 4-C). 바뀐 항목은 확정 값으로 되돌리고, 그 이름을 돌려줍니다.
pub fn restore_locked_facts(locked: &BTreeMap<String, String>, doc: &mut BTreeMap<String, String>) -> Vec<String> {
    let mut restored = Vec::new();
    for (key, value) in locked {
        if doc.get(key) != Some(value) {
            doc.insert(key.clone(), value.clone());
            restored.push(key.clone());
        }
    }
    restored
}

/// Gate 5 용 보존 행. 보존 만료일은 기준일로부터 5년입니다.
#[allow(clippy::too_many_arguments)]
pub fn build_archive_row(
    archive_id: &str,
    case_id: &str,
    final_code: &str,
    officer_doc: String,
    citizen_doc: String,
    special_audit_flag: bool,
    today: NaiveDate,
) -> ArchiveRow {
    ArchiveRow {
        archive_id: archive_id.to_string(),
        case_id: case_id.to_string(),
        final_code: final_code.to_string(),
        officer_doc,
        citizen_doc,
        special_audit_flag,
        retention_until: today
            .checked_add_months(Months::new(12 * RETENTION_YEARS))
            .unwrap_or(today),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_is_replaced_by_supplement_when_simple_documents_are_missing() {
        // 명세 Case 4-A: 소방필증 누락, 보완 기회 0회
        let c = select_form(Form::RejectNotice, 0, true);
        assert_eq!(c, FormChoice { form: Form::SupplementRequest, overridden: true });
        // 보완 기회를 이미 준 뒤에는 반려가 가능합니다
        assert_eq!(select_form(Form::RejectNotice, 1, true).form, Form::RejectNotice);
    }

    #[test]
    fn citizen_document_with_internal_terms_is_blocked() {
        // 명세 Case 4-B: 내부 보고서 문구를 붙여넣기
        let pasted = "LatticeDB 추적 결과 모친 박영희와의 위장계약 96.4% 확실";
        let err = check_citizen_text(pasted, &[]).unwrap_err();
        assert_eq!(err.0, Verdict::G4BlockInternalLeak);
        assert!(err.1.len() >= 3);
    }

    #[test]
    fn third_party_real_name_is_blocked_too() {
        let text = "임대인 박영희 님과의 계약 차액을 보완해 주세요";
        assert!(check_citizen_text(text, &["박영희".to_string()]).is_err());
        assert!(check_citizen_text("차액을 보완해 주세요", &["박영희".to_string()]).is_ok());
    }

    #[test]
    fn changed_fact_is_restored_to_the_locked_value() {
        // 명세 Case 4-C: 차액 200,000 이 20,000 으로 훼손됨
        let locked = BTreeMap::from([("diff".to_string(), "200,000".to_string()), ("clause".to_string(), "제2조".to_string())]);
        let mut doc = BTreeMap::from([("diff".to_string(), "20,000".to_string()), ("clause".to_string(), "제2조".to_string())]);
        let restored = restore_locked_facts(&locked, &mut doc);
        assert_eq!(restored, vec!["diff".to_string()]);
        assert_eq!(doc["diff"], "200,000");
    }

    #[test]
    fn retention_is_five_years() {
        let today = NaiveDate::from_ymd_opt(2026, 10, 31).unwrap();
        let row = build_archive_row("A-1", "c", "X", String::new(), String::new(), false, today);
        assert_eq!(row.retention_until, NaiveDate::from_ymd_opt(2031, 10, 31).unwrap());
    }
}
