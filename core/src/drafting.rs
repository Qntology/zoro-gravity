//! 모델이 쓰는 초안: 고충민원 3줄 요약, 자유 서술 소명의 정규화, 내부 심사 의견.
//!
//! 명세의 `Trida` 생성 모드입니다. 엔진은 [`TextWriter`] 에만 기대므로 실제 모델이든 시험 대역이든 같은 방식으로 꽂힙니다.
//!
//! 초안은 처분이 아닙니다. 글에 나온 숫자는 모두 원문과 대조하고(`factcheck`), 어긋나면 한 번 다시 쓰게 한 뒤에도 어긋나면
//! `verified = false` 로 공무원에게 표시합니다. 검증을 못 넘긴 글은 공문서에 넣지 않습니다.

use anyhow::Result;

use crate::case::Excuse;
use crate::factcheck::{mentions, unsupported};
use crate::privacy::mask_long_numbers;
use crate::verdict::{OutcomeClass, Verdict};

/// 지시문과 입력을 받아 글을 쓰는 모델
pub trait TextWriter: Send + Sync {
    fn complete(&self, system: &str, user: &str, max_tokens: usize) -> Result<String>;
}

/// 고충민원을 배정할 수 있는 부서와 하는 일. 이 목록 밖의 부서는 받아들이지 않습니다.
pub const DEPARTMENTS: [(&str, &str); 11] = [
    ("건설관리과", "공사 현장 관리, 공사 소음과 먼지"),
    ("환경위생과", "악취, 대기 오염, 음식점 위생"),
    ("교통행정과", "신호등, 횡단보도, 주정차 단속"),
    ("도로관리과", "도로 파손, 포장, 보도 보수"),
    ("공원녹지과", "공원, 가로수, 산책로"),
    ("청소행정과", "쓰레기 수거, 불법 투기"),
    ("주택과", "주택 관리, 불법 증축"),
    ("복지정책과", "기초연금, 긴급 복지 지원"),
    ("세무과", "지방세, 세금 고지와 체납"),
    ("안전총괄과", "재난, 붕괴 위험, 시설물 안전 점검"),
    ("민원여권과", "민원 서류 발급, 여권"),
];

/// 증빙이 있다는 말로 볼 수 있는 낱말. 이 말이 없으면 모델이 영수증이 있다고 답해도 믿지 않습니다.
const EVIDENCE_WORDS: [&str; 4] = ["영수증", "증빙", "계산서", "전표"];

const SUMMARY_SYSTEM: &str = "행정기관의 민원 요약 담당입니다. 민원 글을 사실 그대로 요약합니다. 원문에 없는 숫자, 시간, 금액, 날짜를 만들지 않습니다. 시간과 금액은 원문의 표현을 그대로 옮깁니다.";
const SUMMARY_TOKENS: usize = 160;
const EXCUSE_TOKENS: usize = 24;
const OPINION_TOKENS: usize = 120;

/// 고충민원 요약 초안
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrievanceSummary {
    /// 민원 핵심 요지
    pub gist: String,
    /// 민원인 요구 사항
    pub demand: String,
    /// 담당 부서 후보. 목록에 없는 답이면 없습니다.
    pub department: Option<String>,
    /// 숫자 대조와 형식을 모두 통과했는가
    pub verified: bool,
    /// 걸린 내용 (원문에 없는 숫자 등)
    pub problems: Vec<String>,
}

fn line_after(text: &str, label: &str) -> Option<String> {
    text.lines().find_map(|l| {
        let t = l.trim().trim_start_matches(['-', '*', '•', ' ']).replace("**", "");
        let rest = t.strip_prefix(label)?.trim_start_matches([':', '：', ' ']).trim();
        (!rest.is_empty()).then(|| rest.to_string())
    })
}

fn parse_summary(text: &str, source: &str) -> GrievanceSummary {
    let gist = line_after(text, "요지").unwrap_or_default();
    let demand = line_after(text, "요구").unwrap_or_default();
    let named = line_after(text, "부서").unwrap_or_default();
    let department = DEPARTMENTS.iter().find(|(d, _)| named.contains(d)).map(|(d, _)| d.to_string());
    let mut problems = Vec::new();
    if gist.is_empty() {
        problems.push("요지가 없습니다".to_string());
    }
    if demand.is_empty() {
        problems.push("요구 사항이 없습니다".to_string());
    }
    if department.is_none() {
        problems.push("목록에 있는 부서가 아닙니다".to_string());
    }
    for n in unsupported(&format!("{gist}\n{demand}"), source) {
        problems.push(format!("원문에 없는 숫자: {n}"));
    }
    GrievanceSummary { gist, demand, department, verified: problems.is_empty(), problems }
}

/// 고충민원을 세 줄로 요약합니다. 숫자가 어긋나면 원문의 숫자를 알려 주고 한 번 다시 쓰게 합니다.
pub fn summarize_grievance(writer: &dyn TextWriter, text: &str) -> Result<GrievanceSummary> {
    let source = mask_long_numbers(text);
    let departments: Vec<String> = DEPARTMENTS.iter().map(|(d, work)| format!("{d} ({work})")).collect();
    let ask = format!(
        "다음 민원을 아래 형식의 세 줄로 요약하세요.\n요지: 민원의 핵심 내용 한 문장\n요구: 민원인이 요구하는 것 한 문장\n부서: 담당 부서 이름 하나\n\n부서 목록:\n{}\n\n민원:\n{source}",
        departments.join("\n")
    );
    let first = parse_summary(&writer.complete(SUMMARY_SYSTEM, &ask, SUMMARY_TOKENS)?, &source);
    if first.verified {
        return Ok(first);
    }
    let known: Vec<String> = mentions(&source).into_iter().map(|m| m.text).collect();
    let retry = format!("{ask}\n\n주의: 숫자는 원문에 있는 것만 원문 그대로 쓰세요. 원문의 숫자: {}", if known.is_empty() { "없음".to_string() } else { known.join(", ") });
    let second = parse_summary(&writer.complete(SUMMARY_SYSTEM, &retry, SUMMARY_TOKENS)?, &source);
    // 두 번 다 못 넘기면 문제가 적은 쪽을 공무원에게 보입니다.
    Ok(if second.verified || second.problems.len() <= first.problems.len() { second } else { first })
}

/// 자유 서술 소명을 표준 소명으로 바꿉니다. 코드 뜻은 명세의 `CLAIM_*` 와 같습니다.
/// 공무원이나 민원인에게 불리해지는 분류(신청서 오타 주장 잠금 등)는 하지 않습니다. 영수증을 요구하거나 공무원에게 넘기는 길로만 보냅니다.
/// 글에 증빙 얘기가 없으면 모델이 영수증이 있다고 답해도 영수증을 요구하는 쪽으로 보냅니다.
pub fn normalize_excuse(writer: &dyn TextWriter, text: &str) -> Result<Option<Excuse>> {
    const SYSTEM: &str = "민원인의 소명을 표준 코드로 분류합니다. 코드 하나만 답합니다.";
    let ask = format!(
        "코드 목록:\nCLAIM_REPAIR_DEDUCTION_NO_RECEIPT: 수리비를 월세에서 뺐다는 주장, 영수증이 있다는 말이 없음\nCLAIM_REPAIR_DEDUCTION_WITH_RECEIPT: 수리비를 월세에서 뺐고 영수증이 있다는 주장\nCLAIM_CASH_PAID_DIRECTLY: 차액을 현금으로 직접 줬다는 주장\nCLAIM_OTHER: 그 밖의 내용\n\n소명: {}",
        mask_long_numbers(text)
    );
    let answer = writer.complete(SYSTEM, &ask, EXCUSE_TOKENS)?;
    Ok(if answer.contains("CLAIM_REPAIR_DEDUCTION_NO_RECEIPT") {
        Some(Excuse::RepairDeduction { has_receipt: false })
    } else if answer.contains("CLAIM_REPAIR_DEDUCTION_WITH_RECEIPT") {
        Some(Excuse::RepairDeduction { has_receipt: EVIDENCE_WORDS.iter().any(|w| text.contains(w)) })
    } else if answer.contains("CLAIM_CASH_PAID_DIRECTLY") {
        Some(Excuse::CashPaidDirectly)
    } else {
        None
    })
}

/// 처분 코드를 우리말로 풉니다. 모델에 코드를 그대로 주면 의견에 코드가 그대로 나옵니다.
pub fn disposition(code: &str) -> &'static str {
    match Verdict::from_code(code).map(Verdict::class) {
        Some(OutcomeClass::Favorable) => "요건 충족, 지급 또는 승인",
        Some(OutcomeClass::Adverse) => "요건 위반, 지급 또는 승인 불가",
        Some(OutcomeClass::Procedural) => "절차 처리",
        None if code.starts_with("SUPPLEMENT") => "서류 보완 요구",
        None => "담당 공무원 결정",
    }
}

/// `G3_ELIGIBLE_NORMAL` 같은 영문 코드가 글에 있는가
fn has_code(text: &str) -> bool {
    text.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).any(|w| w.contains('_') && w.bytes().any(|b| b.is_ascii_uppercase()))
}

/// 끝맺지 못한 글을 마지막으로 완성된 문장까지만 남깁니다. 숫자 안의 소수점에서는 끊지 않습니다.
pub fn through_last_sentence(text: &str) -> &str {
    let b = text.as_bytes();
    let end = (0..b.len()).rev().find(|&i| matches!(b[i], b'.' | b'!' | b'?' | b'\n') && !(b[i] == b'.' && b.get(i + 1).is_some_and(u8::is_ascii_digit)));
    end.map_or("", |i| text[..=i].trim_end())
}

/// 내부결재 문서의 심사 의견 초안. `facts` 는 식별 정보가 없는 사실 줄입니다.
/// 숫자가 사실 줄과 어긋나거나 영문 코드가 그대로 나오면 넣지 않습니다.
pub fn review_opinion(writer: &dyn TextWriter, facts: &[String]) -> Result<Option<String>> {
    const SYSTEM: &str = "행정기관의 내부결재 문서에 들어갈 심사 의견을 씁니다. 주어진 사실만 씁니다. 사실에 없는 숫자, 법령, 이름을 쓰지 않습니다. 영문 코드를 쓰지 않습니다.";
    let source = facts.join("\n");
    let ask = format!("다음 사실로 심사 의견을 두 문장 이내로 쓰세요.\n\n{source}");
    let text = writer.complete(SYSTEM, &ask, OPINION_TOKENS)?;
    // 문서 쪽에서 머리말을 붙이므로 모델이 쓴 머리말은 뺍니다.
    let text = text.trim().trim_start_matches("심사 의견").trim_start_matches([':', ' ']).to_string();
    Ok((!text.is_empty() && !has_code(&text) && unsupported(&text, &source).is_empty()).then_some(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// 정해 둔 답을 차례로 돌려주는 대역. 받은 입력을 기록합니다.
    struct Script {
        answers: Mutex<Vec<String>>,
        asked: Mutex<Vec<String>>,
    }

    fn script(answers: &[&str]) -> Script {
        Script { answers: Mutex::new(answers.iter().rev().map(|s| s.to_string()).collect()), asked: Mutex::new(Vec::new()) }
    }

    impl TextWriter for Script {
        fn complete(&self, _system: &str, user: &str, _max_tokens: usize) -> Result<String> {
            self.asked.lock().unwrap().push(user.to_string());
            Ok(self.answers.lock().unwrap().pop().unwrap_or_default())
        }
    }

    const NOISE: &str = "매일 아침 일곱 시부터 저녁 여덟 시까지 공사 소음이 이어집니다. 구청에 두 차례 전화했지만 조치가 없습니다. 방음벽을 설치해 주세요.";

    #[test]
    fn a_clean_summary_passes_on_the_first_try() -> Result<()> {
        let w = script(&["요지: 오전 7시부터 저녁 8시까지 공사 소음 피해\n요구: 방음벽 설치\n부서: 건설관리과"]);
        let s = summarize_grievance(&w, NOISE)?;
        assert!(s.verified, "{:?}", s.problems);
        assert_eq!(s.gist, "오전 7시부터 저녁 8시까지 공사 소음 피해");
        assert_eq!(s.demand, "방음벽 설치");
        assert_eq!(s.department.as_deref(), Some("건설관리과"));
        assert_eq!(w.asked.lock().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn an_invented_number_is_rewritten_once_with_the_real_numbers() -> Result<()> {
        let w = script(&["요지: 매일 7시~18시 소음\n요구: 방음벽\n부서: 건설관리과", "요지: 일곱 시부터 저녁 여덟 시까지 소음\n요구: 방음벽\n부서: 건설관리과"]);
        let s = summarize_grievance(&w, NOISE)?;
        assert!(s.verified, "{:?}", s.problems);
        let asked = w.asked.lock().unwrap();
        assert_eq!(asked.len(), 2);
        assert!(asked[1].contains("저녁 여덟"), "다시 쓸 때 원문 숫자를 알려 줍니다: {}", asked[1]);
        Ok(())
    }

    #[test]
    fn a_draft_that_keeps_failing_is_marked_for_the_officer() -> Result<()> {
        let bad = "요지: 매일 7시~18시 소음\n요구: 방음벽\n부서: 건설관리과";
        let w = script(&[bad, bad]);
        let s = summarize_grievance(&w, NOISE)?;
        assert!(!s.verified);
        assert_eq!(s.problems, vec!["원문에 없는 숫자: 18".to_string()]);
        assert_eq!(s.gist, "매일 7시~18시 소음", "공무원이 보도록 글은 남깁니다");
        Ok(())
    }

    #[test]
    fn departments_outside_the_list_are_not_accepted() -> Result<()> {
        let w = script(&["요지: 소음\n요구: 방음벽\n부서: 소음대책본부", "요지: 소음\n요구: 방음벽\n부서: 소음대책본부"]);
        let s = summarize_grievance(&w, NOISE)?;
        assert_eq!(s.department, None);
        assert!(!s.verified);
        Ok(())
    }

    #[test]
    fn a_badly_formatted_answer_fails_verification() -> Result<()> {
        let w = script(&["공사 소음이 심합니다.", "공사 소음이 심합니다."]);
        let s = summarize_grievance(&w, NOISE)?;
        assert!(!s.verified);
        assert!(s.problems.iter().any(|p| p.contains("요지")));
        Ok(())
    }

    #[test]
    fn identifiers_are_masked_before_the_model_sees_them() -> Result<()> {
        let w = script(&["요지: 소음\n요구: 조치\n부서: 건설관리과"]);
        summarize_grievance(&w, "연락처 010-1234-5678, 주민번호 900101-1234567 입니다. 소음이 심합니다.")?;
        let asked = w.asked.lock().unwrap();
        assert!(!asked[0].contains("1234-5678") && !asked[0].contains("1234567"), "{}", asked[0]);
        Ok(())
    }

    #[test]
    fn a_receipt_the_excuse_never_mentions_does_not_count() -> Result<()> {
        let n = |text: &str| normalize_excuse(&script(&["CLAIM_REPAIR_DEDUCTION_WITH_RECEIPT"]), text);
        assert_eq!(n("보일러 수리비 20만원을 월세에서 깠어요")?, Some(Excuse::RepairDeduction { has_receipt: false }), "말하지 않은 영수증을 만들면 안 됩니다");
        assert_eq!(n("수리비를 뺐고 영수증도 있습니다")?, Some(Excuse::RepairDeduction { has_receipt: true }));
        Ok(())
    }

    #[test]
    fn free_text_excuses_map_to_standard_claims() -> Result<()> {
        let n = |answer: &str| normalize_excuse(&script(&[answer]), "보일러 수리비 20만원을 월세에서 깠어요");
        assert_eq!(n("CLAIM_REPAIR_DEDUCTION_NO_RECEIPT")?, Some(Excuse::RepairDeduction { has_receipt: false }));
        assert_eq!(n("CLAIM_CASH_PAID_DIRECTLY")?, Some(Excuse::CashPaidDirectly));
        assert_eq!(n("CLAIM_OTHER")?, None);
        assert_eq!(n("잘 모르겠습니다")?, None);
        Ok(())
    }

    #[test]
    fn dispositions_are_spelled_out_for_the_model() {
        assert_eq!(disposition("G3_ELIGIBLE_NORMAL"), "요건 충족, 지급 또는 승인");
        assert_eq!(disposition("G3_VIOLATION_HIGH_RISK"), "요건 위반, 지급 또는 승인 불가");
        assert_eq!(disposition("SUPPLEMENT_7DAYS"), "서류 보완 요구");
        assert!(!disposition("모르는 값").contains('_'));
    }

    #[test]
    fn an_opinion_that_copies_a_code_is_left_out() -> Result<()> {
        let facts = vec!["처분: 요건 충족, 지급 또는 승인".to_string()];
        assert_eq!(review_opinion(&script(&["G3_ELIGIBLE_NORMAL 에 해당합니다."]), &facts)?, None);
        assert!(review_opinion(&script(&["요건을 충족하여 지급합니다."]), &facts)?.is_some());
        Ok(())
    }

    #[test]
    fn a_heading_the_model_adds_is_dropped() -> Result<()> {
        let facts = vec!["처분: 서류 보완 요구".to_string()];
        assert_eq!(review_opinion(&script(&["심사 의견: 서류 보완을 요청합니다."]), &facts)?.as_deref(), Some("서류 보완을 요청합니다."));
        Ok(())
    }

    #[test]
    fn a_cut_off_text_keeps_only_finished_sentences() {
        assert_eq!(through_last_sentence("요건을 충족합니다. 계약 월세 500,000"), "요건을 충족합니다.");
        assert_eq!(through_last_sentence("비율은 3.5"), "");
        assert_eq!(through_last_sentence("요지: 소음\n요구: 방음"), "요지: 소음");
        assert_eq!(through_last_sentence("완성되지 못한 글"), "");
        assert_eq!(through_last_sentence(""), "");
    }

    #[test]
    fn an_opinion_with_unsupported_numbers_is_left_out() -> Result<()> {
        let facts = vec!["월세 500,000원".to_string(), "이체액 500,000원".to_string()];
        assert_eq!(review_opinion(&script(&["월세 500,000원이 이체되어 적격입니다."]), &facts)?.as_deref(), Some("월세 500,000원이 이체되어 적격입니다."));
        assert_eq!(review_opinion(&script(&["월세 300,000원이 이체되었습니다."]), &facts)?, None);
        assert_eq!(review_opinion(&script(&["   "]), &facts)?, None);
        Ok(())
    }
}
