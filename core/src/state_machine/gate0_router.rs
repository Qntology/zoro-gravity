//! Gate 0: 5대 법정 민원 분류와 필수 서류 스키마 로드 (명세 Part 2 §1).
//!
//! 접수 양식의 분류 코드가 규칙 판정입니다. Zoro-1 이 있으면 `CAT_*` 선택지 중 하나를 골라 의견을 붙입니다.
//! 민원 분류는 공무원이 바꿀 수 없는 입력이므로, 모델이 다르게 답하면 확신도가 없어져 자동 확정이 막힙니다.

use anyhow::Result;

use crate::case::{CaseFile, Category};
use crate::decision::{DecisionModel, DecisionTask, Fact};
use crate::verdict::{Judgement, Verdict, judge};

/// 5대 민원 분류 선택지 (명세 `Options`)
pub const CATEGORY_OPTIONS: [Verdict; 5] = [
    Verdict::Cat01Welfare,
    Verdict::Cat02Subsidy,
    Verdict::Cat03Permit,
    Verdict::Cat04Cert,
    Verdict::Cat05Grievance,
];

pub fn verdict_of(category: Category) -> Verdict {
    match category {
        Category::Welfare => Verdict::Cat01Welfare,
        Category::Subsidy => Verdict::Cat02Subsidy,
        Category::Permit => Verdict::Cat03Permit,
        Category::Cert => Verdict::Cat04Cert,
        Category::Grievance => Verdict::Cat05Grievance,
    }
}

/// 분류별 필수 서류 (명세 Part 2 §1.1). 서류 이름은 스캐너가 붙인 `ScannedDoc::kind` 와 같아야 합니다.
pub fn mandatory_docs(category: Category) -> &'static [&'static str] {
    match category {
        Category::Welfare => &["신청서", "임대차계약서", "이체내역서", "금융정보동의서"],
        Category::Subsidy => &["정산총괄표", "영수증철", "세금계산서", "통장사본"],
        Category::Permit => &["영업신고서", "위생교육필증", "소방안전필증", "건축물대장"],
        Category::Cert => &["위임장", "위임자신분증사본", "대리인신분증"],
        Category::Grievance => &[],
    }
}

/// 민원 종류 판정. 규칙(접수 양식)이 정하고, 모델은 스캔된 서류 종류만 보고 독립적으로 의견을 냅니다.
/// 접수 양식의 분류 코드는 모델에 알려 주지 않습니다 (알려 주면 답을 베끼게 됩니다).
pub fn route(case: &CaseFile, model: Option<&dyn DecisionModel>) -> Result<Judgement> {
    let rule = verdict_of(case.category);
    let docs: Vec<&str> = case.docs.iter().map(|d| d.kind.as_str()).collect();
    let task = DecisionTask {
        title: "민원 종류 분류".into(),
        facts: vec![Fact::new("스캔된 서류", docs.join(", "))],
        options: CATEGORY_OPTIONS.to_vec(),
    };
    judge(rule, &task, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_category_maps_to_its_code_and_schema() {
        assert_eq!(verdict_of(Category::Welfare), Verdict::Cat01Welfare);
        assert_eq!(mandatory_docs(Category::Welfare).len(), 4);
        assert!(mandatory_docs(Category::Grievance).is_empty());
        assert!(mandatory_docs(Category::Cert).contains(&"위임장"));
    }
}
