//! Gate 0: 5대 법정 민원 분류와 필수 서류 스키마 로드 (명세 Part 2 §1).
//!
//! 접수 양식의 분류 코드가 민원 종류입니다. 모델에는 묻지 않습니다 (`docs/automation-mode.md` 5절).

use crate::case::Category;
use crate::verdict::Verdict;

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
