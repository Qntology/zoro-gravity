//! 공문서 HWP 생성 (명세 Part 5). rhwp 로 내부결재용과 민원인 교부용 문서를 함께 만듭니다.
//!
//! 본문과 누출 검사 규칙은 코어의 Gate 4 가 만들고, 이 크레이트는 그 글을 HWP 바이너리로 바꾼 뒤
//! 파일을 다시 읽어 검사합니다. 저장한 글이 입력과 같고, 교부용에 금지어가 없고, 두 문서에 확정 수치가 있어야 합니다.
//! rhwp 의존성이 커서 코어에 넣지 않고 따로 둡니다.

use anyhow::{anyhow, bail, Context, Result};
use rhwp::document_core::DocumentCore;

/// 문서 하나의 제목과 본문. 본문은 줄마다 문단 하나입니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Notice<'a> {
    pub title: &'a str,
    pub body: &'a str,
}

/// 이중 문서의 입력
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DualInput<'a> {
    pub officer: Notice<'a>,
    pub citizen: Notice<'a>,
    /// 교부용에 있으면 안 되는 말 (내부 감사 용어, 제3자 실명)
    pub forbidden: &'a [String],
    /// 두 문서에 그대로 있어야 하는 확정 수치 (차액, 적용 조항)
    pub locked_facts: &'a [String],
}

/// 만든 이중 문서
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DualHwp {
    pub officer: Vec<u8>,
    pub citizen: Vec<u8>,
}

fn rhwp_error(what: &str, e: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("{what}: {e}")
}

/// 제목과 본문으로 HWP 문서를 만듭니다.
pub fn build_hwp(notice: &Notice<'_>) -> Result<Vec<u8>> {
    let mut doc = DocumentCore::new_empty();
    doc.create_blank_document_native().map_err(|e| rhwp_error("빈 문서를 만들 수 없습니다", e))?;
    let lines = std::iter::once(notice.title).chain(notice.body.lines());
    for (i, line) in lines.enumerate() {
        if i > 0 {
            let end = doc.document().sections[0].paragraphs[i - 1].text.chars().count();
            doc.split_paragraph_native(0, i - 1, end, None).map_err(|e| rhwp_error("문단을 나눌 수 없습니다", e))?;
        }
        if !line.is_empty() {
            doc.insert_text_native(0, i, 0, line).map_err(|e| rhwp_error("글을 넣을 수 없습니다", e))?;
        }
    }
    let title_len = notice.title.chars().count();
    doc.apply_para_format_native(0, 0, r#"{"alignment":"center"}"#).map_err(|e| rhwp_error("제목을 가운데로 맞출 수 없습니다", e))?;
    doc.apply_char_format_native(0, 0, 0, title_len, r#"{"bold":true,"fontSize":1600}"#).map_err(|e| rhwp_error("제목 글꼴을 바꿀 수 없습니다", e))?;
    doc.export_hwp_native().map_err(|e| rhwp_error("HWP 로 저장할 수 없습니다", e))
}

/// HWP 파일의 글을 문단마다 한 줄로 읽습니다.
pub fn read_text(hwp: &[u8]) -> Result<String> {
    let doc = DocumentCore::from_bytes(hwp).map_err(|e| rhwp_error("HWP 를 읽을 수 없습니다", e))?;
    let lines: Vec<&str> = doc.document().sections.iter().flat_map(|s| &s.paragraphs).map(|p| p.text.as_str()).collect();
    Ok(lines.join("\n"))
}

/// 편집을 마친 문서에 확정 수치가 그대로 있는지 봅니다 (명세 Case 4-C). 없어진 수치를 돌려줍니다.
pub fn missing_facts(hwp: &[u8], locked_facts: &[String]) -> Result<Vec<String>> {
    let text = read_text(hwp)?;
    Ok(locked_facts.iter().filter(|fact| !text.contains(fact.as_str())).cloned().collect())
}

/// 저장한 파일을 다시 읽어 입력과 같은지 확인합니다.
fn check_saved(hwp: &[u8], notice: &Notice<'_>) -> Result<String> {
    let text = read_text(hwp)?;
    let want = std::iter::once(notice.title).chain(notice.body.lines()).collect::<Vec<_>>().join("\n");
    if text != want {
        bail!("{} 의 저장 내용이 입력과 다릅니다", notice.title);
    }
    Ok(text)
}

/// 내부결재용과 민원인 교부용 문서를 만듭니다. 검사를 통과해야만 돌려줍니다.
pub fn build_dual(input: &DualInput<'_>) -> Result<DualHwp> {
    let officer = build_hwp(&input.officer).context("내부결재용 문서")?;
    let citizen = build_hwp(&input.citizen).context("민원인 교부용 문서")?;
    check_saved(&officer, &input.officer)?;
    let citizen_text = check_saved(&citizen, &input.citizen)?;

    // 내용은 호출자에게 돌려주지 않고 개수만 알립니다. 오류 메시지가 로그에 남기 때문입니다.
    let leaks = input.forbidden.iter().filter(|term| citizen_text.contains(term.as_str())).count();
    if leaks > 0 {
        bail!("민원인 교부용 문서에 금지어 {leaks}개가 있어서 발급을 막았습니다");
    }
    for (name, hwp) in [("내부결재용", &officer), ("민원인 교부용", &citizen)] {
        let missing = missing_facts(hwp, input.locked_facts)?.len();
        if missing > 0 {
            bail!("{name} 문서에 확정 수치 {missing}개가 없어서 발급을 막았습니다");
        }
    }
    Ok(DualHwp { officer, citizen })
}

#[cfg(test)]
mod tests {
    use super::*;

    const OFFICER: Notice<'_> = Notice {
        title: "청년월세 심사 및 감사 보고서",
        body: "1. 대상자: 김철수 (임대인: 박영희, 모친)\n2. 차액: 200,000원\n\n3. 처분의견: 지침 제2조 의거 지급 반려함.",
    };
    const CITIZEN: Notice<'_> = Notice {
        title: "청년월세 한시특별지원 심사 안내문",
        body: "1. 심사결과: 지급 보류 및 소명자료 안내\n2. 사유: 지침 제2조, 월세 차액(200,000원) 증빙 보완 필요\n3. 불복절차: 처분일로부터 90일 이내 이의신청 가능",
    };

    fn strings(items: &[&str]) -> Vec<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn dual<'a>(citizen: Notice<'a>, forbidden: &'a [String], locked_facts: &'a [String]) -> DualInput<'a> {
        DualInput { officer: OFFICER, citizen, forbidden, locked_facts }
    }

    #[test]
    fn a_saved_document_reads_back_the_same_text() -> Result<()> {
        let hwp = build_hwp(&OFFICER)?;
        assert_eq!(&hwp[..4], [0xD0, 0xCF, 0x11, 0xE0], "HWP 5.0 복합 파일이어야 합니다");
        assert_eq!(read_text(&hwp)?, format!("{}\n{}", OFFICER.title, OFFICER.body));
        Ok(())
    }

    #[test]
    fn both_documents_are_built_when_the_checks_pass() -> Result<()> {
        let (forbidden, facts) = (strings(&["LatticeDB", "위장", "96.4%", "박영희"]), strings(&["200,000원", "제2조"]));
        let out = build_dual(&dual(CITIZEN, &forbidden, &facts))?;
        assert!(read_text(&out.officer)?.contains("박영희"), "내부 문서에는 있어도 됩니다");
        assert!(!read_text(&out.citizen)?.contains("박영희"));
        Ok(())
    }

    #[test]
    fn an_internal_term_in_the_citizen_document_blocks_it_without_quoting() {
        let forbidden = strings(&["박영희"]);
        let leaking = Notice { title: "안내문", body: "임대인 박영희 님과의 차액" };
        let err = build_dual(&dual(leaking, &forbidden, &[])).unwrap_err().to_string();
        assert!(err.contains("금지어 1개"), "{err}");
        assert!(!err.contains("박영희"), "오류에 실명이 남으면 안 됩니다");
    }

    #[test]
    fn a_changed_locked_fact_blocks_the_documents() -> Result<()> {
        let facts = strings(&["200,000원"]);
        let damaged = Notice { title: "안내문", body: "월세 차액(20,000원) 증빙 보완 필요" };
        assert!(build_dual(&dual(damaged, &[], &facts)).unwrap_err().to_string().contains("민원인 교부용"));
        // 명세 Case 4-C: 편집기에서 0 을 지운 파일을 찾습니다.
        assert_eq!(missing_facts(&build_hwp(&damaged)?, &facts)?, facts);
        assert!(missing_facts(&build_hwp(&CITIZEN)?, &facts)?.is_empty());
        Ok(())
    }

    #[test]
    fn a_broken_file_is_an_error() {
        assert!(read_text(b"not a hwp file").is_err());
    }
}
