//! 합성 사례로 이중 문서를 만듭니다. 만든 파일을 한글에서 열어 보는 용도입니다.
//!
//!   cargo run -p zorogravity-hwp --example sample -- out/hwp

use std::path::PathBuf;

use anyhow::Result;
use zorogravity_hwp::{build_dual, DualInput, Notice};

fn main() -> Result<()> {
    let dir = PathBuf::from(std::env::args().nth(1).unwrap_or_else(|| "out/hwp".into()));
    std::fs::create_dir_all(&dir)?;
    let forbidden = ["LatticeDB", "위장", "96.4%", "홍길순"].map(String::from);
    let locked_facts = ["200,000원", "제2조"].map(String::from);
    let out = build_dual(&DualInput {
        officer: Notice {
            title: "청년월세 심사 및 감사 보고서",
            body: "1. 대상자: 홍길동 (임대인: 홍길순, 모친)\n2. 적발경위: 월세 차액 200,000원을 현금 지급으로 주장하나, LatticeDB 조회 결과 임대인이 모친으로 확인됨.\n3. 처분의견: 지침 제2조 의거 지급 반려함.",
        },
        citizen: Notice {
            title: "청년월세 한시특별지원 심사 안내문",
            body: "1. 심사결과: 지급 보류 및 소명자료 안내\n2. 안내사유 (관련근거: 사업지침 제2조): 임대인과의 가족관계 확인 서류 및 월세 차액(200,000원) 금융 증빙 보완이 필요합니다.\n3. 불복절차: 처분일로부터 90일 이내 행정심판 또는 이의신청 가능",
        },
        forbidden: &forbidden,
        locked_facts: &locked_facts,
    })?;
    std::fs::write(dir.join("내부결재용.hwp"), &out.officer)?;
    std::fs::write(dir.join("민원인교부용.hwp"), &out.citizen)?;
    println!("{}", dir.display());
    Ok(())
}
