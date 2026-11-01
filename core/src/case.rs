//! 민원 한 건의 입력 모델. 키오스크 입력, 스캐너 결과, 공무원 입력이 이 구조로 모입니다.
//!
//! 주민등록번호 원문은 `Person::identifier` 에만 있습니다. 관계망에는 솔트 해시만 올라갑니다.
//! `CaseFile` 이 버려질 때 원문 문자열은 0 으로 덮어씁니다 (zeroize). `Debug` 출력에는 원문이 나오지 않습니다.

use std::fmt;

use chrono::NaiveDate;
use zeroize::Zeroize;

/// 5대 법정 민원 분류 (명세 Gate 0)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    Welfare,
    Subsidy,
    Permit,
    Cert,
    Grievance,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannedDoc {
    /// 서류 종류 (스캐너 OCR 이 분류한 이름, 예: "임대차계약서")
    pub kind: String,
    /// 스캔 이미지 내용의 해시. 같은 해시가 두 번 들어오면 중복 스캔입니다.
    pub content_hash: String,
    pub pages: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub stroke_count: u32,
    /// 서명 바운딩 박스가 서명란을 차지하는 비율 (0.0 ~ 1.0)
    pub bbox_area_ratio: f32,
    /// 한 줄 직선(`ㅡ`)처럼 생긴 획인지
    pub single_line: bool,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Person {
    pub name: String,
    pub birth_prefix: String,
    /// 주민등록번호 원문 (세션 안에서만 쓰고, 관계망에는 해시만 올립니다)
    pub identifier: String,
}

impl fmt::Debug for Person {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Person")
            .field("name", &crate::privacy::mask_name(&self.name))
            .field("birth_prefix", &self.birth_prefix)
            .field("identifier", &"***")
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeaseFacts {
    /// 신청서에 적힌 월세 (민원인 손글씨)
    pub applied_rent: i64,
    /// 인쇄된 임대차계약서의 월세 (공인중개사 날인 문서, 2순위 진실)
    pub contract_rent: i64,
    /// 은행 이체내역서의 최근 이체액 (1순위 진실)
    pub latest_transfer: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankFacts {
    /// 신청서 손글씨 계좌번호
    pub handwritten_account: String,
    /// 인쇄된 통장 사본의 계좌번호 (1순위 진실)
    pub printed_account: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Excuse {
    /// 신청서 금액이 잘못 적었다는 소명
    TypoInApplication,
    /// 차액을 현금으로 직접 드렸다는 소명
    CashPaidDirectly,
    /// 수리비를 월세에서 공제했다는 소명
    RepairDeduction { has_receipt: bool },
    /// 자유 서술 소명 (정규화는 Trida 생성 모드가 맡습니다)
    FreeText(String),
}

/// 관계망 노드. `hash` 는 솔트 해시이고, `name` 은 동명이인 판별에만 씁니다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    pub hash: String,
    pub name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Link {
    /// 혈연. `degree` 는 촌수입니다 (1 = 부모·자녀, 2 = 형제자매·조부모).
    Family { degree: u8 },
    /// 동일 세대 (해당 연도 기준)
    Household { year: i32 },
    /// 같은 계좌를 함께 씀
    SharedAccount,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    pub a: Node,
    pub b: Node,
    pub link: Link,
}

pub struct CaseFile {
    pub case_id: String,
    pub category: Category,
    pub docs: Vec<ScannedDoc>,
    pub applicant: Person,
    pub landlord: Option<Person>,
    pub lease: Option<LeaseFacts>,
    pub bank: Option<BankFacts>,
    pub signature: Option<Signature>,
    /// 위임장 작성일 (인감 대리발급)
    pub poa_written_on: Option<NaiveDate>,
    /// 심사 기준일. 테스트와 재현을 위해 외부에서 넣습니다.
    pub today: NaiveDate,
    pub excuse: Option<Excuse>,
    pub relations: Vec<Relation>,
    pub supplement_count: u32,
    /// 고충민원의 본문 (OCR 결과). 정형 민원에는 없습니다.
    pub grievance_text: Option<String>,
}

impl Drop for CaseFile {
    fn drop(&mut self) {
        wipe_person(&mut self.applicant);
        if let Some(p) = self.landlord.as_mut() {
            wipe_person(p);
        }
        if let Some(Excuse::FreeText(text)) = self.excuse.as_mut() {
            text.zeroize();
        }
        if let Some(text) = self.grievance_text.as_mut() {
            text.zeroize();
        }
        for r in self.relations.iter_mut() {
            r.a.name.zeroize();
            r.b.name.zeroize();
        }
    }
}

fn wipe_person(p: &mut Person) {
    p.name.zeroize();
    p.identifier.zeroize();
}
