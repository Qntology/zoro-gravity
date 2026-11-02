//! 민원 한 건의 입력 모델. 키오스크 입력, 스캐너 결과, 공무원 입력이 이 구조로 모입니다.
//!
//! 주민등록번호 원문은 `Person::identifier` 에만 있습니다. 관계망에는 솔트 해시만 올라갑니다.
//! `CaseFile` 이 버려질 때 원문 문자열은 0 으로 덮어씁니다 (zeroize). `Debug` 출력에는 원문이 나오지 않습니다.

use std::fmt;

use chrono::{DateTime, NaiveDate, Utc};
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

impl Signature {
    /// 종이 서류의 서명. 스캔에서 서명이 보이면 패드 측정값 대신 이 값을 씁니다.
    pub fn on_paper() -> Self {
        Self { stroke_count: 2, bbox_area_ratio: 1.0, single_line: false }
    }
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
    /// 배우자
    Spouse,
    /// 단체의 임원
    OfficerOf,
    /// 거래처의 대표
    RepresentedBy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Relation {
    pub a: Node,
    pub b: Node,
    pub link: Link,
}

/// 사건 입력의 출처. 스캔은 판독 값이고, 양식 입력은 민원인이 적은 신고값입니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaseSource {
    #[default]
    Scanned,
    KioskForm,
}

/// 필수 서류를 어떻게 확인했는지
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    /// 민원인이 낸 서류를 스캔해서 읽음
    Scanned,
    /// 합성 대장과 맞춤 (실제 운영에서는 공공 데이터 연계가 들어갈 자리)
    Registry,
    /// 키오스크 입력과 서명으로 갈음함
    KioskSigned,
    /// 창구에서 공무원이 원본을 확인함
    Counter,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocEvidence {
    pub kind: String,
    pub how: Evidence,
}

/// 본인 확인 수준
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Identity {
    /// 민원인이 적은 값뿐입니다
    #[default]
    Declared,
    /// 신분증을 스캔해서 읽었습니다
    IdCardScanned,
    /// 창구에서 공무원이 신분증을 확인했습니다
    Counter,
}

/// 영수증 한 장 (스캔 또는 합성 카드 원장)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub approval: String,
    pub vendor: String,
    /// YYYY-MM-DD
    pub date: String,
    pub time: String,
    pub amount: i64,
    pub item: String,
}

/// 지원금 정산 사실
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SubsidyFacts {
    pub org: String,
    pub program: String,
    /// 정산을 신청한 금액
    pub claimed_total: i64,
    pub receipts: Vec<Receipt>,
    /// 단체 노드의 해시 (관계망에 있을 때)
    pub org_node: Option<String>,
    /// (거래처 이름, 노드 해시)
    pub vendor_nodes: Vec<(String, String)>,
    /// 비인정 업종으로 등록된 거래처 이름
    pub barred_vendors: Vec<String>,
}

/// 영업신고 사실
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PermitFacts {
    pub business_type: String,
    pub building_use: Option<String>,
    /// 위반건축물 표기 내용
    pub building_violation: Option<String>,
    pub hygiene_expires: Option<NaiveDate>,
    pub fire_expires: Option<NaiveDate>,
}

/// 위임 사실
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CertFacts {
    /// 위임자 (대리인이 신청인입니다)
    pub principal: Option<Person>,
    /// 위임장에 위임자 서명이나 도장이 있는지 (모르면 None)
    pub principal_signed: Option<bool>,
    /// 인감 대장에 위임자가 등록되어 있는지
    pub seal_registered: Option<bool>,
}

/// 키오스크 접수에서 생기는 값. 스캔 사건에는 비어 있습니다.
#[derive(Debug, Clone, Default)]
pub struct IntakeFacts {
    /// 서류별 확인 방법
    pub evidence: Vec<DocEvidence>,
    /// 창구에서 확인할 일. 비어 있지 않으면 자동 확정하지 않습니다.
    pub pending: Vec<String>,
    /// 양식 입력 값을 공무원 화면에 보이는 줄로 바꾼 것 (항목 이름, 값). 종료하면 지웁니다.
    pub form_lines: Vec<(String, String)>,
    pub identity: Identity,
    /// 손떨림 등으로 서명이 어려워 대면 날인을 요청했습니다 (명세 `G1_SIGN_ELDERLY_STAMP`)
    pub stamp_requested: bool,
    /// 창구 확인을 마치고 다시 심사하는 중입니다. 민원인에게 되돌리지 않고 공무원에게 남깁니다.
    pub counter_done: bool,
    /// 처음 접수한 시각 (창구 확인 뒤 다시 심사해도 그대로 둡니다)
    pub received_at: Option<DateTime<Utc>>,
    pub subsidy: Option<SubsidyFacts>,
    pub permit: Option<PermitFacts>,
    pub cert: Option<CertFacts>,
}

pub struct CaseFile {
    pub case_id: String,
    pub category: Category,
    /// 입력 출처. 키오스크 접수 건은 증빙이 모자라면 정정 · 잠금 · 자동 확정을 하지 않습니다.
    pub source: CaseSource,
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
    /// 고충민원의 본문 (OCR 결과 또는 키오스크 입력). 정형 민원에는 없습니다.
    pub grievance_text: Option<String>,
    /// 민원인 연락처 (회신용). 관계망에는 올리지 않습니다.
    pub contact: Option<String>,
    /// 키오스크 접수에서 생기는 값
    pub intake: IntakeFacts,
}

impl Drop for CaseFile {
    fn drop(&mut self) {
        wipe_person(&mut self.applicant);
        if let Some(p) = self.landlord.as_mut() {
            wipe_person(p);
        }
        if let Some(p) = self.intake.cert.as_mut().and_then(|c| c.principal.as_mut()) {
            wipe_person(p);
        }
        if let Some(Excuse::FreeText(text)) = self.excuse.as_mut() {
            text.zeroize();
        }
        if let Some(text) = self.grievance_text.as_mut() {
            text.zeroize();
        }
        if let Some(contact) = self.contact.as_mut() {
            contact.zeroize();
        }
        for (label, value) in self.intake.form_lines.iter_mut() {
            label.zeroize();
            value.zeroize();
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
