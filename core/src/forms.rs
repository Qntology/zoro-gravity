//! 키오스크 입력을 검사해 사건의 뼈대를 만듭니다. 서류와 대장에서 오는 사실은 `evidence.rs` 가 채웁니다.
//!
//! 민원인이 직접 적는 값은 신청 금액 · 계좌처럼 본인이 정할 값만입니다. 계약 월세 · 이체액 · 필증 같은 값은
//! 서류 스캔이나 합성 대장에서 옵니다. 그래서 서류가 없어도 입력 칸이 늘지 않습니다.

use chrono::NaiveDate;
use serde_json::Value;

use crate::case::{BankFacts, CaseFile, CaseSource, Category, CertFacts, Excuse, Identity, IntakeFacts, LeaseFacts, Person, PermitFacts, Signature, SubsidyFacts};
use crate::intake::{birth_date, FormError, GrievanceInput};

/// 신분증 스캔의 서류 이름. 민원별 필수 서류가 아니라 본인 확인에 씁니다.
pub const ID_CARD: &str = "신분증";

/// 스캔해서 읽은 서류 한 건
#[derive(Debug, Clone, PartialEq)]
pub struct ScanDoc {
    pub kind: String,
    pub hash: String,
    pub pages: u32,
    /// 서명이나 도장이 보이는지
    pub signed: bool,
    /// 읽은 값
    pub fields: Value,
    /// 신뢰도가 낮아 사람이 확인해야 하는 항목 이름
    pub low: Vec<String>,
}

impl ScanDoc {
    pub fn text(&self, key: &str) -> Option<&str> {
        text(&self.fields, key)
    }

    pub fn number(&self, key: &str) -> Option<i64> {
        self.fields.get(key).and_then(parse_number)
    }
}

/// 입력 검사를 통과한 사건과 함께 넘길 스캔 결과
pub struct Draft {
    pub case: CaseFile,
    pub scans: Vec<ScanDoc>,
    /// 민원인이 스캔 값을 고친 항목
    pub edited: Vec<String>,
}

fn invalid(msg: impl Into<String>) -> FormError {
    FormError::Invalid(msg.into())
}

fn text<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key)?.as_str().map(str::trim).filter(|s| !s.is_empty())
}

fn required_text(v: &Value, key: &str, label: &str, max: usize) -> Result<String, FormError> {
    let s = text(v, key).ok_or_else(|| invalid(format!("{label}: 값을 적어 주세요.")))?;
    if s.chars().count() > max {
        return Err(invalid(format!("{label}: {max}자 이내로 적어 주세요.")));
    }
    Ok(s.to_string())
}

/// 숫자나 `500,000` 같은 글자를 정수로 읽습니다. 읽을 수 없으면 None 입니다.
pub fn parse_number(v: &Value) -> Option<i64> {
    if let Some(n) = v.as_i64() {
        return Some(n);
    }
    let raw: String = v.as_str()?.chars().filter(|c| !matches!(c, ',' | ' ' | '원')).collect();
    if raw.is_empty() || raw.len() > 12 || !raw.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    raw.parse().ok()
}

/// 금액이나 면적. `positive` 이면 0 을 허용하지 않습니다.
fn number(v: &Value, key: &str, label: &str, positive: bool) -> Result<i64, FormError> {
    let n = v.get(key).and_then(parse_number).ok_or_else(|| invalid(format!("{label}: 숫자로 적어 주세요.")))?;
    if positive && n <= 0 {
        return Err(invalid(format!("{label}: 0보다 큰 숫자로 적어 주세요.")));
    }
    Ok(n)
}

fn account(v: &Value, key: &str, label: &str) -> Result<String, FormError> {
    let s = required_text(v, key, label, 30)?;
    if s.len() < 6 || !s.chars().all(|c| c.is_ascii_digit() || c == '-') {
        return Err(invalid(format!("{label}: 숫자와 하이픈만 적어 주세요.")));
    }
    Ok(s)
}

fn date(v: &Value, key: &str, label: &str, today: NaiveDate) -> Result<NaiveDate, FormError> {
    let s = text(v, key).ok_or_else(|| invalid(format!("{label}: 날짜를 골라 주세요.")))?;
    let d = NaiveDate::parse_from_str(s, "%Y-%m-%d").map_err(|_| invalid(format!("{label}: 날짜를 확인해 주세요.")))?;
    if d > today {
        return Err(invalid(format!("{label}: 오늘 이후 날짜는 쓸 수 없습니다.")));
    }
    Ok(d)
}

/// 고른 값이 목록에 있는지 봅니다. 목록에 없는 값은 받지 않습니다.
fn choice(v: &Value, key: &str, label: &str, allowed: &[&str]) -> Result<String, FormError> {
    let s = text(v, key).ok_or_else(|| invalid(format!("{label}: 하나를 골라 주세요.")))?;
    if !allowed.contains(&s) {
        return Err(invalid(format!("{label}: 목록에서 다시 골라 주세요.")));
    }
    Ok(s.to_string())
}

/// 신청인 · 위임자 · 대리인 한 사람. 성함과 생년월일 앞 6자리를 받습니다.
fn person(v: &Value, prefix: &str, label: &str, today: NaiveDate) -> Result<Person, FormError> {
    let name = required_text(v, &format!("{prefix}_name"), &format!("{label} 성함"), 20)?;
    if name.chars().any(|c| c.is_ascii_digit()) {
        return Err(invalid(format!("{label} 성함: 숫자를 쓸 수 없습니다.")));
    }
    let birth = text(v, &format!("{prefix}_birth")).unwrap_or("").to_string();
    if birth_date(&birth, today).is_none() {
        return Err(invalid(format!("{label} 생년월일 앞 6자리를 확인해 주세요. 예: 990101")));
    }
    Ok(Person { name, birth_prefix: birth, identifier: String::new() })
}

fn flag(v: &Value, key: &str) -> bool {
    v.get(key).and_then(Value::as_bool) == Some(true)
}

/// 서명 측정값. 없으면 `None` 이고, 서명 기준 검사(Gate 1)가 다시 적으라고 돌려보냅니다.
fn signature(v: &Value) -> Option<Signature> {
    let s = v.get("signature")?;
    Some(Signature {
        stroke_count: s.get("stroke_count")?.as_u64()? as u32,
        bbox_area_ratio: s.get("bbox_area_ratio")?.as_f64()? as f32,
        single_line: s.get("single_line")?.as_bool()?,
    })
}

fn excuse(v: &Value) -> Result<Option<Excuse>, FormError> {
    let key = text(v, "excuse").unwrap_or("none");
    Ok(match key {
        "none" => None,
        "typo" => Some(Excuse::TypoInApplication),
        "cash" => Some(Excuse::CashPaidDirectly),
        "repair_receipt" => Some(Excuse::RepairDeduction { has_receipt: true }),
        "repair_no_receipt" => Some(Excuse::RepairDeduction { has_receipt: false }),
        "free_text" => Some(Excuse::FreeText(required_text(v, "excuse_text", "소명 내용", 300)?)),
        _ => return Err(invalid("소명: 목록에서 다시 골라 주세요.")),
    })
}

fn excuse_label(e: &Option<Excuse>) -> String {
    match e {
        None => "없음".into(),
        Some(Excuse::TypoInApplication) => "신청서 금액 오기입".into(),
        Some(Excuse::CashPaidDirectly) => "차액을 현금으로 직접 지급".into(),
        Some(Excuse::RepairDeduction { has_receipt: true }) => "수리비를 월세에서 공제 (영수증 있음)".into(),
        Some(Excuse::RepairDeduction { has_receipt: false }) => "수리비를 월세에서 공제 (영수증 없음)".into(),
        Some(Excuse::FreeText(text)) => format!("직접 적은 소명: {text}"),
    }
}

fn scans(v: &Value) -> Vec<ScanDoc> {
    let Some(list) = v.get("scanned").and_then(Value::as_array) else {
        return Vec::new();
    };
    list.iter()
        .filter_map(|d| {
            Some(ScanDoc {
                kind: text(d, "kind")?.to_string(),
                hash: text(d, "hash")?.to_string(),
                pages: d.get("pages").and_then(Value::as_u64).unwrap_or(1) as u32,
                signed: flag(d, "signed"),
                fields: d.get("fields").cloned().unwrap_or(Value::Null),
                low: d.get("low").and_then(Value::as_array).map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect()).unwrap_or_default(),
            })
        })
        .collect()
}

/// 민원 종류 이름 (화면과 감사 기록에 씁니다)
pub fn label(category: Category) -> &'static str {
    match category {
        Category::Welfare => "복지 지원",
        Category::Subsidy => "지원금 정산",
        Category::Permit => "허가 · 신고",
        Category::Cert => "증명 · 대리",
        Category::Grievance => "고충민원",
    }
}

/// 민원 종류에 맞는 접수 번호 앞 글자
pub fn prefix(category: Category) -> char {
    match category {
        Category::Welfare => 'W',
        Category::Subsidy => 'S',
        Category::Permit => 'P',
        Category::Cert => 'C',
        Category::Grievance => 'G',
    }
}

/// 양식 값을 검사하고 사건 뼈대를 만듭니다. 접수 번호(`case_id`)는 호출하는 쪽이 넣습니다.
pub fn build(category: Category, v: &Value, today: NaiveDate) -> Result<Draft, FormError> {
    let scans = scans(v);
    let edited = v.get("edited").and_then(Value::as_array).map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect()).unwrap_or_default();
    let mut case = match category {
        Category::Welfare => welfare(v, &scans, today),
        Category::Subsidy => subsidy(v, today),
        Category::Permit => permit(v, today),
        Category::Cert => cert(v, today),
        Category::Grievance => grievance(v, today),
    }?;
    case.intake.stamp_requested = flag(v, "stamp_requested");
    Ok(Draft { case: finish_identity(case, &scans), scans, edited })
}

/// 신분증 스캔이 신청인과 맞으면 본인 확인 수준을 올립니다. 다르면 창구 확인으로 넘깁니다.
fn finish_identity(mut case: CaseFile, scans: &[ScanDoc]) -> CaseFile {
    let cards: Vec<&ScanDoc> = scans.iter().filter(|s| s.kind == ID_CARD).collect();
    if cards.is_empty() {
        return case;
    }
    let matches_applicant = cards
        .iter()
        .any(|c| c.text("name") == Some(case.applicant.name.as_str()) && c.text("birth") == Some(case.applicant.birth_prefix.as_str()) && c.low.is_empty());
    if matches_applicant {
        case.intake.identity = Identity::IdCardScanned;
    } else {
        case.intake.pending.push("신분증과 입력한 신청인 정보가 다릅니다".to_string());
    }
    case
}

fn base(category: Category, applicant: Person, today: NaiveDate, form_lines: Vec<(String, String)>) -> CaseFile {
    CaseFile {
        case_id: String::new(),
        category,
        source: CaseSource::KioskForm,
        docs: Vec::new(),
        applicant,
        landlord: None,
        lease: None,
        bank: None,
        signature: None,
        poa_written_on: None,
        today,
        excuse: None,
        relations: Vec::new(),
        supplement_count: 0,
        grievance_text: None,
        contact: None,
        intake: IntakeFacts { form_lines, ..IntakeFacts::default() },
    }
}

/// 민원인 줄: 이름과 생년월일을 한 줄로 보입니다.
fn who(label: &str, p: &Person) -> (String, String) {
    (label.to_string(), format!("{} ({})", p.name, p.birth_prefix))
}

fn welfare(v: &Value, scans: &[ScanDoc], today: NaiveDate) -> Result<CaseFile, FormError> {
    let applicant = person(v, "applicant", "신청인", today)?;
    let applied = number(v, "applied_rent", "신청 월세", true)?;
    let handwritten = account(v, "handwritten_account", "입금받을 계좌")?;
    let excuse = excuse(v)?;
    let paper_consent = scans.iter().any(|s| s.kind == "금융정보동의서" && s.signed);
    if !flag(v, "consent") && !paper_consent {
        return Err(invalid("금융정보 조회에 동의해 주세요."));
    }
    let lines = vec![
        who("신청인", &applicant),
        ("신청 월세".to_string(), format!("{applied}원")),
        ("입금받을 계좌".to_string(), handwritten.clone()),
        ("소명".to_string(), excuse_label(&excuse)),
    ];
    let mut c = base(Category::Welfare, applicant, today, lines);
    c.bank = Some(BankFacts { handwritten_account: handwritten, printed_account: String::new() });
    c.lease = Some(LeaseFacts { applied_rent: applied, contract_rent: 0, latest_transfer: 0 });
    c.signature = signature(v).or(paper_consent.then(Signature::on_paper));
    c.excuse = excuse;
    Ok(c)
}

fn subsidy(v: &Value, today: NaiveDate) -> Result<CaseFile, FormError> {
    let applicant = person(v, "applicant", "신청인", today)?;
    let org = required_text(v, "org_name", "단체명", 40)?;
    let program = required_text(v, "program", "사업명", 40)?;
    let claimed = number(v, "claimed_total", "정산 금액", true)?;
    let lines = vec![
        who("신청인", &applicant),
        ("단체명".to_string(), org.clone()),
        ("사업명".to_string(), program.clone()),
        ("정산 신청 금액".to_string(), format!("{claimed}원")),
    ];
    let mut c = base(Category::Subsidy, applicant, today, lines);
    c.intake.subsidy = Some(SubsidyFacts { org, program, claimed_total: claimed, ..SubsidyFacts::default() });
    Ok(c)
}

fn permit(v: &Value, today: NaiveDate) -> Result<CaseFile, FormError> {
    let applicant = person(v, "applicant", "신고인", today)?;
    let business = required_text(v, "business_name", "상호", 40)?;
    let kind = choice(v, "business_type", "업종", &["일반음식점", "휴게음식점", "제과점", "기타"])?;
    let address = required_text(v, "address", "영업장 주소", 80)?;
    let area = number(v, "area_m2", "면적 (㎡)", true)?;
    let lines = vec![
        who("신고인", &applicant),
        ("상호".to_string(), business),
        ("업종".to_string(), kind.clone()),
        ("영업장 주소".to_string(), address),
        ("면적".to_string(), format!("{area}㎡")),
    ];
    let mut c = base(Category::Permit, applicant, today, lines);
    c.intake.permit = Some(PermitFacts { business_type: kind, ..PermitFacts::default() });
    Ok(c)
}

fn cert(v: &Value, today: NaiveDate) -> Result<CaseFile, FormError> {
    let principal = person(v, "principal", "위임자", today)?;
    let agent = person(v, "applicant", "대리인", today)?;
    let relation = choice(v, "relation", "위임자와의 관계", &["배우자", "자녀", "부모", "형제자매", "기타"])?;
    let document = choice(v, "document", "발급 종류", &["인감증명서", "기타 증명서"])?;
    let purpose = required_text(v, "purpose", "용도", 40)?;
    let poa = date(v, "poa_date", "위임장 작성일", today)?;
    let lines = vec![
        who("위임자", &principal),
        who("대리인", &agent),
        ("관계".to_string(), relation),
        ("발급 종류".to_string(), document),
        ("용도".to_string(), purpose),
        ("위임장 작성일".to_string(), poa.to_string()),
    ];
    // 대리인이 창구에 직접 오므로 대리인을 신청인으로 봅니다.
    let mut c = base(Category::Cert, agent, today, lines);
    c.poa_written_on = Some(poa);
    c.signature = signature(v);
    c.intake.cert = Some(CertFacts { principal: Some(principal), ..CertFacts::default() });
    Ok(c)
}

fn grievance(v: &Value, today: NaiveDate) -> Result<CaseFile, FormError> {
    let input = GrievanceInput {
        name: text(v, "applicant_name").unwrap_or("").to_string(),
        birth_prefix: text(v, "applicant_birth").unwrap_or("").to_string(),
        body: v.get("body").and_then(Value::as_str).unwrap_or("").to_string(),
        contact: text(v, "contact").map(str::to_string),
    };
    crate::intake::grievance_case(input, "", today)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, 2).unwrap()
    }

    fn sig() -> Value {
        json!({ "stroke_count": 8, "bbox_area_ratio": 0.42, "single_line": false })
    }

    fn welfare_values() -> Value {
        json!({
            "applicant_name": "이서연", "applicant_birth": "000315",
            "applied_rent": "400,000", "handwritten_account": "110-123-100315",
            "excuse": "none", "consent": true, "signature": sig()
        })
    }

    #[test]
    fn welfare_needs_only_what_the_applicant_decides() {
        let d = build(Category::Welfare, &welfare_values(), today()).unwrap();
        assert_eq!(d.case.source, CaseSource::KioskForm);
        assert_eq!(d.case.lease.as_ref().unwrap().applied_rent, 400_000, "쉼표는 무시합니다");
        assert_eq!(d.case.intake.identity, Identity::Declared, "신분증이 없으면 본인 확인이 끝나지 않습니다");
        assert!(d.case.intake.form_lines.iter().any(|(k, v)| k == "소명" && v == "없음"));
    }

    #[test]
    fn welfare_requires_consent_and_rejects_bad_numbers_and_accounts() {
        let mut v = welfare_values();
        v["consent"] = json!(false);
        assert!(matches!(build(Category::Welfare, &v, today()), Err(FormError::Invalid(m)) if m.contains("동의")));
        let mut v = welfare_values();
        v["applied_rent"] = json!("0");
        assert!(build(Category::Welfare, &v, today()).is_err());
        let mut v = welfare_values();
        v["handwritten_account"] = json!("abc");
        assert!(build(Category::Welfare, &v, today()).is_err());
    }

    #[test]
    fn a_signed_paper_consent_replaces_the_kiosk_consent_and_signature() {
        let mut v = welfare_values();
        v["consent"] = json!(false);
        v.as_object_mut().unwrap().remove("signature");
        v["scanned"] = json!([{ "kind": "금융정보동의서", "hash": "h1", "signed": true, "fields": {} }]);
        let d = build(Category::Welfare, &v, today()).unwrap();
        assert!(d.case.signature.is_some());
    }

    #[test]
    fn excuses_map_to_the_rule_codes() {
        let mut v = welfare_values();
        v["excuse"] = json!("repair_receipt");
        assert_eq!(build(Category::Welfare, &v, today()).unwrap().case.excuse, Some(Excuse::RepairDeduction { has_receipt: true }));
        v["excuse"] = json!("free_text");
        assert!(build(Category::Welfare, &v, today()).is_err(), "자유 서술 내용이 없으면 안 됩니다");
        v["excuse_text"] = json!("이사 비용을 줄이려고 현금으로 드렸습니다");
        assert!(matches!(build(Category::Welfare, &v, today()).unwrap().case.excuse, Some(Excuse::FreeText(_))));
    }

    #[test]
    fn a_missing_signature_is_left_for_gate_one_to_ask_again() {
        let mut v = welfare_values();
        v.as_object_mut().unwrap().remove("signature");
        assert!(build(Category::Welfare, &v, today()).unwrap().case.signature.is_none());
    }

    #[test]
    fn an_id_card_that_matches_raises_the_identity_level() {
        let mut v = welfare_values();
        v["scanned"] = json!([{ "kind": "신분증", "hash": "id", "fields": { "name": "이서연", "birth": "000315" } }]);
        assert_eq!(build(Category::Welfare, &v, today()).unwrap().case.intake.identity, Identity::IdCardScanned);
        v["scanned"] = json!([{ "kind": "신분증", "hash": "id", "fields": { "name": "김철수", "birth": "990101" } }]);
        let d = build(Category::Welfare, &v, today()).unwrap();
        assert_eq!(d.case.intake.identity, Identity::Declared);
        assert!(d.case.intake.pending.iter().any(|p| p.contains("신분증")));
    }

    #[test]
    fn subsidy_takes_the_claimed_total_only() {
        let v = json!({ "applicant_name": "장성민", "applicant_birth": "750606", "org_name": "마을공동체협의회", "program": "마을 환경 개선 사업", "claimed_total": "750,000" });
        let c = build(Category::Subsidy, &v, today()).unwrap().case;
        assert_eq!(c.intake.subsidy.as_ref().unwrap().claimed_total, 750_000);
    }

    #[test]
    fn permit_checks_the_business_type_list() {
        let v = json!({ "applicant_name": "박민수", "applicant_birth": "800202", "business_name": "행복식당", "business_type": "일반음식점", "address": "서울특별시 강남구 테헤란로 1", "area_m2": "66" });
        assert!(build(Category::Permit, &v, today()).is_ok());
        let mut bad = v.clone();
        bad["business_type"] = json!("유흥업");
        assert!(build(Category::Permit, &bad, today()).is_err());
    }

    #[test]
    fn cert_treats_the_agent_as_the_applicant_and_keeps_the_principal() {
        let v = json!({
            "principal_name": "최순자", "principal_birth": "450505", "applicant_name": "정현우", "applicant_birth": "750909",
            "relation": "자녀", "document": "인감증명서", "purpose": "부동산 매매", "poa_date": "2026-10-01", "signature": sig()
        });
        let c = build(Category::Cert, &v, today()).unwrap().case;
        assert_eq!(c.poa_written_on, NaiveDate::from_ymd_opt(2026, 10, 1));
        assert_eq!(c.applicant.name, "정현우");
        assert_eq!(c.intake.cert.as_ref().unwrap().principal.as_ref().unwrap().name, "최순자");
    }

    #[test]
    fn future_dates_are_refused() {
        let v = json!({
            "principal_name": "최순자", "principal_birth": "450505", "applicant_name": "정현우", "applicant_birth": "750909",
            "relation": "자녀", "document": "인감증명서", "purpose": "부동산 매매", "poa_date": "2027-01-01"
        });
        assert!(build(Category::Cert, &v, today()).is_err());
    }

    #[test]
    fn prefixes_are_distinct_per_category() {
        let all = [Category::Welfare, Category::Subsidy, Category::Permit, Category::Cert, Category::Grievance];
        let mut letters: Vec<char> = all.iter().map(|c| prefix(*c)).collect();
        letters.sort();
        letters.dedup();
        assert_eq!(letters.len(), 5);
    }
}
