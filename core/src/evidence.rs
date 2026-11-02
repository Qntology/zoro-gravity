//! 증빙 대조: 민원 한 건의 사실을 서류 스캔과 합성 대장에서 채우고, 무엇으로 확인했는지 기록합니다.
//!
//! 사실의 출처는 `서류 스캔 > 합성 대장` 순서입니다. 둘 다 있으면 맞춰 보고, 어긋나면 창구 확인으로 넘깁니다.
//! 둘 다 없으면 지어내지 않고 `pending` 에 남깁니다. 공무원이 창구에서 원본을 보고 값을 넣습니다.
//! 대장 값은 민원인 화면에 보이지 않습니다. 성명과 생년월일만으로 남의 대장 값이 보이면 안 되기 때문입니다.

use chrono::Months;

use crate::case::{CaseFile, Category, DocEvidence, Evidence, Person, Receipt};
use crate::forms::{Draft, ScanDoc, ID_CARD};
use crate::lattice::{edit_distance, Lattice};
use crate::state_machine::gate0_router::mandatory_docs;
use crate::state_machine::gate3_substantive::MAX_HOPS;

fn doc<'a>(scans: &'a [ScanDoc], kind: &str) -> Option<&'a ScanDoc> {
    scans.iter().find(|s| s.kind == kind)
}

fn prove(case: &mut CaseFile, kind: &str, how: Evidence) {
    if !case.intake.evidence.iter().any(|e| e.kind == kind) {
        case.intake.evidence.push(DocEvidence { kind: kind.to_string(), how });
    }
}

fn line(case: &mut CaseFile, label: &str, value: String) {
    case.intake.form_lines.push((label.to_string(), value));
}

/// 사람을 합성 DB 에서 찾습니다. 하나로 정해지면 식별값을 채우고, 아니면 창구 확인으로 넘깁니다.
fn resolve(db: &Lattice, person: &mut Person, role: &str, pending: &mut Vec<String>) -> Option<String> {
    let found = db.persons_named(&person.name, &person.birth_prefix);
    match found.as_slice() {
        [one] => {
            person.identifier = one.id.clone();
            Some(one.id.clone())
        }
        [] => {
            pending.push(format!("관계망 미등록: {role}"));
            None
        }
        _ => {
            pending.push(format!("관계망 동명이인: {role} (성명과 생년월일이 같은 사람이 둘 이상입니다)"));
            None
        }
    }
}

/// 민원인이 고쳐도 심사에 쓰이지 않는 항목. 스캔 값을 고쳤다고 사람에게 넘기지 않습니다.
const FREE_EDITS: &[&str] = &["business_name", "area_m2", "contact", "purpose", "body", "relation", "document"];

/// 대장이나 신청 내용이 확인해 준 스캔 항목 (서류 종류, 항목 이름)
type Confirmed = Vec<(&'static str, &'static str)>;

/// 스캔 문서의 신뢰도 낮은 값과 민원인이 고친 값은 사람이 확인합니다. 다른 출처가 확인해 준 값은 뺍니다.
fn note_uncertain(case: &mut CaseFile, scans: &[ScanDoc], edited: &[String], confirmed: &Confirmed) {
    for s in scans {
        for f in &s.low {
            if !confirmed.iter().any(|(kind, label)| *kind == s.kind && label == f) {
                case.intake.pending.push(format!("스캔 값 확인: {} {f}", s.kind));
            }
        }
    }
    for f in edited.iter().filter(|f| !FREE_EDITS.contains(&f.as_str())) {
        case.intake.pending.push(format!("민원인이 고친 값 확인: {f}"));
    }
}

/// 서류의 이름이 이 사람의 것인지 봅니다. 글자 하나는 오독으로 봅니다.
fn same_name(paper: &str, name: &str) -> bool {
    edit_distance(paper, name) <= 1
}

/// 띄어쓰기를 뺀 주소 비교. 긴 주소의 두 글자까지는 오독으로 봅니다.
fn same_address(a: &str, b: &str) -> bool {
    let squeeze = |s: &str| s.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    edit_distance(&squeeze(a), &squeeze(b)) <= 2
}

/// 스캔한 서류를 사건에 올립니다. 신분증은 필수 서류가 아니라 본인 확인이라서 따로 둡니다.
fn register_scans(case: &mut CaseFile, scans: &[ScanDoc]) {
    for s in scans.iter().filter(|s| s.kind != ID_CARD) {
        case.docs.push(crate::case::ScannedDoc { kind: s.kind.clone(), content_hash: s.hash.clone(), pages: s.pages });
        if mandatory_docs(case.category).contains(&s.kind.as_str()) {
            prove(case, &s.kind, Evidence::Scanned);
        }
    }
}

/// 민원 한 건의 사실을 채웁니다. `hash` 는 노드 식별자를 솔트 해시로 바꾸는 함수입니다.
pub fn enrich(draft: Draft, db: &Lattice, hash: &dyn Fn(&str) -> String) -> CaseFile {
    let Draft { mut case, scans, edited } = draft;
    register_scans(&mut case, &scans);
    let confirmed = match case.category {
        Category::Welfare => welfare(&mut case, &scans, db, hash),
        Category::Subsidy => subsidy(&mut case, &scans, db, hash),
        Category::Permit => permit(&mut case, &scans, db),
        Category::Cert => cert(&mut case, &scans, db),
        Category::Grievance => Confirmed::new(),
    };
    note_uncertain(&mut case, &scans, &edited, &confirmed);
    case
}

fn welfare(case: &mut CaseFile, scans: &[ScanDoc], db: &Lattice, hash: &dyn Fn(&str) -> String) -> Confirmed {
    let mut pending = std::mem::take(&mut case.intake.pending);
    let mut confirmed = Confirmed::new();
    let tenant = resolve(db, &mut case.applicant, "신청인", &mut pending);
    let reg = tenant.as_deref().and_then(|t| db.lease_of(t));
    let paper = doc(scans, "임대차계약서");

    // 임대인: 서류가 먼저이고, 없으면 대장에서 찾습니다.
    let from_paper = paper.and_then(|d| Some(Person { name: d.text("landlord_name")?.to_string(), birth_prefix: d.text("landlord_birth")?.to_string(), identifier: String::new() }));
    let mut landlord = from_paper.or_else(|| {
        let row = db.person(&reg?.landlord)?;
        Some(Person { name: row.name.clone(), birth_prefix: row.birth.clone(), identifier: String::new() })
    });
    let landlord_id = landlord.as_mut().and_then(|p| resolve(db, p, "임대인", &mut pending));
    if let (Some(id), Some(r), Some(_)) = (&landlord_id, reg, paper) {
        if *id != r.landlord {
            pending.push("계약서의 임대인이 대장과 다릅니다".to_string());
        } else {
            confirmed.extend([("임대차계약서", "임대인 성명"), ("임대차계약서", "임대인 주민번호")]);
        }
    }
    match paper.and_then(|d| d.text("tenant_name")) {
        Some(n) if !same_name(n, &case.applicant.name) => pending.push("계약서의 임차인이 신청인과 다릅니다".to_string()),
        Some(_) => confirmed.push(("임대차계약서", "임차인 성명")),
        None if reg.is_some() => confirmed.push(("임대차계약서", "임차인 성명")),
        None => {}
    }

    // 계약 월세
    let contract = paper.and_then(|d| d.number("contract_rent")).or(reg.map(|r| r.rent));
    match (paper.and_then(|d| d.number("contract_rent")), reg) {
        (Some(d), Some(r)) if d != r.rent => pending.push(format!("계약서 월세 {d}원이 대장 {}원과 다릅니다", r.rent)),
        // 못 읽은 값도 대장이 있으면 서류가 없는 경우와 같습니다.
        (_, Some(_)) => confirmed.push(("임대차계약서", "계약 월세")),
        _ => {}
    }
    if paper.is_none() && reg.is_some() {
        prove(case, "임대차계약서", Evidence::Registry);
    }

    // 이체액과 통장 계좌
    let statement = doc(scans, "이체내역서");
    let tenant_account = tenant.as_deref().and_then(|t| db.account_of(t));
    let ledger = tenant_account.and_then(|a| db.latest_transfer(&a.id));
    let transfer = statement.and_then(|d| d.number("latest_transfer")).or(ledger);
    match (statement.and_then(|d| d.number("latest_transfer")), ledger) {
        (Some(d), Some(r)) if d != r => pending.push(format!("이체내역서 금액 {d}원이 거래 원장 {r}원과 다릅니다")),
        (_, Some(_)) => confirmed.push(("이체내역서", "월세 이체액")),
        _ => {}
    }
    match statement.and_then(|d| d.text("holder")) {
        Some(n) if !same_name(n, &case.applicant.name) => pending.push("이체내역서의 예금주가 신청인과 다릅니다".to_string()),
        Some(_) => confirmed.push(("이체내역서", "예금주")),
        None if tenant_account.is_some() => confirmed.push(("이체내역서", "예금주")),
        None => {}
    }
    if let Some(a) = tenant_account {
        if statement.and_then(|d| d.text("printed_account")).is_none_or(|p| p == a.number) {
            confirmed.push(("이체내역서", "계좌번호"));
        }
    }
    if statement.is_none() && ledger.is_some() {
        prove(case, "이체내역서", Evidence::Registry);
    }
    let printed = statement.and_then(|d| d.text("printed_account").map(str::to_string)).or(tenant_account.map(|a| a.number.clone()));

    // 신청서와 동의서는 키오스크 입력과 서명으로 갈음합니다.
    if !case.docs.iter().any(|d| d.kind == "신청서") && case.signature.is_some() {
        prove(case, "신청서", Evidence::KioskSigned);
    }
    if !case.docs.iter().any(|d| d.kind == "금융정보동의서") && case.signature.is_some() {
        prove(case, "금융정보동의서", Evidence::KioskSigned);
    }

    let applied = case.lease.as_ref().map(|l| l.applied_rent).unwrap_or(0);
    case.lease = contract.zip(transfer).map(|(contract_rent, latest_transfer)| crate::case::LeaseFacts { applied_rent: applied, contract_rent, latest_transfer });
    let handwritten = case.bank.as_ref().map(|b| b.handwritten_account.clone()).unwrap_or_default();
    case.bank = printed.map(|printed_account| crate::case::BankFacts { handwritten_account: handwritten, printed_account });
    case.landlord = landlord;

    let source = |scanned: bool, known: bool| match (scanned, known) {
        (true, _) => "서류",
        (false, true) => "대장",
        _ => "미확인",
    };
    if let Some(p) = &case.landlord {
        let shown = format!("{} ({})", p.name, p.birth_prefix);
        line(case, "임대인", shown);
    }
    if let Some(c) = contract {
        line(case, "계약 월세", format!("{c}원 ({})", source(paper.is_some(), reg.is_some())));
    }
    if let Some(t) = transfer {
        line(case, "최근 이체액", format!("{t}원 ({})", source(statement.is_some(), ledger.is_some())));
    }
    if let Some(b) = &case.bank {
        let shown = b.printed_account.clone();
        line(case, "통장 계좌", shown);
    }

    let mut start: Vec<&str> = Vec::new();
    if let Some(t) = &tenant {
        start.push(t);
    }
    if let Some(l) = &landlord_id {
        start.push(l);
    }
    case.relations = db.relations_around(&start, MAX_HOPS, hash);
    case.intake.pending = pending;
    confirmed
}

fn receipts_from_scan(d: &ScanDoc) -> Vec<Receipt> {
    let Some(rows) = d.fields.get("receipts").and_then(|r| r.as_array()) else {
        return Vec::new();
    };
    rows.iter()
        .filter_map(|r| {
            Some(Receipt {
                approval: r.get("approval")?.as_str()?.to_string(),
                vendor: r.get("vendor")?.as_str()?.to_string(),
                date: r.get("date").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                time: r.get("time").and_then(|v| v.as_str()).unwrap_or("").to_string(),
                amount: r.get("amount").and_then(crate::forms::parse_number)?,
                item: r.get("item").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            })
        })
        .collect()
}

fn subsidy(case: &mut CaseFile, scans: &[ScanDoc], db: &Lattice, hash: &dyn Fn(&str) -> String) -> Confirmed {
    let mut pending = std::mem::take(&mut case.intake.pending);
    let mut facts = case.intake.subsidy.take().unwrap_or_default();
    resolve(db, &mut case.applicant, "신청인", &mut pending);
    let org = db.org_named(&facts.org);
    if org.is_none() {
        pending.push(format!("대장에 없는 단체: {}", facts.org));
    }
    let program = org.and_then(|o| db.program(&o.id, &facts.program));
    if org.is_some() && program.is_none() {
        pending.push(format!("대장에 없는 사업: {}", facts.program));
    }

    let paper = doc(scans, "영수증철");
    let rows = org.map(|o| db.card_rows(&o.id)).unwrap_or_default();
    facts.receipts = match paper {
        Some(d) => receipts_from_scan(d),
        None => rows
            .iter()
            .map(|r| Receipt {
                approval: r.approval.clone(),
                vendor: db.vendor(&r.vendor).map(|v| v.name.clone()).unwrap_or_else(|| r.vendor.clone()),
                date: r.date.clone(),
                time: r.time.clone(),
                amount: r.amount,
                item: r.item.clone(),
            })
            .collect(),
    };
    if paper.is_none() && !rows.is_empty() {
        prove(case, "영수증철", Evidence::Registry);
    }
    // 스캔 영수증은 카드 원장과 승인번호로 맞춥니다. 원장이 없는 단체는 대조할 수 없습니다.
    if paper.is_some() && !rows.is_empty() {
        for r in &facts.receipts {
            match rows.iter().find(|c| c.approval == r.approval) {
                Some(c) if c.amount != r.amount => pending.push(format!("영수증 {} 금액 {}원이 카드 원장 {}원과 다릅니다", r.approval, r.amount, c.amount)),
                Some(_) => {}
                None => pending.push(format!("카드 원장에 없는 승인번호: {}", r.approval)),
            }
        }
    }
    if let Some(d) = doc(scans, "통장사본") {
        if d.text("holder").is_some_and(|h| !same_name(h, &facts.org)) {
            pending.push("통장 예금주가 단체와 다릅니다".to_string());
        }
        let registered = program.and_then(|p| db.account(&p.account));
        if let (Some(a), Some(r)) = (d.text("account"), registered) {
            if a != r.number {
                pending.push("통장 계좌가 사업 대장의 계좌와 다릅니다".to_string());
            }
        }
    }
    if !case.docs.iter().any(|d| d.kind == "정산총괄표") {
        prove(case, "정산총괄표", Evidence::KioskSigned);
    }
    if org.is_some_and(|o| !db.tax_invoices(&o.id).is_empty()) {
        prove(case, "세금계산서", Evidence::Registry);
    }
    if program.is_some() {
        prove(case, "통장사본", Evidence::Registry);
    }

    // 거래처를 관계망 노드로 바꿉니다.
    let mut vendor_ids: Vec<String> = Vec::new();
    for r in &facts.receipts {
        if let Some(v) = db.vendor_named(&r.vendor) {
            if !vendor_ids.contains(&v.id) {
                vendor_ids.push(v.id.clone());
                facts.vendor_nodes.push((v.name.clone(), hash(&v.id)));
                if v.business.contains("유흥") {
                    facts.barred_vendors.push(v.name.clone());
                }
            }
        } else {
            pending.push(format!("대장에 없는 거래처: {}", r.vendor));
        }
    }
    facts.org_node = org.map(|o| hash(&o.id));
    let mut start: Vec<&str> = vendor_ids.iter().map(String::as_str).collect();
    if let Some(o) = org {
        start.push(&o.id);
    }
    case.relations = db.relations_around(&start, MAX_HOPS, hash);

    let total: i64 = facts.receipts.iter().map(|r| r.amount).sum();
    line(case, "영수증", format!("{}건, 합계 {total}원 ({})", facts.receipts.len(), if paper.is_some() { "서류" } else { "대장" }));
    for r in &facts.receipts {
        line(case, "결제", format!("{} {} · {} · {}원 · {}", r.date, r.time, r.vendor, r.amount, r.item));
    }
    case.intake.subsidy = Some(facts);
    case.intake.pending = pending;
    Confirmed::new()
}

/// 필증 유효기간: 합성 규칙으로 위생 2년, 소방 1년입니다.
const HYGIENE_MONTHS: u32 = 24;
const FIRE_MONTHS: u32 = 12;

fn permit(case: &mut CaseFile, scans: &[ScanDoc], db: &Lattice) -> Confirmed {
    let mut pending = std::mem::take(&mut case.intake.pending);
    let mut confirmed = Confirmed::new();
    let mut facts = case.intake.permit.take().unwrap_or_default();
    let today = case.today;
    let person = resolve(db, &mut case.applicant, "신고인", &mut pending);

    let address = case.intake.form_lines.iter().find(|(k, _)| k == "영업장 주소").map(|(_, v)| v.clone()).unwrap_or_default();
    let building = db.building_at(&address);
    let ledger = doc(scans, "건축물대장");
    facts.building_use = ledger.and_then(|d| d.text("use").map(str::to_string)).or(building.map(|b| b.usage.clone()));
    facts.building_violation = ledger
        .and_then(|d| d.text("violation").map(str::to_string))
        .or(building.and_then(|b| b.violation.as_ref().map(|v| format!("{}년 {}", v.year, v.note))));
    if ledger.is_none() && building.is_some() {
        prove(case, "건축물대장", Evidence::Registry);
    }
    // 위반 표기는 서류의 한 줄이라서 못 읽으면 조용히 빠집니다. 대장이 없으면 사람이 원본을 봅니다.
    if building.is_none() {
        pending.push(format!("대장에 없는 건축물: {address}"));
    }
    match ledger.and_then(|d| d.text("address")) {
        Some(a) if !same_address(a, &address) => pending.push("건축물대장의 주소가 신고한 영업장과 다릅니다".to_string()),
        Some(_) => confirmed.push(("건축물대장", "대지위치")),
        None if building.is_some() => confirmed.push(("건축물대장", "대지위치")),
        None => {}
    }
    if building.is_some_and(|b| ledger.and_then(|d| d.text("use")).is_none_or(|u| u == b.usage)) {
        confirmed.push(("건축물대장", "주용도"));
    }
    match doc(scans, "소방안전필증").and_then(|d| d.text("address")) {
        Some(a) if !same_address(a, &address) => pending.push("소방안전필증의 주소가 신고한 영업장과 다릅니다".to_string()),
        Some(_) => confirmed.push(("소방안전필증", "소재지")),
        None if building.is_some() => confirmed.push(("소방안전필증", "소재지")),
        None => {}
    }
    match doc(scans, "위생교육필증").and_then(|d| d.text("name")) {
        Some(n) if !same_name(n, &case.applicant.name) => pending.push("위생교육필증의 성명이 신고인과 다릅니다".to_string()),
        Some(_) => confirmed.extend([("위생교육필증", "성명"), ("위생교육필증", "생년월일")]),
        None => {}
    }
    if !case.docs.iter().any(|d| d.kind == "영업신고서") {
        prove(case, "영업신고서", Evidence::KioskSigned);
    }

    let from_scan = |kind: &str, months: u32| doc(scans, kind).and_then(|d| d.text("issued")).and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()).and_then(|d| d.checked_add_months(Months::new(months)));
    facts.hygiene_expires = from_scan("위생교육필증", HYGIENE_MONTHS).or(person.as_deref().and_then(|p| db.hygiene_of(p)).map(|c| c.expires));
    facts.fire_expires = from_scan("소방안전필증", FIRE_MONTHS).or(building.and_then(|b| db.fire_of(&b.id)).map(|c| c.expires));

    // 유효한 필증만 서류로 칩니다. 만료된 필증은 누락과 같아서 보완요구 대상입니다.
    let mut check = |case: &mut CaseFile, kind: &str, expires: Option<chrono::NaiveDate>| match expires {
        Some(e) if e >= today => {
            if doc(scans, kind).is_none() {
                prove(case, kind, Evidence::Registry);
            }
        }
        Some(e) => {
            case.intake.evidence.retain(|x| x.kind != kind);
            case.docs.retain(|d| d.kind != kind);
            pending.push(format!("{kind} 유효기간 만료 ({e})"));
        }
        None => {}
    };
    check(case, "위생교육필증", facts.hygiene_expires);
    check(case, "소방안전필증", facts.fire_expires);

    if let Some(u) = &facts.building_use {
        line(case, "건축물 용도", u.clone());
    }
    if let Some(v) = &facts.building_violation {
        line(case, "위반건축물 표기", v.clone());
    }
    if let Some(e) = facts.hygiene_expires {
        line(case, "위생교육필증 유효기간", e.to_string());
    }
    if let Some(e) = facts.fire_expires {
        line(case, "소방안전필증 유효기간", e.to_string());
    }
    case.intake.permit = Some(facts);
    case.intake.pending = pending;
    confirmed
}

fn cert(case: &mut CaseFile, scans: &[ScanDoc], db: &Lattice) -> Confirmed {
    let mut pending = std::mem::take(&mut case.intake.pending);
    let mut facts = case.intake.cert.take().unwrap_or_default();
    resolve(db, &mut case.applicant, "대리인", &mut pending);
    let mut principal = facts.principal.take();
    let principal_id = principal.as_mut().and_then(|p| resolve(db, p, "위임자", &mut pending));
    facts.seal_registered = principal_id.as_deref().and_then(|p| db.seal_of(p));
    if principal_id.is_some() && facts.seal_registered != Some(true) {
        pending.push("인감 대장에 위임자가 없습니다".to_string());
    }

    // 위임장 원본과 신분증은 대장으로 갈음할 수 없습니다. 스캔하거나 창구에서 봅니다.
    let paper = doc(scans, "위임장");
    facts.principal_signed = paper.map(|d| d.signed);
    // 서명이 없는 위임장은 서류로 치지 않습니다. 원본에 서명이 있으면 창구에서 확인하고, 없으면 보완을 요구합니다.
    if paper.is_some_and(|d| !d.signed) {
        pending.push("위임장에 위임자 서명이나 도장이 보이지 않습니다".to_string());
        case.intake.evidence.retain(|e| e.kind != "위임장");
        case.docs.retain(|d| d.kind != "위임장");
    }
    if let Some(p) = paper.and_then(|d| d.text("poa_date")).and_then(|s| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()) {
        case.poa_written_on = Some(p);
    }
    if paper.and_then(|d| d.text("agent_name")).is_some_and(|n| !same_name(n, &case.applicant.name)) {
        pending.push("위임장의 대리인이 신청인과 다릅니다".to_string());
    }
    if let (Some(n), Some(p)) = (paper.and_then(|d| d.text("principal_name")), &principal) {
        if !same_name(n, &p.name) {
            pending.push("위임장의 위임자가 입력과 다릅니다".to_string());
        }
    }
    for card in scans.iter().filter(|s| s.kind == ID_CARD) {
        let name = card.text("name");
        if principal.as_ref().is_some_and(|p| Some(p.name.as_str()) == name) {
            prove(case, "위임자신분증사본", Evidence::Scanned);
        }
        if Some(case.applicant.name.as_str()) == name {
            prove(case, "대리인신분증", Evidence::Scanned);
        }
    }
    facts.principal = principal;
    if let Some(s) = facts.seal_registered {
        line(case, "인감 대장", if s { "등록됨".to_string() } else { "없음".to_string() });
    }
    if let Some(signed) = facts.principal_signed {
        line(case, "위임자 서명 · 도장", if signed { "있음".to_string() } else { "없음".to_string() });
    }
    case.intake.cert = Some(facts);
    case.intake.pending = pending;
    Confirmed::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::Identity;
    use crate::forms::build;
    use chrono::NaiveDate;
    use serde_json::{json, Value};

    fn today() -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, 2).unwrap()
    }

    fn db() -> Lattice {
        Lattice::embedded().unwrap()
    }

    fn sig() -> Value {
        json!({ "stroke_count": 8, "bbox_area_ratio": 0.42, "single_line": false })
    }

    fn welfare(name: &str, birth: &str, rent: &str, account: &str) -> Value {
        json!({ "applicant_name": name, "applicant_birth": birth, "applied_rent": rent, "handwritten_account": account, "excuse": "none", "consent": true, "signature": sig() })
    }

    fn run(category: Category, v: &Value) -> CaseFile {
        let hash = |id: &str| id.to_string();
        enrich(build(category, v, today()).unwrap(), &db(), &hash)
    }

    #[test]
    fn with_no_papers_the_registry_fills_the_welfare_facts() {
        let c = run(Category::Welfare, &welfare("이서연", "000315", "400000", "110-123-100315"));
        let lease = c.lease.as_ref().unwrap();
        assert_eq!((lease.contract_rent, lease.latest_transfer), (400_000, 400_000));
        assert_eq!(c.bank.as_ref().unwrap().printed_account, "110-123-100315");
        assert_eq!(c.landlord.as_ref().unwrap().name, "최민수");
        assert!(c.intake.pending.is_empty(), "{:?}", c.intake.pending);
        assert!(c.intake.evidence.iter().any(|e| e.kind == "임대차계약서" && e.how == Evidence::Registry));
        assert_eq!(c.intake.evidence.len(), 4, "필수 서류 넷이 모두 확인됨");
    }

    #[test]
    fn a_person_outside_the_seed_goes_to_the_counter_and_gets_no_invented_facts() {
        let c = run(Category::Welfare, &welfare("없는사람", "000101", "400000", "110-123-100315"));
        assert!(c.intake.pending.iter().any(|p| p.contains("관계망 미등록")));
        assert!(c.lease.is_none());
    }

    #[test]
    fn two_people_with_the_same_name_and_birth_are_not_guessed() {
        let c = run(Category::Welfare, &welfare("김민재", "850101", "400000", "110-123-100315"));
        assert!(c.intake.pending.iter().any(|p| p.contains("동명이인")), "{:?}", c.intake.pending);
        assert!(c.applicant.identifier.is_empty());
    }

    #[test]
    fn a_paper_that_disagrees_with_the_registry_goes_to_the_counter() {
        let mut v = welfare("이서연", "000315", "400000", "110-123-100315");
        v["scanned"] = json!([{ "kind": "임대차계약서", "hash": "l1", "fields": { "landlord_name": "최민수", "landlord_birth": "701122", "contract_rent": "450,000" } }]);
        let c = run(Category::Welfare, &v);
        assert!(c.intake.pending.iter().any(|p| p.contains("대장")), "{:?}", c.intake.pending);
        assert_eq!(c.lease.as_ref().unwrap().contract_rent, 450_000, "서류 값이 먼저입니다");
    }

    #[test]
    fn low_confidence_and_edited_values_need_a_person() {
        let mut v = welfare("이서연", "000315", "400000", "110-123-100315");
        v["scanned"] = json!([{ "kind": "임대차계약서", "hash": "l1", "low": ["contract_rent"], "fields": { "landlord_name": "최민수", "landlord_birth": "701122", "contract_rent": "400000" } }]);
        v["edited"] = json!(["applied_rent"]);
        let c = run(Category::Welfare, &v);
        assert_eq!(c.intake.pending.len(), 2, "{:?}", c.intake.pending);
    }

    #[test]
    fn a_value_the_registry_confirms_needs_no_person_but_an_unread_one_without_a_registry_does() {
        let mut v = welfare("이서연", "000315", "400000", "110-123-100315");
        v["scanned"] = json!([{ "kind": "임대차계약서", "hash": "l1", "low": ["계약 월세"], "fields": { "landlord_name": "최민수", "landlord_birth": "701122", "tenant_name": "이서연" } }]);
        let c = run(Category::Welfare, &v);
        assert!(c.intake.pending.is_empty(), "{:?}", c.intake.pending);
        assert_eq!(c.lease.as_ref().unwrap().contract_rent, 400_000);

        let mut v = welfare("윤지아", "010909", "420000", "110-90-010909");
        v["scanned"] = json!([{ "kind": "임대차계약서", "hash": "l1", "low": ["계약 월세"], "fields": { "landlord_name": "한미지", "landlord_birth": "690101", "tenant_name": "윤지아" } }]);
        let c = run(Category::Welfare, &v);
        assert!(c.intake.pending.iter().any(|p| p.contains("스캔 값 확인")), "{:?}", c.intake.pending);
    }

    #[test]
    fn papers_of_someone_else_go_to_the_counter() {
        let mut v = welfare("이서연", "000315", "400000", "110-123-100315");
        v["scanned"] = json!([{ "kind": "임대차계약서", "hash": "l1", "fields": { "tenant_name": "박다른", "landlord_name": "최민수", "landlord_birth": "701122", "contract_rent": "400000" } }]);
        assert!(run(Category::Welfare, &v).intake.pending.iter().any(|p| p.contains("임차인")));

        let mut v = json!({ "applicant_name": "권서연", "applicant_birth": "860130", "business_name": "권씨네", "business_type": "일반음식점", "address": "서울특별시 영등포구 국회대로 40", "area_m2": "50" });
        v["scanned"] = json!([{ "kind": "위생교육필증", "hash": "c1", "fields": { "name": "남남남", "issued": "2026-03-01" } }]);
        assert!(run(Category::Permit, &v).intake.pending.iter().any(|p| p.contains("위생교육필증의 성명")));
    }

    #[test]
    fn a_building_outside_the_registry_is_always_seen_by_a_person() {
        let mut v = json!({ "applicant_name": "박민수", "applicant_birth": "800202", "business_name": "새식당", "business_type": "일반음식점", "address": "서울특별시 없는구 없는로 9", "area_m2": "30" });
        v["scanned"] = json!([{ "kind": "건축물대장", "hash": "b1", "fields": { "use": "제2종 근린생활시설", "address": "서울특별시 없는구 없는로 9" } }]);
        let c = run(Category::Permit, &v);
        assert!(c.intake.pending.iter().any(|p| p.contains("대장에 없는 건축물")), "{:?}", c.intake.pending);
    }

    #[test]
    fn a_scanned_receipt_is_matched_to_the_card_ledger_by_approval_number() {
        let mut v = json!({ "applicant_name": "장성민", "applicant_birth": "750606", "org_name": "마을공동체협의회", "program": "마을 환경 개선 사업", "claimed_total": "750000" });
        v["scanned"] = json!([{ "kind": "영수증철", "hash": "r1", "fields": { "receipts": [
            { "approval": "71200451", "vendor": "우리인쇄", "date": "2026-10-05", "time": "10:12", "amount": "999000", "item": "홍보물 인쇄" },
            { "approval": "99999999", "vendor": "대한문구", "date": "2026-10-07", "time": "15:40", "amount": "180000", "item": "사무용품" }
        ] } }]);
        let c = run(Category::Subsidy, &v);
        assert!(c.intake.pending.iter().any(|p| p.contains("카드 원장 320000원")), "{:?}", c.intake.pending);
        assert!(c.intake.pending.iter().any(|p| p.contains("카드 원장에 없는 승인번호")), "{:?}", c.intake.pending);
    }

    #[test]
    fn editing_a_value_the_review_does_not_use_stays_automatic() {
        let mut v = welfare("이서연", "000315", "400000", "110-123-100315");
        v["edited"] = json!(["contact", "applied_rent"]);
        let c = run(Category::Welfare, &v);
        assert_eq!(c.intake.pending.len(), 1, "{:?}", c.intake.pending);
    }

    #[test]
    fn the_relation_network_reaches_the_cases_that_need_it() {
        let c = run(Category::Welfare, &welfare("김철수", "990101", "500000", "110-123-991023"));
        assert_eq!(c.landlord.as_ref().unwrap().birth_prefix, "640312");
        assert!(!c.relations.is_empty(), "모친과의 연결이 사건에 붙습니다");
    }

    #[test]
    fn subsidy_takes_receipts_from_the_ledger_and_marks_barred_vendors() {
        let v = json!({ "applicant_name": "백승호", "applicant_birth": "740404", "org_name": "동네문화모임", "program": "주민 화합 행사", "claimed_total": "500000" });
        let c = run(Category::Subsidy, &v);
        let s = c.intake.subsidy.as_ref().unwrap();
        assert_eq!(s.receipts.len(), 2);
        assert_eq!(s.barred_vendors, vec!["달빛주점".to_string()]);
    }

    #[test]
    fn an_expired_certificate_counts_as_missing() {
        let v = json!({ "applicant_name": "권서연", "applicant_birth": "860130", "business_name": "권씨네", "business_type": "일반음식점", "address": "서울특별시 영등포구 국회대로 40", "area_m2": "50" });
        let c = run(Category::Permit, &v);
        assert!(c.intake.pending.iter().any(|p| p.contains("위생교육필증 유효기간 만료")));
        assert!(!c.intake.evidence.iter().any(|e| e.kind == "위생교육필증"));
    }

    #[test]
    fn an_unsigned_power_of_attorney_does_not_count_as_a_document() {
        let mut v = json!({
            "principal_name": "최순자", "principal_birth": "450505", "applicant_name": "정현우", "applicant_birth": "750909",
            "relation": "자녀", "document": "인감증명서", "purpose": "부동산 매매", "poa_date": "2026-10-01", "signature": sig()
        });
        v["scanned"] = json!([{ "kind": "위임장", "hash": "p1", "signed": false, "fields": { "poa_date": "2026-10-01" } }]);
        let c = run(Category::Cert, &v);
        assert!(!c.intake.evidence.iter().any(|e| e.kind == "위임장"));
        assert!(c.docs.is_empty());
        assert!(c.intake.pending.iter().any(|p| p.contains("서명")));
    }

    #[test]
    fn a_power_of_attorney_cannot_come_from_a_registry() {
        let v = json!({
            "principal_name": "최순자", "principal_birth": "450505", "applicant_name": "정현우", "applicant_birth": "750909",
            "relation": "자녀", "document": "인감증명서", "purpose": "부동산 매매", "poa_date": "2026-10-01", "signature": sig()
        });
        let c = run(Category::Cert, &v);
        assert!(!c.intake.evidence.iter().any(|e| e.kind == "위임장"));
        assert_eq!(c.intake.identity, Identity::Declared);
    }
}
