//! 합성 관계망 · 공공 대장 DB (명세 Table 3 `tb_graph_entities` 자리).
//!
//! 정부 API 와 데이터 파이프라인이 없어서 시드 파일을 미리 넣어 둔 상태로 심사합니다. 값은 모두 합성입니다.
//! 정의는 `docs/synthetic-data.md` 에 있습니다. 실제 연계가 생기면 이 모듈의 조회 함수 자리가 바뀝니다.
//!
//! 노드는 솔트 해시로 내보냅니다 (`relations_around`). 이름은 동명이인을 가려내는 데만 씁니다.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use anyhow::{Context, Result};
use chrono::NaiveDate;
use serde::Deserialize;

use crate::case::{Link, Node, Relation};

const EMBEDDED: &str = include_str!("../fixtures/synthetic_db.json");

#[derive(Debug, Clone, Deserialize)]
pub struct PersonRow {
    pub id: String,
    pub name: String,
    pub birth: String,
    pub address: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct OrgRow {
    pub id: String,
    pub name: String,
    pub kind: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VendorRow {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub business: String,
    pub address: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AccountRow {
    pub id: String,
    pub number: String,
    pub holder: String,
    pub bank: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ViolationRow {
    pub year: i32,
    pub note: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuildingRow {
    pub id: String,
    pub address: String,
    #[serde(rename = "use")]
    pub usage: String,
    pub violation: Option<ViolationRow>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum LinkRow {
    Family { degree: u8 },
    Household { year: i32 },
    SharedAccount,
    Spouse {
        #[allow(dead_code)]
        same_address_year: i32,
    },
    OfficerOf {
        #[allow(dead_code)]
        role: String,
    },
    RepresentedBy,
}

impl LinkRow {
    fn link(&self) -> Link {
        match self {
            LinkRow::Family { degree } => Link::Family { degree: *degree },
            LinkRow::Household { year } => Link::Household { year: *year },
            LinkRow::SharedAccount => Link::SharedAccount,
            LinkRow::Spouse { .. } => Link::Spouse,
            LinkRow::OfficerOf { .. } => Link::OfficerOf,
            LinkRow::RepresentedBy => Link::RepresentedBy,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct EdgeRow {
    a: String,
    b: String,
    link: LinkRow,
    label: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LeaseRow {
    pub tenant: String,
    pub landlord: String,
    pub address: String,
    pub deposit: i64,
    pub rent: i64,
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub account: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LedgerTransfer {
    pub from: String,
    pub to: String,
    pub months: BTreeMap<String, i64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProgramRow {
    pub org: String,
    pub program: String,
    pub budget: i64,
    pub account: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CardRow {
    pub org: String,
    pub approval: String,
    pub vendor: String,
    pub date: String,
    pub time: String,
    pub amount: i64,
    pub item: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TaxInvoiceRow {
    pub org: String,
    pub vendor: String,
    pub date: String,
    pub amount: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HygieneRow {
    pub person: String,
    pub no: String,
    pub issued: NaiveDate,
    pub expires: NaiveDate,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FireRow {
    pub building: String,
    pub no: String,
    pub issued: NaiveDate,
    pub expires: NaiveDate,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SealRow {
    pub person: String,
    pub registered: bool,
}

#[derive(Debug, Clone, Deserialize)]
struct Registries {
    leases: Vec<LeaseRow>,
    transfers: Vec<LedgerTransfer>,
    subsidy_programs: Vec<ProgramRow>,
    card_ledger: Vec<CardRow>,
    tax_invoices: Vec<TaxInvoiceRow>,
    hygiene_certs: Vec<HygieneRow>,
    fire_certs: Vec<FireRow>,
    seal_registry: Vec<SealRow>,
}

#[derive(Debug, Clone, Deserialize)]
struct Seed {
    as_of: NaiveDate,
    persons: Vec<PersonRow>,
    orgs: Vec<OrgRow>,
    vendors: Vec<VendorRow>,
    accounts: Vec<AccountRow>,
    buildings: Vec<BuildingRow>,
    edges: Vec<EdgeRow>,
    registries: Registries,
}

/// 합성 DB. 시작할 때 한 번 읽고 바꾸지 않습니다.
#[derive(Debug, Clone)]
pub struct Lattice {
    seed: Seed,
    by_name_birth: HashMap<(String, String), Vec<usize>>,
    adjacency: HashMap<String, Vec<usize>>,
}

/// 주소는 띄어쓰기를 빼고 맞춥니다.
fn squeeze(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

/// 두 글의 편집 거리 (글자 단위)
pub fn edit_distance(a: &str, b: &str) -> usize {
    let b: Vec<char> = b.chars().collect();
    let mut col: Vec<usize> = (0..=b.len()).collect();
    for (i, x) in a.chars().enumerate() {
        let mut diag = col[0];
        col[0] = i + 1;
        for (j, y) in b.iter().enumerate() {
            let next = (diag + (x != *y) as usize).min(col[j] + 1).min(col[j + 1] + 1);
            diag = col[j + 1];
            col[j + 1] = next;
        }
    }
    col[b.len()]
}

/// 길이에 맞게 허용하는 오독 글자 수. 짧은 값은 한 글자로도 다른 값이 됩니다.
fn tolerance(s: &str) -> usize {
    match s.chars().count() {
        0..=3 => 0,
        4..=9 => 1,
        _ => 2,
    }
}

/// 후보 가운데 `want` 와 가까운 값이 하나뿐이면 돌려줍니다. 둘 이상이면 고르지 않습니다.
fn only_close<'a, T>(rows: &'a [T], want: &str, key: impl Fn(&T) -> &str) -> Option<&'a T> {
    let want = squeeze(want);
    let allowed = tolerance(&want);
    let mut hits = rows.iter().filter(|r| edit_distance(&want, &squeeze(key(r))) <= allowed);
    let first = hits.next()?;
    hits.next().is_none().then_some(first)
}

impl Lattice {
    /// 저장소에 들어 있는 시드를 읽습니다.
    pub fn embedded() -> Result<Self> {
        Self::from_json(EMBEDDED)
    }

    pub fn from_json(text: &str) -> Result<Self> {
        let seed: Seed = serde_json::from_str(text).context("합성 DB 시드를 읽을 수 없습니다")?;
        let mut by_name_birth: HashMap<(String, String), Vec<usize>> = HashMap::new();
        for (i, p) in seed.persons.iter().enumerate() {
            by_name_birth.entry((p.name.clone(), p.birth.clone())).or_default().push(i);
        }
        let mut adjacency: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, e) in seed.edges.iter().enumerate() {
            adjacency.entry(e.a.clone()).or_default().push(i);
            adjacency.entry(e.b.clone()).or_default().push(i);
        }
        Ok(Self { seed, by_name_birth, adjacency })
    }

    /// 시드의 기준일. 유효기간이 있는 값은 이 날짜를 기준으로 만들어졌습니다.
    pub fn as_of(&self) -> NaiveDate {
        self.seed.as_of
    }

    /// 관계망 노드 수 (사람 · 단체 · 거래처 · 계좌 · 건축물)
    pub fn node_count(&self) -> usize {
        self.seed.persons.len() + self.seed.orgs.len() + self.seed.vendors.len() + self.seed.accounts.len() + self.seed.buildings.len()
    }

    pub fn person(&self, id: &str) -> Option<&PersonRow> {
        self.seed.persons.iter().find(|p| p.id == id)
    }

    /// 성명과 생년월일 앞자리로 후보를 찾습니다 (명세 Part 4 §1.1 의 1차 후보군).
    pub fn persons_named(&self, name: &str, birth: &str) -> Vec<&PersonRow> {
        self.by_name_birth
            .get(&(name.trim().to_string(), birth.trim().to_string()))
            .into_iter()
            .flatten()
            .map(|i| &self.seed.persons[*i])
            .collect()
    }

    pub fn org_named(&self, name: &str) -> Option<&OrgRow> {
        let want = squeeze(name);
        self.seed.orgs.iter().find(|o| squeeze(&o.name) == want)
    }

    pub fn vendor(&self, id: &str) -> Option<&VendorRow> {
        self.seed.vendors.iter().find(|v| v.id == id)
    }

    /// 스캔 오독 보정: 성명과 생년월일을 합쳐 한 글자 차이인 사람이 하나뿐이면 돌려줍니다.
    pub fn similar_person(&self, name: &str, birth: &str) -> Option<&PersonRow> {
        let mut hits = self.seed.persons.iter().filter(|p| edit_distance(name, &p.name) + edit_distance(birth, &p.birth) <= 1);
        let first = hits.next()?;
        hits.next().is_none().then_some(first)
    }

    pub fn similar_org(&self, name: &str) -> Option<&OrgRow> {
        only_close(&self.seed.orgs, name, |o| &o.name)
    }

    pub fn similar_vendor(&self, name: &str) -> Option<&VendorRow> {
        only_close(&self.seed.vendors, name, |v| &v.name)
    }

    pub fn similar_address(&self, address: &str) -> Option<&BuildingRow> {
        only_close(&self.seed.buildings, address, |b| &b.address)
    }

    pub fn similar_program(&self, org: &str, name: &str) -> Option<&ProgramRow> {
        // 단체가 정해지면 후보가 한둘이라서 세 글자에 한 글자까지 봐 줍니다.
        let want = squeeze(name);
        let allowed = want.chars().count() / 3;
        let mut hits = self.seed.registries.subsidy_programs.iter().filter(|p| p.org == org && edit_distance(&want, &squeeze(&p.program)) <= allowed);
        let first = hits.next()?;
        hits.next().is_none().then_some(first)
    }

    /// 카드 원장의 승인번호 조회
    pub fn card_by_approval(&self, approval: &str) -> Option<&CardRow> {
        self.seed.registries.card_ledger.iter().find(|c| c.approval == approval)
    }

    pub fn vendor_named(&self, name: &str) -> Option<&VendorRow> {
        let want = squeeze(name);
        self.seed.vendors.iter().find(|v| squeeze(&v.name) == want)
    }

    pub fn account(&self, id: &str) -> Option<&AccountRow> {
        self.seed.accounts.iter().find(|a| a.id == id)
    }

    /// 이름이 있는 노드의 표시 이름 (사람 · 단체 · 거래처)
    pub fn display_name(&self, id: &str) -> String {
        if let Some(p) = self.person(id) {
            return p.name.clone();
        }
        if let Some(o) = self.seed.orgs.iter().find(|o| o.id == id) {
            return o.name.clone();
        }
        if let Some(v) = self.vendor(id) {
            return v.name.clone();
        }
        id.to_string()
    }

    /// `start` 노드들에서 `hops` 홉 안에 닿는 관계를 사건에 붙일 모양으로 돌려줍니다.
    pub fn relations_around(&self, start: &[&str], hops: usize, hash: &dyn Fn(&str) -> String) -> Vec<Relation> {
        let mut seen: HashSet<&str> = HashSet::new();
        let mut used: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<(&str, usize)> = VecDeque::new();
        for id in start {
            if seen.insert(id) {
                queue.push_back((id, 0));
            }
        }
        while let Some((id, depth)) = queue.pop_front() {
            if depth >= hops {
                continue;
            }
            for &i in self.adjacency.get(id).into_iter().flatten() {
                used.insert(i);
                let e = &self.seed.edges[i];
                let next = if e.a == id { e.b.as_str() } else { e.a.as_str() };
                if seen.insert(next) {
                    queue.push_back((next, depth + 1));
                }
            }
        }
        let mut edges: Vec<usize> = used.into_iter().collect();
        edges.sort_unstable();
        edges
            .into_iter()
            .map(|i| {
                let e = &self.seed.edges[i];
                Relation {
                    a: Node { hash: hash(&e.a), name: self.display_name(&e.a) },
                    b: Node { hash: hash(&e.b), name: self.display_name(&e.b) },
                    link: e.link.link(),
                }
            })
            .collect()
    }

    /// 두 노드를 잇는 관계의 이름 (공무원 화면의 경로 표시용)
    pub fn edge_label(&self, a: &str, b: &str) -> Option<&str> {
        self.adjacency.get(a)?.iter().map(|i| &self.seed.edges[*i]).find(|e| e.a == b || e.b == b).map(|e| e.label.as_str())
    }

    pub fn lease_of(&self, tenant: &str) -> Option<&LeaseRow> {
        self.seed.registries.leases.iter().find(|l| l.tenant == tenant)
    }

    pub fn account_of(&self, holder: &str) -> Option<&AccountRow> {
        self.seed.accounts.iter().find(|a| a.holder == holder)
    }

    /// 계좌에서 나간 이체 중 가장 최근 달의 금액
    pub fn latest_transfer(&self, from_account: &str) -> Option<i64> {
        self.seed.registries.transfers.iter().find(|t| t.from == from_account).and_then(|t| t.months.values().next_back().copied())
    }

    pub fn program(&self, org: &str, name: &str) -> Option<&ProgramRow> {
        let want = squeeze(name);
        self.seed.registries.subsidy_programs.iter().find(|p| p.org == org && squeeze(&p.program) == want)
    }

    pub fn card_rows(&self, org: &str) -> Vec<&CardRow> {
        self.seed.registries.card_ledger.iter().filter(|r| r.org == org).collect()
    }

    pub fn tax_invoices(&self, org: &str) -> Vec<&TaxInvoiceRow> {
        self.seed.registries.tax_invoices.iter().filter(|r| r.org == org).collect()
    }

    pub fn building_at(&self, address: &str) -> Option<&BuildingRow> {
        let want = squeeze(address);
        self.seed.buildings.iter().find(|b| {
            let have = squeeze(&b.address);
            !want.is_empty() && (want.contains(&have) || have.contains(&want))
        })
    }

    pub fn hygiene_of(&self, person: &str) -> Option<&HygieneRow> {
        self.seed.registries.hygiene_certs.iter().find(|c| c.person == person)
    }

    pub fn fire_of(&self, building: &str) -> Option<&FireRow> {
        self.seed.registries.fire_certs.iter().find(|c| c.building == building)
    }

    pub fn seal_of(&self, person: &str) -> Option<bool> {
        self.seed.registries.seal_registry.iter().find(|s| s.person == person).map(|s| s.registered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Lattice {
        Lattice::embedded().unwrap()
    }

    #[test]
    fn the_seed_refers_only_to_nodes_it_defines() {
        let db = db();
        let known = |id: &str| {
            db.person(id).is_some() || db.seed.orgs.iter().any(|o| o.id == id) || db.vendor(id).is_some() || db.seed.buildings.iter().any(|b| b.id == id)
        };
        for e in &db.seed.edges {
            assert!(known(&e.a), "{}", e.a);
            assert!(known(&e.b), "{}", e.b);
        }
        for a in &db.seed.accounts {
            assert!(known(&a.holder), "{}", a.holder);
        }
        for l in &db.seed.registries.leases {
            assert!(db.person(&l.tenant).is_some() && db.person(&l.landlord).is_some() && db.account(&l.account).is_some(), "{}", l.tenant);
        }
        for t in &db.seed.registries.transfers {
            assert!(db.account(&t.from).is_some() && db.account(&t.to).is_some(), "{}", t.from);
        }
        for r in &db.seed.registries.card_ledger {
            assert!(known(&r.org) && db.vendor(&r.vendor).is_some(), "{}", r.approval);
        }
    }

    #[test]
    fn name_and_birth_find_people_and_keep_homonyms_apart() {
        let db = db();
        assert_eq!(db.persons_named("박영희", "640312").len(), 1);
        assert_eq!(db.persons_named("박영희", "580808").len(), 1);
        assert_eq!(db.persons_named("김민재", "850101").len(), 2, "이름과 생년월일이 같은 두 사람");
        assert!(db.persons_named("없는사람", "000101").is_empty());
    }

    #[test]
    fn relations_stop_at_the_hop_limit() {
        let db = db();
        let hash = |id: &str| id.to_string();
        let one = db.relations_around(&["O001"], 1, &hash);
        assert_eq!(one.len(), 1, "단체 → 총무이사");
        let three = db.relations_around(&["O001"], 3, &hash);
        assert_eq!(three.len(), 3, "단체 → 총무이사 → 배우자 → 거래처");
        assert!(three.iter().any(|r| r.link == Link::Spouse));
    }

    #[test]
    fn the_registries_answer_the_scenarios() {
        let db = db();
        assert_eq!(db.lease_of("P001").unwrap().rent, 400_000);
        assert_eq!(db.latest_transfer("A003"), Some(300_000), "2026-10 이체액");
        assert_eq!(db.building_at("서울특별시 성동구 왕십리로 22").unwrap().violation.as_ref().unwrap().year, 2025);
        assert!(db.building_at("서울특별시강남구테헤란로1").is_some(), "띄어쓰기가 달라도 찾습니다");
        assert!(db.hygiene_of("P033").unwrap().expires < db.as_of(), "만료된 필증");
        assert!(db.fire_of("B005").is_none(), "소방 필증이 없는 건물");
    }
}
