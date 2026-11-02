//! 스캔한 서류의 글자에서 서류 종류와 항목을 뽑습니다.
//!
//! OCR 은 글자만 읽습니다. 어떤 칸이 어떤 항목인지는 서식의 항목 이름으로 정합니다.
//! 글자 신뢰도가 낮거나 형식이 맞지 않거나 읽지 못한 값은 `low` 에 넣어 사람이 확인하게 합니다.
//! 값을 고치는 경우는 하나뿐입니다. 합성 대장에 비슷한 값이 하나만 있으면 그 값으로 바꿉니다 (`correct`).
//! 대장이 값을 확인해 주면 `low` 에서도 뺍니다. 무작위 오독이 대장의 값과 맞을 일은 드물기 때문입니다.

use serde_json::{json, Map, Value};

use crate::ai::document::PageReading;
use crate::ai::siglip2::plan::CropKind;
use crate::case::Category;
use crate::forms::ID_CARD;
use crate::lattice::{edit_distance, Lattice};

/// 값의 글자 신뢰도가 이보다 낮으면 사람이 확인합니다. 합성 서류에서 재면 0.9 에서 오독 7건 중 5건을 잡고,
/// 대장이 확인해 준 값은 넘기지 않으므로 자동 확정은 줄지 않았습니다. 나머지는 대장 대조와 민원인 확인이 잡습니다.
const LOW_CONF: f32 = 0.9;

/// OCR 조각 하나
#[derive(Debug, Clone, Default)]
pub struct PageCrop {
    /// 제목이면 서류 종류를 먼저 여기서 찾습니다.
    pub title: bool,
    pub bbox: [u32; 4],
    pub text: String,
    /// 글자별 신뢰도. 없으면 글자를 모두 믿지 않습니다.
    pub confs: Vec<f32>,
}

/// 페이지 한 장의 읽기 결과
#[derive(Debug, Clone, Default)]
pub struct PageText {
    pub hash: String,
    pub crops: Vec<PageCrop>,
    /// 서명 · 도장으로 잡힌 상자
    pub signs: Vec<[u32; 4]>,
}

impl PageText {
    pub fn from_reading(hash: &str, r: &PageReading) -> Self {
        let crops = r
            .crops
            .iter()
            .map(|c| PageCrop { title: c.kind == CropKind::Title, bbox: [c.bbox.0, c.bbox.1, c.bbox.2, c.bbox.3], text: c.output.text.clone(), confs: c.output.char_confidence.clone() })
            .collect();
        let signs = r.plan.crops.iter().filter(|c| c.kind == CropKind::Sign).map(|c| [c.bbox.0, c.bbox.1, c.bbox.2, c.bbox.3]).collect();
        Self { hash: hash.to_string(), crops, signs }
    }

    /// `case_read` 가 저장한 읽기 결과
    pub fn from_json(hash: &str, v: &Value) -> Self {
        let bbox = |b: &Value| -> [u32; 4] {
            let mut out = [0; 4];
            for (i, x) in b.as_array().into_iter().flatten().take(4).enumerate() {
                out[i] = x.as_u64().unwrap_or(0) as u32;
            }
            out
        };
        let crops = v["crops"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|c| PageCrop {
                title: c["kind"].as_str() == Some(CropKind::Title.name()),
                bbox: bbox(&c["bbox"]),
                text: c["text"].as_str().unwrap_or("").to_string(),
                confs: c["confs"].as_array().map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect()).unwrap_or_default(),
            })
            .collect();
        let signs = v["signs"].as_array().into_iter().flatten().map(bbox).collect();
        Self { hash: hash.to_string(), crops, signs }
    }
}

/// 표의 칸 하나. `conf` 는 칸에서 가장 낮은 글자 신뢰도입니다.
#[derive(Debug, Clone)]
struct Cell {
    text: String,
    conf: f32,
}

#[derive(Debug, Clone)]
struct Row {
    cells: Vec<Cell>,
    /// 가로 중심과 세로 위치
    x: u32,
    y: u32,
    title: bool,
}

/// 글자 · 숫자만 남깁니다. 항목 이름은 이 모양으로 맞춥니다.
fn squeeze(s: &str) -> String {
    s.chars().filter(|c| c.is_alphanumeric()).collect()
}

/// 표의 항목 이름이 맞는지 봅니다. 짧은 이름은 한 글자 차이로 다른 항목이 되므로 그대로 맞아야 합니다.
fn near(cell: &str, label: &str) -> bool {
    let (c, l) = (squeeze(cell), squeeze(label));
    if c == l {
        return true;
    }
    let n = l.chars().count();
    n >= 3 && edit_distance(&c, &l) <= if n >= 6 { 2 } else { 1 }
}

/// `want` 와 `text` 의 어떤 부분 문자열 사이의 가장 작은 편집 거리
fn best_distance(want: &[char], text: &[char]) -> usize {
    let mut col: Vec<usize> = (0..=want.len()).collect();
    let mut best = want.len();
    for &c in text {
        let mut diag = col[0];
        col[0] = 0;
        for (i, &w) in want.iter().enumerate() {
            let next = (diag + (w != c) as usize).min(col[i] + 1).min(col[i + 1] + 1);
            diag = col[i + 1];
            col[i + 1] = next;
        }
        best = best.min(col[want.len()]);
    }
    best
}

fn push_cell(cells: &mut Vec<Cell>, chars: &mut Vec<(char, f32)>) {
    let text: String = chars.iter().map(|(c, _)| *c).collect();
    // 글자만 봅니다. 공백과 쉼표의 신뢰도는 값과 상관이 적습니다.
    let conf = chars.iter().filter(|(c, _)| c.is_alphanumeric()).map(|(_, f)| *f).fold(1.0f32, f32::min);
    chars.clear();
    let text = text.trim().to_string();
    // `상호: 행복식당` 같은 칸을 항목과 값으로 나눕니다. 시각(10:12)은 나누지 않습니다.
    if let Some((label, value)) = text.split_once([':', '：']) {
        if !label.is_empty() && label.chars().count() <= 10 && !label.chars().any(|c| c.is_ascii_digit()) {
            cells.push(Cell { text: label.trim().to_string(), conf });
            if !value.trim().is_empty() {
                cells.push(Cell { text: value.trim().to_string(), conf });
            }
            return;
        }
    }
    cells.push(Cell { text, conf });
}

/// 조각의 글자를 표의 줄과 칸으로 나눕니다. 표 태그가 없으면 한 줄을 한 칸으로 봅니다.
fn rows_of(crop: &PageCrop) -> Vec<Row> {
    let chars: Vec<char> = crop.text.chars().collect();
    let conf = |i: usize| crop.confs.get(i).copied().unwrap_or(0.0);
    let mut rows: Vec<Row> = Vec::new();
    let make = |cells: Vec<Cell>| Row { cells, x: (crop.bbox[0] + crop.bbox[2]) / 2, y: 0, title: crop.title };
    let mut cells = Vec::new();
    let mut cell: Vec<(char, f32)> = Vec::new();
    let mut in_table = false;
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            if let Some(end) = chars[i..].iter().position(|c| *c == '>') {
                let tag: String = chars[i + 1..i + end].iter().collect();
                let name = tag.trim_start_matches('/').split_whitespace().next().unwrap_or("");
                match name {
                    "table" | "tr" | "td" | "th" | "tbody" | "thead" => {
                        in_table = true;
                        if name == "td" || name == "th" {
                            if tag.starts_with('/') {
                                push_cell(&mut cells, &mut cell);
                            }
                        } else if !cells.is_empty() {
                            rows.push(make(std::mem::take(&mut cells)));
                        }
                        i += end + 1;
                        continue;
                    }
                    _ => {}
                }
            }
        }
        if chars[i] == '\n' && !in_table {
            push_cell(&mut cells, &mut cell);
            rows.push(make(std::mem::take(&mut cells)));
        } else {
            cell.push((chars[i], conf(i)));
        }
        i += 1;
    }
    if !cell.is_empty() {
        push_cell(&mut cells, &mut cell);
    }
    if !cells.is_empty() {
        rows.push(make(cells));
    }
    // 조각 안의 줄은 높이를 고르게 나눠 가진다고 봅니다.
    let (top, height, n) = (crop.bbox[1], crop.bbox[3].saturating_sub(crop.bbox[1]), rows.len().max(1) as u32);
    for (i, row) in rows.iter_mut().enumerate() {
        row.y = top + height * (2 * i as u32 + 1) / (2 * n);
    }
    rows
}

/// 값의 모양
#[derive(Debug, Clone, Copy, PartialEq)]
enum Shape {
    Name,
    /// 주민번호 · 생년월일의 앞 6자리
    Birth,
    Money,
    Account,
    Text,
    /// 목록 안의 값 하나
    Choice(&'static [&'static str]),
    /// `117.1㎡` 같은 면적의 정수 부분
    Area,
}

/// 읽은 값을 표준 모양으로 바꿉니다. 모양이 맞지 않으면 None 입니다.
fn normalize(shape: Shape, raw: &str) -> Option<String> {
    let trimmed = raw.trim();
    match shape {
        Shape::Name => {
            let s: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
            let n = s.chars().count();
            ((2..=5).contains(&n) && s.chars().all(is_hangul)).then_some(s)
        }
        Shape::Birth => {
            let digits: String = trimmed.chars().take_while(|c| c.is_ascii_digit()).collect();
            if digits.len() < 6 {
                return None;
            }
            let birth = &digits[..6];
            let (month, day): (u32, u32) = (birth[2..4].parse().ok()?, birth[4..6].parse().ok()?);
            ((1..=12).contains(&month) && (1..=31).contains(&day)).then(|| birth.to_string())
        }
        Shape::Money => money(trimmed),
        Shape::Account => {
            let s: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
            let digits = s.chars().filter(char::is_ascii_digit).count();
            (digits >= 8 && s.chars().all(|c| c.is_ascii_digit() || c == '-') && !s.starts_with('-') && !s.ends_with('-')).then_some(s)
        }
        Shape::Text => {
            let s = trimmed.split_whitespace().collect::<Vec<_>>().join(" ");
            (!s.is_empty()).then_some(s)
        }
        Shape::Choice(list) => {
            let c = squeeze(trimmed);
            if let Some(hit) = list.iter().find(|x| squeeze(x) == c) {
                return Some(hit.to_string());
            }
            let close: Vec<&&str> = list.iter().filter(|x| edit_distance(&c, &squeeze(x)) <= 1).collect();
            match close.as_slice() {
                [one] => Some(one.to_string()),
                _ => None,
            }
        }
        Shape::Area => {
            let digits: String = trimmed.chars().skip_while(|c| !c.is_ascii_digit()).take_while(char::is_ascii_digit).collect();
            (!digits.is_empty() && digits.len() <= 6).then_some(digits)
        }
    }
}

fn is_hangul(c: char) -> bool {
    ('가'..='힣').contains(&c)
}

/// `금 1,000,000 원정` 에서 금액을 읽습니다. 쉼표 묶음이 세 자리가 아니면 자리를 잃은 오독으로 봅니다.
fn money(s: &str) -> Option<String> {
    let run: String = s.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit() || matches!(c, ',' | '.')).collect();
    let run = run.trim_end_matches([',', '.']);
    if run.is_empty() {
        return None;
    }
    let groups: Vec<&str> = run.split([',', '.']).collect();
    if groups.len() > 1 && (groups[0].len() > 3 || groups[1..].iter().any(|g| g.len() != 3)) {
        return None;
    }
    let digits: String = groups.concat();
    (digits.len() <= 12 && !digits.starts_with('0')).then_some(digits)
}

/// `2026 년 10 월 01 일`, `2026.10.01`, `2026-10-01` 을 `2026-10-01` 로 바꿉니다.
fn date(s: &str) -> Option<String> {
    let mut groups: Vec<String> = Vec::new();
    let mut now = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            now.push(c);
        } else if !now.is_empty() {
            groups.push(std::mem::take(&mut now));
        }
    }
    if !now.is_empty() {
        groups.push(now);
    }
    let [y, m, d, ..] = groups.as_slice() else {
        return None;
    };
    let (year, month, day): (i32, u32, u32) = (y.parse().ok()?, m.parse().ok()?, d.parse().ok()?);
    chrono::NaiveDate::from_ymd_opt(year, month, day).filter(|_| (2000..2100).contains(&year) && y.len() == 4).map(|d| d.to_string())
}

/// 서식의 항목 하나
struct Field {
    key: &'static str,
    /// 사람이 보는 이름
    label: &'static str,
    section: Option<&'static str>,
    labels: &'static [&'static str],
    shape: Shape,
    required: bool,
}

const fn field(key: &'static str, label: &'static str, section: Option<&'static str>, labels: &'static [&'static str], shape: Shape, required: bool) -> Field {
    Field { key, label, section, labels, shape, required }
}

const RELATIONS: &[&str] = &["배우자", "자녀", "부모", "형제자매", "형제", "기타"];
const BUSINESS: &[&str] = &["일반음식점", "휴게음식점", "제과점"];
/// 건축물 주용도 (건축법 시행령 별표 1 의 이름)
const USES: &[&str] = &["단독주택", "공동주택", "제1종 근린생활시설", "제2종 근린생활시설", "판매시설", "업무시설", "숙박시설", "위락시설"];

/// 서식 하나
struct Form {
    kind: &'static str,
    /// 띄어쓰기를 뺀 제목
    titles: &'static [&'static str],
    category: Option<Category>,
    /// 표의 머리 칸. 같은 항목 이름을 어느 사람의 것인지 가립니다.
    sections: &'static [&'static str],
    fields: &'static [Field],
    /// 서명란이 있는 서식
    signed: bool,
    /// 민원인 본인의 진술을 적은 서식. 값은 입력 칸을 채우는 데만 쓰고 민원인이 키오스크에서 확인합니다.
    statement: bool,
}

const FORMS: &[Form] = &[
    Form {
        kind: "임대차계약서",
        titles: &["임대차계약서"],
        category: Some(Category::Welfare),
        sections: &["임대인", "임차인"],
        fields: &[
            field("landlord_name", "임대인 성명", Some("임대인"), &["성명"], Shape::Name, true),
            field("landlord_birth", "임대인 주민번호", Some("임대인"), &["주민번호", "생년월일"], Shape::Birth, true),
            field("tenant_name", "임차인 성명", Some("임차인"), &["성명"], Shape::Name, true),
            field("contract_rent", "계약 월세", None, &["차임(월세)", "차임", "월세"], Shape::Money, true),
            field("deposit", "보증금", None, &["보증금"], Shape::Money, false),
        ],
        signed: false, statement: false,
    },
    Form {
        kind: "이체내역서",
        titles: &["이체내역서", "거래내역확인서"],
        category: Some(Category::Welfare),
        sections: &[],
        fields: &[field("holder", "예금주", None, &["예금주"], Shape::Name, true), field("printed_account", "계좌번호", None, &["계좌번호"], Shape::Account, true)],
        signed: false, statement: false,
    },
    Form {
        kind: "신청서",
        titles: &["지원신청서"],
        category: Some(Category::Welfare),
        sections: &["신청인", "임대차", "지급계좌"],
        fields: &[
            field("name", "신청인 성명", Some("신청인"), &["성명"], Shape::Name, true),
            field("birth", "신청인 생년월일", Some("신청인"), &["생년월일"], Shape::Birth, true),
            field("applied_rent", "신청 월세", Some("임대차"), &["월세"], Shape::Money, true),
            field("account", "입금 계좌", Some("지급계좌"), &["계좌번호"], Shape::Account, true),
        ],
        signed: true, statement: true,
    },
    Form { kind: "금융정보동의서", titles: &["금융정보등제공동의서", "금융정보제공동의서"], category: Some(Category::Welfare), sections: &[], fields: &[], signed: true, statement: true },
    Form { kind: "영수증철", titles: &["영수증첨부지"], category: Some(Category::Subsidy), sections: &[], fields: &[], signed: false, statement: false },
    Form {
        kind: "정산총괄표",
        titles: &["정산보고서총괄표"],
        category: Some(Category::Subsidy),
        sections: &[],
        fields: &[
            field("program", "사업명", None, &["사업명"], Shape::Text, true),
            field("org_name", "단체명", None, &["단체명"], Shape::Text, true),
            field("claimed_total", "정산금액", None, &["정산금액"], Shape::Money, true),
        ],
        signed: true, statement: true,
    },
    Form { kind: "세금계산서", titles: &["세금계산서"], category: Some(Category::Subsidy), sections: &[], fields: &[], signed: false, statement: false },
    Form {
        kind: "통장사본",
        titles: &["보통예금통장"],
        category: None,
        sections: &[],
        fields: &[field("holder", "예금주", None, &["예금주"], Shape::Text, true), field("account", "계좌번호", None, &["계좌번호"], Shape::Account, true)],
        signed: false, statement: false,
    },
    Form {
        kind: "건축물대장",
        titles: &["건축물대장"],
        category: Some(Category::Permit),
        sections: &[],
        fields: &[field("use", "주용도", None, &["주용도"], Shape::Choice(USES), true), field("address", "대지위치", None, &["대지위치"], Shape::Text, true)],
        signed: false, statement: false,
    },
    Form {
        kind: "위생교육필증",
        titles: &["위생교육이수증"],
        category: Some(Category::Permit),
        sections: &[],
        fields: &[field("name", "성명", None, &["성명"], Shape::Name, true), field("birth", "생년월일", None, &["생년월일"], Shape::Birth, true)],
        signed: false, statement: false,
    },
    Form {
        kind: "소방안전필증",
        titles: &["안전시설등완비증명서"],
        category: Some(Category::Permit),
        sections: &[],
        fields: &[field("address", "소재지", None, &["소재지"], Shape::Text, true)],
        signed: false, statement: false,
    },
    Form {
        kind: "영업신고서",
        titles: &["영업신고서"],
        category: Some(Category::Permit),
        sections: &[],
        fields: &[
            field("name", "신고인 성명", None, &["성명"], Shape::Name, false),
            field("birth", "신고인 생년월일", None, &["생년월일"], Shape::Birth, false),
            field("business_name", "상호", None, &["상호"], Shape::Text, true),
            field("address", "영업소 소재지", None, &["소재지"], Shape::Text, false),
            field("business_type", "영업의 종류", None, &["영업의종류"], Shape::Choice(BUSINESS), false),
            field("area_m2", "영업장 면적", None, &["영업장면적"], Shape::Area, false),
        ],
        signed: true, statement: true,
    },
    Form {
        kind: "위임장",
        titles: &["위임장"],
        category: Some(Category::Cert),
        sections: &["위임자", "대리인"],
        fields: &[
            field("principal_name", "위임자 성명", Some("위임자"), &["성명"], Shape::Name, true),
            field("principal_birth", "위임자 주민번호", Some("위임자"), &["주민번호"], Shape::Birth, true),
            field("agent_name", "대리인 성명", Some("대리인"), &["성명"], Shape::Name, true),
            field("agent_birth", "대리인 주민번호", Some("대리인"), &["주민번호"], Shape::Birth, false),
            field("relation", "관계", None, &["관계"], Shape::Choice(RELATIONS), false),
        ],
        signed: true, statement: false,
    },
    Form { kind: ID_CARD, titles: &["주민등록증", "운전면허증"], category: None, sections: &[], fields: &[], signed: false, statement: false },
    Form { kind: "진정서", titles: &["진정서"], category: Some(Category::Grievance), sections: &[], fields: &[], signed: true, statement: true },
];

/// 서식 종류를 정합니다. 제목 조각을 먼저 보고, 없으면 위쪽의 짧은 줄을 봅니다.
/// 본문 문장에 든 서류 이름(첨부 서류 안내 등)에는 속지 않습니다.
fn classify(rows: &[Row]) -> Option<&'static Form> {
    let tries = |row: &Row, tolerance: usize| -> Option<&'static Form> {
        let line = squeeze(&row.cells.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(""));
        let text: Vec<char> = line.chars().collect();
        FORMS.iter().find(|f| {
            f.titles.iter().any(|t| {
                let want: Vec<char> = t.chars().collect();
                let allowed = if want.len() >= 5 { tolerance } else { 0 };
                text.len() <= want.len() + 16 && best_distance(&want, &text) <= allowed
            })
        })
    };
    let titled = rows.iter().filter(|r| r.title).find_map(|r| tries(r, 2));
    titled.or_else(|| rows.iter().take(8).find_map(|r| tries(r, 1)))
}

#[derive(Debug, Clone)]
struct Reading {
    value: String,
    conf: f32,
    /// 모양이 맞는 값
    valid: bool,
}

impl Reading {
    /// 모양이 맞는 값을 먼저, 그다음 신뢰도가 높은 값을 고릅니다.
    fn better_than(&self, other: &Reading) -> bool {
        (self.valid, self.conf) > (other.valid, other.conf)
    }
}

fn keep(best: &mut Option<Reading>, next: Reading) {
    if best.as_ref().is_none_or(|b| next.better_than(b)) {
        *best = Some(next);
    }
}

/// 머리 칸은 세로로 합친 칸이라서 묶음의 가운데 줄에서 읽힙니다. 항목보다 뒤에 나올 수 있으므로 위치가 가까운 머리 칸을 씁니다.
fn sections_of(rows: &[Row], form: &Form) -> Vec<(u32, &'static str)> {
    let mut marks = Vec::new();
    for row in rows {
        for cell in &row.cells {
            // 임대인 · 임차인처럼 한 글자만 다른 머리 칸이 있어서 가장 가까운 이름을 고릅니다.
            let close = form.sections.iter().filter(|s| near(&cell.text, s)).min_by_key(|s| edit_distance(&squeeze(&cell.text), &squeeze(s)));
            if let Some(s) = close {
                marks.push((row.y, *s));
            }
        }
    }
    marks
}

fn section_at(marks: &[(u32, &'static str)], y: u32) -> Option<&'static str> {
    marks.iter().min_by_key(|(at, _)| at.abs_diff(y)).map(|(_, s)| *s)
}

fn read_field(rows: &[Row], form: &Form, f: &Field) -> Option<Reading> {
    let marks = sections_of(rows, form);
    let is_label = |text: &str| f.labels.iter().any(|l| near(text, l));
    let any_label = |text: &str| form.fields.iter().any(|o| o.labels.iter().any(|l| near(text, l)));
    let mut best = None;
    for row in rows {
        if f.section.is_some() && f.section != section_at(&marks, row.y) {
            continue;
        }
        for (i, cell) in row.cells.iter().enumerate() {
            let value = if is_label(&cell.text) {
                // 다음 칸도 항목 이름이면 이 칸은 머리 칸입니다 (`사업 | 사업명 | ...`).
                row.cells[i + 1..].iter().find(|c| !squeeze(&c.text).is_empty()).filter(|c| !any_label(&c.text)).cloned()
            } else {
                // `예금주 홍길동` 처럼 한 칸에 붙어 나온 경우
                cell.text.split_once(char::is_whitespace).filter(|(head, _)| f.labels.iter().any(|l| near(head, l))).map(|(_, tail)| Cell { text: tail.trim().to_string(), conf: cell.conf })
            };
            let Some(value) = value else { continue };
            let normal = normalize(f.shape, &value.text);
            let valid = normal.is_some();
            keep(&mut best, Reading { value: normal.unwrap_or(value.text), conf: value.conf, valid });
        }
    }
    best
}

fn all_cells(rows: &[Row]) -> impl Iterator<Item = &Cell> {
    rows.iter().flat_map(|r| r.cells.iter())
}

/// 페이지 아래쪽에서 처음 만나는 날짜 (발급일 · 작성일)
fn last_date(rows: &[Row]) -> Option<Reading> {
    let rows_text: Vec<(String, f32)> = rows
        .iter()
        .map(|r| (r.cells.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" "), r.cells.iter().map(|c| c.conf).fold(1.0f32, f32::min)))
        .collect();
    rows_text.iter().rev().filter(|(t, _)| t.contains('년') || t.contains('.') || t.contains('-')).find_map(|(t, conf)| Some(Reading { value: date(t)?, conf: *conf, valid: true }))
}

/// 거래 표에서 월세로 나간 마지막 금액. 표가 조각 여럿으로 나뉘어도 줄 모양으로 찾습니다.
fn latest_rent(rows: &[Row]) -> Option<Reading> {
    let mut out_col = 2;
    let mut best: Option<(String, Reading)> = None;
    for row in rows {
        if let Some(j) = row.cells.iter().position(|c| squeeze(&c.text) == "출금") {
            out_col = j;
        }
        let Some(first) = row.cells.first() else { continue };
        let Some(day) = date(&first.text) else { continue };
        if !row.cells.iter().any(|c| c.text.contains("월세")) {
            continue;
        }
        let Some(cell) = row.cells.get(out_col) else { continue };
        let normal = normalize(Shape::Money, &cell.text);
        let valid = normal.is_some();
        let reading = Reading { value: normal.unwrap_or(cell.text.clone()), conf: cell.conf.min(first.conf), valid };
        if best.as_ref().is_none_or(|(d, r)| day > *d || (day == *d && reading.better_than(r))) {
            best = Some((day, reading));
        }
    }
    best.map(|(_, r)| r)
}

/// 영수증 한 장의 항목
const RECEIPT_KEYS: &[(&str, &[&str])] = &[("vendor", &["가맹점"]), ("approval", &["승인번호"]), ("when", &["거래일시"]), ("item", &["품목"]), ("amount", &["합계"])];

#[derive(Debug, Default, Clone)]
struct ReceiptDraft {
    values: Map<String, Value>,
    low: bool,
}

/// 영수증 첨부지는 전표가 좌우 두 줄로 붙어 있어서 읽는 순서가 섞입니다. 조각의 가로 위치로 전표를 가릅니다.
fn receipts(rows: &[Row]) -> (Vec<Value>, Vec<String>) {
    let xs: Vec<u32> = rows.iter().filter(|r| !r.title).map(|r| r.x).collect();
    let mid = (xs.iter().copied().min().unwrap_or(0) + xs.iter().copied().max().unwrap_or(0)) / 2;
    let mut open: [Vec<ReceiptDraft>; 2] = [Vec::new(), Vec::new()];
    for row in rows {
        let side = usize::from(row.x > mid);
        let mut pairs = 0;
        for (i, cell) in row.cells.iter().enumerate() {
            let Some((key, _)) = RECEIPT_KEYS.iter().find(|(_, labels)| labels.iter().any(|l| near(&cell.text, l))) else { continue };
            let Some(value) = row.cells.get(i + 1) else { continue };
            // 한 줄에 전표 둘이 함께 읽히면 뒤쪽은 오른쪽 전표입니다.
            let col = (side + pairs).min(1);
            pairs += 1;
            if *key == "vendor" || open[col].is_empty() {
                open[col].push(ReceiptDraft::default());
            }
            let Some(draft) = open[col].last_mut() else { continue };
            draft.low |= value.conf < LOW_CONF;
            draft.values.insert((*key).to_string(), json!(value.text));
        }
    }
    let mut out = Vec::new();
    let mut low = Vec::new();
    // 위에서 아래로, 왼쪽에서 오른쪽으로 번호를 붙입니다.
    let longest = open[0].len().max(open[1].len());
    for i in 0..longest {
        for side in &open {
            let Some(draft) = side.get(i) else { continue };
            let text = |k: &str| draft.values.get(k).and_then(Value::as_str).unwrap_or("");
            let when: Vec<&str> = text("when").split_whitespace().collect();
            let day = when.first().copied().and_then(date);
            let amount = money(text("amount"));
            let approval: String = text("approval").chars().filter(char::is_ascii_digit).collect();
            let number = out.len() + 1;
            if draft.low || day.is_none() || amount.is_none() || approval.len() != 8 {
                low.push(format!("영수증 {number}"));
            }
            out.push(json!({
                "vendor": squeeze(text("vendor")),
                "approval": approval,
                "date": day.unwrap_or_default(),
                "time": when.get(1).copied().unwrap_or(""),
                "item": text("item"),
                "amount": amount.unwrap_or_default(),
            }));
        }
    }
    (out, low)
}

/// 건축물대장의 위반 표기. 변동 내용 표의 줄에서 찾습니다.
fn violation(rows: &[Row]) -> Option<Reading> {
    for row in rows {
        if let Some(i) = row.cells.iter().position(|c| c.text.contains("위반") && c.text.chars().count() <= 12) {
            let note = row.cells[i + 1..].iter().find(|c| !squeeze(&c.text).is_empty()).unwrap_or(&row.cells[i]);
            return Some(Reading { value: note.text.clone(), conf: note.conf, valid: true });
        }
    }
    None
}

/// 신분증: 번호 줄 위의 짧은 한글 줄이 이름입니다. 조각이 겹치면 같은 줄을 두 번 읽으므로 신뢰도가 높은 쪽을 씁니다.
fn id_card(rows: &[Row]) -> (Option<Reading>, Option<Reading>, Vec<String>) {
    let is_birth = |c: &Cell| c.text.contains('-') && normalize(Shape::Birth, &c.text).is_some();
    let lines: Vec<&Cell> = all_cells(rows).collect();
    let first_number = lines.iter().position(|c| is_birth(c)).unwrap_or(lines.len());
    // 제목이 겹쳐 읽히면 이름처럼 보이는 줄이 됩니다. 번호 줄 바로 위의 두 줄만 봅니다.
    let candidates: Vec<(String, f32)> = lines[..first_number]
        .iter()
        .filter(|c| !near(&c.text, "주민등록증") && !squeeze(&c.text).contains("주민"))
        .filter_map(|c| Some((normalize(Shape::Name, &c.text)?, c.conf)))
        .filter(|(n, _)| n.chars().count() <= 4)
        .collect();
    let mut name = None;
    let mut names = Vec::new();
    for (n, conf) in candidates.into_iter().rev().take(2) {
        names.push(n.clone());
        keep(&mut name, Reading { value: n, conf, valid: true });
    }
    let mut birth = None;
    for c in lines.iter().filter(|c| is_birth(c)) {
        if let Some(b) = normalize(Shape::Birth, &c.text) {
            keep(&mut birth, Reading { value: b, conf: c.conf, valid: true });
        }
    }
    (name, birth, names)
}

/// 주소처럼 보이는 줄. 진정인 줄 바로 다음의 주소를 본문에서 뺍니다.
fn looks_like_address(line: &str) -> bool {
    line.chars().any(|c| c.is_ascii_digit()) && ["특별시", "광역시", "도 ", "시 ", "구 "].iter().any(|w| line.contains(w))
}

/// 진정서: 진정인 줄에서 이름과 연락처를, 작성일 앞까지를 본문으로 봅니다.
fn petition(rows: &[Row], fields: &mut Map<String, Value>, low: &mut Vec<String>) {
    let mut body: Vec<&str> = Vec::new();
    for row in rows.iter().filter(|r| !r.title) {
        let line = row.cells.iter().map(|c| c.text.as_str()).collect::<Vec<_>>().join(" ");
        let words: Vec<&str> = line.split_whitespace().collect();
        if words.first().is_some_and(|w| near(w, "진정인")) {
            if let Some(n) = words.get(1).and_then(|w| normalize(Shape::Name, w)) {
                fields.entry("name").or_insert(json!(n));
            }
            if let Some(tail) = words.iter().find(|t| t.starts_with("01") && t.contains('-')) {
                fields.insert("contact".into(), json!(tail));
            }
            continue;
        }
        // 작성일 줄에서 본문이 끝납니다. 손글씨 연도는 `20 26 년` 처럼 끊겨 읽힙니다.
        if line.contains('년') && line.contains('월') && line.contains('일') && line.chars().count() < 30 {
            break;
        }
        if body.is_empty() && looks_like_address(&line) {
            continue;
        }
        body.extend(row.cells.iter().map(|c| c.text.as_str()).filter(|t| !t.is_empty()));
    }
    if body.is_empty() {
        low.push("본문".to_string());
    } else {
        fields.insert("text".into(), json!(body.join(" ")));
    }
}

/// 서식에 맞는 항목을 뽑습니다. 낮은 신뢰도 · 모양이 틀린 값 · 필수인데 못 읽은 값은 `low` 로 갑니다.
fn read_form(form: &Form, rows: &[Row]) -> (Map<String, Value>, Vec<String>) {
    let mut fields = Map::new();
    let mut low = Vec::new();
    let put = |fields: &mut Map<String, Value>, low: &mut Vec<String>, key: &str, label: &str, reading: Option<Reading>, required: bool| match reading {
        Some(r) => {
            // 필수가 아닌 값은 심사에 쓰이지 않으므로 사람에게 넘기지 않습니다.
            if required && (!r.valid || r.conf < LOW_CONF) {
                low.push(label.to_string());
            }
            if r.valid {
                fields.insert(key.to_string(), json!(r.value));
            }
        }
        None if required => low.push(label.to_string()),
        None => {}
    };
    for f in form.fields {
        put(&mut fields, &mut low, f.key, f.label, read_field(rows, form, f), f.required);
    }
    match form.kind {
        "이체내역서" => put(&mut fields, &mut low, "latest_transfer", "월세 이체액", latest_rent(rows), true),
        "영수증철" => {
            let (list, unsure) = receipts(rows);
            if list.is_empty() {
                low.push("영수증".to_string());
            }
            fields.insert("receipts".into(), Value::Array(list));
            low.extend(unsure);
        }
        "건축물대장" => put(&mut fields, &mut low, "violation", "위반건축물 표기", violation(rows), false),
        "위생교육필증" | "소방안전필증" => put(&mut fields, &mut low, "issued", "발급일", last_date(rows), true),
        "위임장" => put(&mut fields, &mut low, "poa_date", "작성일", last_date(rows), true),
        "진정서" => petition(rows, &mut fields, &mut low),
        ID_CARD => {
            let (name, birth, names) = id_card(rows);
            put(&mut fields, &mut low, "name", "성명", name, true);
            put(&mut fields, &mut low, "birth", "주민번호", birth, true);
            fields.insert(NAME_CANDIDATES.into(), json!(names));
        }
        _ => {}
    }
    (fields, low)
}

/// 신분증에서 이름으로 읽힌 줄들. 대장 보정에만 쓰고 내보내지 않습니다.
const NAME_CANDIDATES: &str = "name_candidates";

/// 서류 한 건. `forms::ScanDoc` 으로 들어가는 값입니다.
#[derive(Debug, Clone, PartialEq)]
pub struct Found {
    pub kind: String,
    pub hash: String,
    pub pages: u32,
    pub signed: bool,
    pub fields: Map<String, Value>,
    pub low: Vec<String>,
    /// 진술 서식에서 믿기 어려운 값. 사람에게 넘기지 않고 키오스크에서 민원인이 다시 봅니다.
    pub unsure: Vec<String>,
}

impl Found {
    pub fn to_value(&self) -> Value {
        json!({ "kind": self.kind, "hash": self.hash, "pages": self.pages, "signed": self.signed, "fields": self.fields, "low": self.low })
    }

    fn text(&self, key: &str) -> Option<&str> {
        self.fields.get(key).and_then(Value::as_str)
    }

    fn set(&mut self, key: &str, value: &str, label: &str) {
        self.fields.insert(key.to_string(), json!(value));
        self.low.retain(|l| l != label);
    }
}

/// 묶음 전체의 결과
#[derive(Debug, Clone, Default)]
pub struct Extracted {
    pub category: Option<Category>,
    pub docs: Vec<Value>,
    /// 키오스크 입력 칸에 미리 넣을 값
    pub prefill: Value,
    /// 어떤 서식인지 모르는 페이지 수
    pub unknown_pages: usize,
    /// 민원인이 다시 봐야 할 항목 이름
    pub unsure: Vec<String>,
}

fn read_page(page: &PageText) -> Option<Found> {
    let mut crops: Vec<&PageCrop> = page.crops.iter().collect();
    crops.sort_by_key(|c| (c.bbox[1], c.bbox[0]));
    let rows: Vec<Row> = crops.iter().flat_map(|c| rows_of(c)).collect();
    let form = classify(&rows)?;
    let (fields, low) = read_form(form, &rows);
    let (low, unsure) = if form.statement { (Vec::new(), low) } else { (low, Vec::new()) };
    Some(Found { kind: form.kind.to_string(), hash: page.hash.clone(), pages: 1, signed: form.signed && !page.signs.is_empty(), fields, low, unsure })
}

/// 사람 값 쌍을 대장에 맞춥니다. 대장에 그 쌍이 있으면 두 값 모두 확인된 것으로 봅니다.
fn correct_person(doc: &mut Found, db: &Lattice, name_key: &str, birth_key: &str, name_label: &str, birth_label: &str) {
    let (Some(name), Some(birth)) = (doc.text(name_key).map(str::to_string), doc.text(birth_key).map(str::to_string)) else {
        return;
    };
    if !db.persons_named(&name, &birth).is_empty() {
        doc.low.retain(|l| l != name_label && l != birth_label);
    } else if let Some(p) = db.similar_person(&name, &birth) {
        let (n, b) = (p.name.clone(), p.birth.clone());
        doc.set(name_key, &n, name_label);
        doc.set(birth_key, &b, birth_label);
    }
}

/// 대장에 비슷한 값이 하나만 있으면 그 값으로 바꿉니다.
fn correct_from(doc: &mut Found, key: &str, label: &str, find: impl Fn(&str) -> Option<String>) {
    if let Some(found) = doc.text(key).and_then(find) {
        doc.set(key, &found, label);
    }
}

/// 합성 대장으로 오독을 바로잡습니다. 대장에 없는 값은 그대로 둡니다.
fn correct(doc: &mut Found, db: &Lattice) {
    let org = |v: &str| db.similar_org(v).map(|o| o.name.clone());
    match doc.kind.as_str() {
        "임대차계약서" => correct_person(doc, db, "landlord_name", "landlord_birth", "임대인 성명", "임대인 주민번호"),
        "신청서" => correct_person(doc, db, "name", "birth", "신청인 성명", "신청인 생년월일"),
        "위생교육필증" => correct_person(doc, db, "name", "birth", "성명", "생년월일"),
        "위임장" => {
            correct_person(doc, db, "principal_name", "principal_birth", "위임자 성명", "위임자 주민번호");
            correct_person(doc, db, "agent_name", "agent_birth", "대리인 성명", "대리인 주민번호");
        }
        ID_CARD => {
            // 같은 줄을 두 번 읽은 후보 가운데 대장에 맞는 쪽이 있으면 그 후보를 씁니다.
            let names = doc.fields.remove(NAME_CANDIDATES);
            let birth = doc.text("birth").map(str::to_string).unwrap_or_default();
            let known = names.iter().flat_map(|n| n.as_array().into_iter().flatten()).filter_map(Value::as_str).find(|n| !db.persons_named(n, &birth).is_empty() || db.similar_person(n, &birth).is_some());
            if let Some(n) = known.map(str::to_string) {
                doc.fields.insert("name".to_string(), json!(n));
            }
            correct_person(doc, db, "name", "birth", "성명", "주민번호");
        }
        "영업신고서" => {
            correct_person(doc, db, "name", "birth", "신고인 성명", "신고인 생년월일");
            correct_from(doc, "address", "영업소 소재지", |v| db.similar_address(v).map(|b| b.address.clone()));
        }
        "정산총괄표" => {
            correct_from(doc, "org_name", "단체명", org);
            let id = doc.text("org_name").and_then(|n| db.org_named(n)).map(|o| o.id.clone());
            if let Some(id) = id {
                correct_from(doc, "program", "사업명", |v| db.similar_program(&id, v).map(|p| p.program.clone()));
            }
        }
        "통장사본" => correct_from(doc, "holder", "예금주", org),
        "건축물대장" => correct_from(doc, "address", "대지위치", |v| db.similar_address(v).map(|b| b.address.clone())),
        "소방안전필증" => correct_from(doc, "address", "소재지", |v| db.similar_address(v).map(|b| b.address.clone())),
        "영수증철" => correct_receipts(doc, db),
        _ => {}
    }
}

/// 영수증은 승인번호가 열쇠입니다. 카드 원장에 있으면 거래처 이름만 원장 값으로 바꿉니다. 금액 대조는 `evidence` 가 합니다.
fn correct_receipts(doc: &mut Found, db: &Lattice) {
    let Some(rows) = doc.fields.get_mut("receipts").and_then(Value::as_array_mut) else {
        return;
    };
    let mut cleared = Vec::new();
    for (i, r) in rows.iter_mut().enumerate() {
        let Some(row) = r["approval"].as_str().and_then(|a| db.card_by_approval(a)) else {
            if let Some(v) = r["vendor"].as_str().and_then(|v| db.similar_vendor(v)) {
                r["vendor"] = json!(v.name);
            }
            continue;
        };
        if let Some(v) = db.vendor(&row.vendor) {
            r["vendor"] = json!(v.name);
        }
        r["date"] = json!(row.date);
        r["time"] = json!(row.time);
        r["item"] = json!(row.item);
        // 금액은 대조에 쓰이므로 서류 값을 둡니다. 못 읽었을 때만 원장 값을 씁니다.
        let read = r["amount"].as_str().and_then(|a| a.parse::<i64>().ok());
        if read.is_none() {
            r["amount"] = json!(row.amount.to_string());
        }
        if read.is_none_or(|a| a == row.amount) {
            cleared.push(format!("영수증 {}", i + 1));
        }
    }
    doc.low.retain(|l| !cleared.contains(l));
}

/// 민원 종류 표. 같은 수이면 정하지 않고 민원인에게 묻습니다.
fn guess_category(docs: &[Found]) -> Option<Category> {
    let order = [Category::Welfare, Category::Subsidy, Category::Permit, Category::Cert, Category::Grievance];
    let count = |c: Category| docs.iter().filter(|d| FORMS.iter().any(|f| f.kind == d.kind && f.category == Some(c))).count();
    let top = order.iter().map(|c| count(*c)).max()?;
    let winners: Vec<Category> = order.iter().copied().filter(|c| count(*c) == top).collect();
    match (top, winners.as_slice()) {
        (0, _) => None,
        (_, [one]) => Some(*one),
        _ => None,
    }
}

fn prefill(category: Option<Category>, docs: &[Found]) -> Value {
    let doc = |kind: &str| docs.iter().find(|d| d.kind == kind);
    let mut out = Map::new();
    let mut copy = |key: &str, value: Option<&Value>| {
        if let Some(v) = value {
            out.insert(key.to_string(), v.clone());
        }
    };
    let field = |kind: &str, key: &str| doc(kind).and_then(|d| d.fields.get(key));

    let poa = doc("위임장");
    let agent = poa.and_then(|d| d.text("agent_name"));
    let principal = poa.and_then(|d| d.text("principal_name"));
    let cards: Vec<&Found> = docs.iter().filter(|d| d.kind == ID_CARD).collect();
    // 위임장이 있으면 대리인의 신분증이 신청인의 것입니다.
    let card = cards.iter().find(|c| agent.is_some() && c.text("name") == agent).or(cards.iter().find(|c| principal.is_none() || c.text("name") != principal)).copied();
    copy("applicant_name", card.and_then(|c| c.fields.get("name")).or(field("신청서", "name")).or(field("영업신고서", "name")).or(field("위임장", "agent_name")).or(field("진정서", "name")));
    copy("applicant_birth", card.and_then(|c| c.fields.get("birth")).or(field("신청서", "birth")).or(field("영업신고서", "birth")).or(field("위임장", "agent_birth")));
    match category {
        Some(Category::Welfare) => {
            copy("applied_rent", field("신청서", "applied_rent"));
            copy("handwritten_account", field("신청서", "account").or(field("이체내역서", "printed_account")));
        }
        Some(Category::Subsidy) => {
            copy("org_name", field("정산총괄표", "org_name"));
            copy("program", field("정산총괄표", "program"));
            copy("claimed_total", field("정산총괄표", "claimed_total"));
        }
        Some(Category::Permit) => {
            copy("business_name", field("영업신고서", "business_name"));
            copy("business_type", field("영업신고서", "business_type"));
            copy("address", field("영업신고서", "address").or(field("건축물대장", "address")).or(field("소방안전필증", "address")));
            copy("area_m2", field("영업신고서", "area_m2"));
        }
        Some(Category::Cert) => {
            copy("principal_name", field("위임장", "principal_name"));
            copy("principal_birth", field("위임장", "principal_birth"));
            copy("relation", field("위임장", "relation"));
            copy("poa_date", field("위임장", "poa_date"));
            if poa.is_some() {
                out.insert("document".to_string(), json!("인감증명서"));
            }
        }
        Some(Category::Grievance) => {
            copy("body", field("진정서", "text"));
            copy("contact", field("진정서", "contact"));
        }
        None => {}
    }
    Value::Object(out)
}

/// 페이지 묶음을 서류 목록으로 바꿉니다. 같은 서류가 이어지면 (영수증철 둘째 장 등) 한 건으로 묶습니다.
pub fn extract(pages: &[PageText], db: &Lattice) -> Extracted {
    let mut docs: Vec<Found> = Vec::new();
    let mut unknown_pages = 0;
    for page in pages {
        let Some(mut found) = read_page(page) else {
            unknown_pages += 1;
            continue;
        };
        correct(&mut found, db);
        match docs.last_mut() {
            Some(last) if last.kind == found.kind && found.kind == "영수증철" => {
                last.pages += 1;
                if let (Some(a), Some(b)) = (last.fields.get_mut("receipts").and_then(Value::as_array_mut), found.fields.get("receipts").and_then(Value::as_array)) {
                    a.extend(b.iter().cloned());
                }
                last.low.extend(found.low);
            }
            _ => docs.push(found),
        }
    }
    let category = guess_category(&docs);
    let unsure = docs.iter().flat_map(|d| d.unsure.iter().cloned()).collect();
    Extracted { category, prefill: prefill(category, &docs), docs: docs.iter().map(Found::to_value).collect(), unknown_pages, unsure }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crop(title: bool, text: &str) -> PageCrop {
        PageCrop { title, bbox: [100, 100, 900, 200], text: text.to_string(), confs: vec![0.99; text.chars().count()] }
    }

    fn page(hash: &str, crops: Vec<PageCrop>) -> PageText {
        PageText { hash: hash.to_string(), crops, signs: Vec::new() }
    }

    fn db() -> Lattice {
        Lattice::embedded().unwrap()
    }

    #[test]
    fn numbers_and_dates_take_the_standard_shape() {
        assert_eq!(money("금 1,000,000 원정").as_deref(), Some("1000000"));
        assert_eq!(money("750,000원").as_deref(), Some("750000"));
        assert_eq!(money("750.000 원").as_deref(), Some("750000"), "쉼표가 점으로 읽혀도 자리는 맞습니다");
        assert_eq!(money("750,00"), None, "자리를 잃은 금액은 믿지 않습니다");
        assert_eq!(date("2026 년 10 월 01 일").as_deref(), Some("2026-10-01"));
        assert_eq!(date("2026.04.01").as_deref(), Some("2026-04-01"));
        assert_eq!(date("제 2023-4094 호"), None);
        assert_eq!(normalize(Shape::Birth, "450505-*****").as_deref(), Some("450505"));
        assert_eq!(normalize(Shape::Birth, "451305-*****"), None, "13월은 없습니다");
        assert_eq!(normalize(Shape::Choice(RELATIONS), "자너").as_deref(), Some("자녀"));
    }

    #[test]
    fn table_rows_keep_cells_and_the_lowest_character_confidence() {
        let text = "<table><tr><td rowspan=\"3\">대리인</td><td>성명</td><td>정현우</td></tr><tr><td></td><td></td></tr></table>";
        let mut c = crop(false, text);
        let at = text.find("정현우").map(|b| text[..b].chars().count()).unwrap();
        c.confs[at + 1] = 0.3;
        let rows = rows_of(&c);
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].cells.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["대리인", "성명", "정현우"]);
        assert!((rows[0].cells[2].conf - 0.3).abs() < 1e-6);
        assert!(rows[0].cells[1].conf > 0.9, "다른 칸은 그대로입니다");
    }

    #[test]
    fn a_colon_splits_a_label_but_not_a_time() {
        let rows = rows_of(&crop(false, "상호: 행복식당\n2026-10-05 10:12"));
        assert_eq!(rows[0].cells.iter().map(|c| c.text.as_str()).collect::<Vec<_>>(), ["상호", "행복식당"]);
        assert_eq!(rows[1].cells[0].text, "2026-10-05 10:12");
    }

    #[test]
    fn a_form_is_named_by_its_title_and_not_by_sentences_in_the_body() {
        let report = page("h", vec![crop(true, "영업 신고서"), crop(false, "첨부서류: 1. 교육이수증 2. 소방 · 안전시설 완비증명서 1부")]);
        assert_eq!(read_page(&report).unwrap().kind, "영업신고서");
        let ledger = page("h", vec![crop(true, "일반건축물대장 (갑)"), crop(false, "이 등본은 건축물대장의 원본 내용과 틀림없음을 증명합니다.")]);
        assert_eq!(read_page(&ledger).unwrap().kind, "건축물대장");
        let misread = page("h", vec![crop(true, "보조사업 정산보고서 충괄표")]);
        assert_eq!(read_page(&misread).unwrap().kind, "정산총괄표", "제목 오독 한두 글자는 허용합니다");
        let other = page("h", vec![crop(true, "건강보험료 납입고지서")]);
        assert!(read_page(&other).is_none());
    }

    #[test]
    fn people_on_a_form_are_told_apart_by_their_section_even_across_crops() {
        let p = page(
            "h",
            vec![
                crop(true, "인감증명서 발급 위임장"),
                crop(false, "<table><tr><td rowspan=\"3\">위임자</td><td>성명</td><td>최순자</td></tr><tr><td>주민번호</td><td>450505-*****</td></tr></table>"),
                crop(false, "<table><tr><td rowspan=\"3\">대리인</td><td>성명</td><td>정현우</td></tr><tr><td>주민번호</td><td>750909-*******</td></tr></table>"),
                crop(false, "<table><tr><td>관계</td><td>자녀</td></tr></table>"),
                crop(false, "<table><tr><td>2026 년 10 월 01 일</td></tr><tr><td>위임자: 최순자</td></tr></table>"),
            ],
        );
        let mut p = p;
        for (i, c) in p.crops.iter_mut().enumerate() {
            c.bbox = [100, 100 * (i as u32 + 1), 900, 100 * (i as u32 + 2)];
        }
        let d = read_page(&p).unwrap();
        assert_eq!(d.text("principal_name"), Some("최순자"));
        assert_eq!(d.text("principal_birth"), Some("450505"));
        assert_eq!(d.text("agent_name"), Some("정현우"));
        assert_eq!(d.text("relation"), Some("자녀"));
        assert_eq!(d.text("poa_date"), Some("2026-10-01"));
        assert!(d.low.iter().all(|l| !l.contains("위임자")), "{:?}", d.low);
    }

    #[test]
    fn a_merged_head_cell_read_after_its_first_row_still_owns_that_row() {
        let lines = [
            (100, "<table><tr><td></td><td>성명</td><td>박영희</td></tr></table>"),
            (150, "<table><tr><td>임대인</td><td>주민번호</td><td>640312-*******</td></tr></table>"),
            (300, "<table><tr><td></td><td>성명</td><td>김철수</td></tr></table>"),
            (350, "<table><tr><td>임차인</td><td>주민번호</td><td>990101-*******</td></tr></table>"),
        ];
        let mut crops = vec![crop(true, "주택 임대차 계약서")];
        for (y, text) in lines {
            crops.push(PageCrop { bbox: [100, y, 900, y + 40], ..crop(false, text) });
        }
        let d = read_page(&page("h", crops)).unwrap();
        assert_eq!(d.text("landlord_name"), Some("박영희"));
        assert_eq!(d.text("landlord_birth"), Some("640312"));
        assert_eq!(d.text("tenant_name"), Some("김철수"));
    }

    #[test]
    fn a_head_cell_named_like_a_label_is_not_the_value() {
        let d = read_page(&page("h", vec![crop(true, "보조사업 정산보고서 총괄표"), crop(false, "<table><tr><td>사업</td><td>사업명</td><td>마을 환경 개선 사업</td></tr></table>")])).unwrap();
        assert_eq!(d.text("program"), Some("마을 환경 개선 사업"));
    }

    #[test]
    fn a_statement_form_never_sends_doubts_to_the_officer() {
        let d = read_page(&page("h", vec![crop(true, "청년월세 지원 신청서"), crop(false, "<table><tr><td>월세</td><td></td></tr></table>")])).unwrap();
        assert!(d.low.is_empty());
        assert!(d.unsure.iter().any(|u| u == "신청 월세"), "{:?}", d.unsure);
    }

    #[test]
    fn low_confidence_and_bad_shapes_go_to_a_person_and_are_never_guessed() {
        let mut c = crop(false, "<table><tr><td>차임(월세)</td><td>금 400,000 원정</td></tr></table>");
        let digit = c.text.find("400").map(|b| c.text[..b].chars().count()).unwrap();
        c.confs[digit] = 0.2;
        let d = read_page(&page("h", vec![crop(true, "주택 임대차 계약서"), c])).unwrap();
        assert!(d.low.iter().any(|l| l == "계약 월세"), "{:?}", d.low);
        assert_eq!(d.text("contract_rent"), Some("400000"), "값은 남기되 사람이 확인합니다");

        let bad = crop(false, "<table><tr><td>차임(월세)</td><td>금 40,00 원정</td></tr></table>");
        let d = read_page(&page("h", vec![crop(true, "주택 임대차 계약서"), bad])).unwrap();
        assert!(d.low.iter().any(|l| l == "계약 월세"));
        assert_eq!(d.text("contract_rent"), None, "모양이 틀린 값은 넣지 않습니다");
    }

    #[test]
    fn the_latest_rent_transfer_comes_from_the_rows_marked_as_rent() {
        let rows = [
            "<table><tr><td>거래일자</td><td>적요</td><td>출금</td><td>입금</td><td>잔액</td></tr><tr><td>2026.09.12</td><td>월세 이체</td><td>500,000</td><td>0</td><td>1,000,000</td></tr></table>",
            "<table><tr><td>2026.10.03</td><td>통신요금</td><td>70,000</td><td>0</td><td>930,000</td></tr><tr><td>2026.10.12</td><td>월세 이체</td><td>300,000</td><td>0</td><td>630,000</td></tr></table>",
        ];
        let d = read_page(&page("h", vec![crop(true, "거래내역 확인서 (이체내역서)"), crop(false, rows[0]), crop(false, rows[1])])).unwrap();
        assert_eq!(d.text("latest_transfer"), Some("300000"));
    }

    #[test]
    fn receipts_laid_out_side_by_side_stay_apart() {
        let mut crops = vec![crop(true, "영수증 첨부지")];
        let place = [
            ([50, 200, 350, 260], "<table><tr><td>가명점</td><td>우리인쇄</td></tr></table>"),
            ([600, 200, 900, 260], "<table><tr><td>가맹점</td><td>대한문구</td></tr></table>"),
            ([600, 230, 900, 300], "<table><tr><td>승인번호</td><td>71200988</td></tr></table>"),
            ([50, 280, 350, 340], "<table><tr><td>승인번호</td><td>71200451</td></tr><tr><td>거래일시</td><td>2026-10-05 10:12</td></tr></table>"),
            ([600, 320, 900, 350], "<table><tr><td>거래일시</td><td>2026-10-07 15:40</td></tr></table>"),
            ([50, 360, 350, 420], "<table><tr><td>품목</td><td>홍보물 인쇄</td></tr><tr><td>합계</td><td>320,000원</td></tr></table>"),
            ([600, 360, 900, 420], "<table><tr><td>품목</td><td>사무용품</td></tr><tr><td>합계</td><td>180,000원</td></tr></table>"),
        ];
        for (bbox, text) in place {
            crops.push(PageCrop { bbox, ..crop(false, text) });
        }
        let d = read_page(&page("h", crops)).unwrap();
        let list = d.fields["receipts"].as_array().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0]["vendor"], "우리인쇄");
        assert_eq!(list[0]["amount"], "320000");
        assert_eq!(list[1]["approval"], "71200988");
        assert_eq!(list[1]["time"], "15:40");
        assert!(d.low.is_empty(), "{:?}", d.low);
    }

    #[test]
    fn an_id_card_takes_the_most_confident_of_overlapping_lines() {
        let mut low = crop(false, "처순자");
        low.confs = vec![0.34; 3];
        low.bbox = [100, 100, 900, 150];
        let mut good = crop(false, "최순자");
        good.confs = vec![0.67; 3];
        good.bbox = [100, 140, 900, 190];
        let mut number = crop(false, "450505-********");
        number.bbox = [100, 200, 900, 250];
        let d = read_page(&page("h", vec![crop(true, "주민등록증"), low, good, number])).unwrap();
        assert_eq!(d.text("name"), Some("최순자"));
        assert_eq!(d.text("birth"), Some("450505"));
    }

    #[test]
    fn a_title_read_twice_is_not_taken_for_the_name() {
        let line = |y: u32, text: &str, conf: f32| {
            let mut c = crop(false, text);
            c.bbox = [100, y, 900, y + 40];
            c.confs = vec![conf; text.chars().count()];
            c
        };
        let crops = vec![line(50, "구간중록", 0.9), line(100, "문가영", 0.7), line(150, "문가용", 0.6), line(200, "920521-*******", 0.9)];
        let d = read_page(&page("h", [vec![crop(true, "주민등록증")], crops].concat())).unwrap();
        assert_eq!(d.text("name"), Some("문가영"));
    }

    #[test]
    fn both_readings_of_a_name_are_tried_against_the_registry() {
        let line = |y: u32, text: &str, conf: f32| {
            let mut c = crop(false, text);
            c.bbox = [100, y, 900, y + 40];
            c.confs = vec![conf; text.chars().count()];
            c
        };
        let crops = vec![line(100, "처순자", 0.39), line(150, "회준자", 0.47), line(200, "450505-*******", 0.9)];
        let page = page("h", [vec![crop(true, "주민등록증")], crops].concat());
        let got = extract(&[page], &db());
        assert_eq!(got.docs[0]["fields"]["name"], "최순자");
        assert!(got.docs[0]["low"].as_array().unwrap().is_empty(), "{:?}", got.docs[0]);
        assert!(got.docs[0]["fields"].get(NAME_CANDIDATES).is_none(), "후보는 내보내지 않습니다");
    }

    #[test]
    fn the_registry_fixes_a_misread_only_when_one_entry_is_close() {
        let db = db();
        let mut d = Found { kind: "정산총괄표".into(), hash: "h".into(), pages: 1, signed: true, fields: Map::new(), low: vec!["단체명".into()], unsure: Vec::new() };
        d.fields.insert("org_name".into(), json!("마을공동체험의회"));
        correct(&mut d, &db);
        assert_eq!(d.text("org_name"), Some("마을공동체협의회"));
        assert!(d.low.is_empty());

        let mut d = Found { kind: ID_CARD.into(), hash: "h".into(), pages: 1, signed: false, fields: Map::new(), low: vec!["성명".into()], unsure: Vec::new() };
        d.fields.insert("name".into(), json!("처순자"));
        d.fields.insert("birth".into(), json!("450505"));
        correct(&mut d, &db);
        assert_eq!(d.text("name"), Some("최순자"));

        let mut d = Found { kind: ID_CARD.into(), hash: "h".into(), pages: 1, signed: false, fields: Map::new(), low: vec!["성명".into()], unsure: Vec::new() };
        d.fields.insert("name".into(), json!("완전다름"));
        d.fields.insert("birth".into(), json!("010101"));
        correct(&mut d, &db);
        assert_eq!(d.text("name"), Some("완전다름"), "가까운 값이 없으면 고치지 않습니다");
        assert_eq!(d.low, vec!["성명".to_string()]);
    }

    #[test]
    fn the_paper_set_decides_the_category_and_fills_the_form() {
        let mut lease = crop(false, "<table><tr><td>차임(월세)</td><td>금 400,000 원정</td></tr></table>");
        lease.bbox = [100, 300, 900, 400];
        let pages = vec![
            page("a", vec![crop(true, "주택 임대차 계약서"), lease]),
            page("b", vec![crop(true, "청년월세 지원 신청서"), crop(false, "<table><tr><td>신청인</td><td>성명</td><td>이서연</td></tr><tr><td>생년월일</td><td>000315</td></tr></table>")]),
            page("c", vec![crop(true, "건강보험료 납입고지서")]),
        ];
        let got = extract(&pages, &db());
        assert_eq!(got.category, Some(Category::Welfare));
        assert_eq!(got.unknown_pages, 1);
        assert_eq!(got.prefill["applicant_name"], "이서연");
        assert_eq!(got.docs.len(), 2);
    }

    #[test]
    fn a_grievance_reads_the_writer_and_the_body() {
        let crops = vec![
            PageCrop { bbox: [100, 100, 900, 150], ..crop(true, "진정서") },
            PageCrop { bbox: [100, 200, 900, 300], ..crop(false, "<table><tr><td>진정인 오서연</td><td>010-2345-6789</td></tr><tr><td>경기도 수원시 팔달구 효원로 194, 1367호</td><td></td></tr></table>") },
            PageCrop { bbox: [100, 310, 900, 500], ..crop(false, "<table><tr><td>공사 소음이 이어집니다.</td></tr><tr><td>방음벽을 설치해 주시기 바랍니다.</td></tr></table>") },
            PageCrop { bbox: [100, 510, 900, 600], ..crop(false, "<table><tr><td>2026 년 10 월 31 일</td></tr><tr><td>진정인 : 오서연</td></tr></table>") },
        ];
        let d = read_page(&page("h", crops)).unwrap();
        assert_eq!(d.text("name"), Some("오서연"));
        assert_eq!(d.text("contact"), Some("010-2345-6789"));
        assert_eq!(d.text("text"), Some("공사 소음이 이어집니다. 방음벽을 설치해 주시기 바랍니다."));
    }
}
