//! `case_read` 가 저장한 글자에서 항목을 뽑아 정답과 맞춥니다. GPU 가 필요 없습니다.
//!
//!   case_extract <읽기 폴더> <truth.json> [낮은 신뢰도 기준을 바꿔 볼 값들]
//!
//! 서류 종류, 항목 값, 서명 여부를 서식별로 셉니다. 틀린 값이 `low` 에 들어갔으면 사람이 잡는 값이고,
//! `low` 없이 틀렸으면 조용한 오독입니다. 안정화 기준은 조용한 오독 0 입니다.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::Result;
use serde_json::Value;
use zorogravity_core::extract::{extract, PageText};
use zorogravity_core::lattice::{edit_distance, Lattice};

#[derive(Default)]
struct Row {
    pages: usize,
    kind_ok: usize,
    fields: usize,
    right: usize,
    /// 틀렸지만 사람이 확인하도록 표시된 값
    caught: usize,
    /// 틀렸는데 표시도 없는 값
    silent: usize,
    /// 맞았지만 사람에게 넘어간 값
    needless: usize,
    missing: usize,
}

fn digits(s: &str) -> String {
    s.chars().filter(char::is_ascii_alphanumeric).collect()
}

fn same(key: &str, want: &str, got: &str) -> bool {
    let squeeze = |s: &str| s.chars().filter(|c| c.is_alphanumeric()).collect::<String>();
    match key {
        "contract_rent" | "deposit" | "latest_transfer" | "applied_rent" | "claimed_total" | "amount" => digits(want) == digits(got),
        "text" | "violation" => {
            let (a, b) = (squeeze(want), squeeze(got));
            edit_distance(&a, &b) * 10 <= a.chars().count()
        }
        _ => squeeze(want) == squeeze(got),
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(&args[0]);
    let truth: Value = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
    let db = Lattice::embedded()?;

    let mut rows: BTreeMap<String, Row> = BTreeMap::new();
    let mut lines = Vec::new();
    let mut category_ok = 0;
    let mut category_all = 0;
    let (mut sign_ok, mut sign_all) = (0, 0);
    let (mut sign_fp, mut sign_fn) = (0, 0);

    for (case, docs) in truth.as_object().unwrap() {
        let mut pages = Vec::new();
        for (file, _) in docs.as_object().unwrap() {
            let path = dir.join(case).join(format!("{file}.json"));
            let Ok(text) = fs::read_to_string(&path) else { continue };
            pages.push((file.clone(), PageText::from_json(&format!("{case}-{file}"), &serde_json::from_str(&text)?)));
        }
        if pages.is_empty() {
            continue;
        }
        for (file, page) in &pages {
            let want = &docs[file];
            let kind = want["kind"].as_str().unwrap();
            let got = extract(std::slice::from_ref(page), &db);
            let row = rows.entry(kind.to_string()).or_default();
            row.pages += 1;
            let doc = got.docs.first();
            let doc_kind = doc.and_then(|d| d["kind"].as_str()).unwrap_or("모름");
            if doc_kind == kind {
                row.kind_ok += 1;
            } else {
                lines.push(format!("{case} {file}: 종류 {doc_kind}"));
                continue;
            }
            let doc = doc.unwrap();
            if want.get("signed").and_then(Value::as_bool).is_some() && ["위임장", "신청서", "금융정보동의서", "영업신고서", "진정서", "정산총괄표"].contains(&kind) {
                let (w, g) = (want["signed"].as_bool().unwrap(), doc["signed"].as_bool().unwrap_or(false));
                sign_all += 1;
                sign_ok += (w == g) as usize;
                sign_fp += (g && !w) as usize;
                sign_fn += (w && !g) as usize;
            }
            let low: Vec<&str> = doc["low"].as_array().unwrap().iter().filter_map(Value::as_str).collect();
            for (key, value) in want["fields"].as_object().unwrap() {
                if key == "receipts" {
                    let list = doc["fields"]["receipts"].as_array().cloned().unwrap_or_default();
                    for (i, r) in value.as_array().unwrap().iter().enumerate() {
                        for k in ["vendor", "approval", "date", "time", "amount", "item"] {
                            let w = r[k].as_str().unwrap_or("");
                            let found = list.iter().find(|x| x["approval"] == r["approval"]).or(list.get(i));
                            let g = found.and_then(|x| x[k].as_str()).unwrap_or("");
                            let label = format!("영수증 {}", i + 1);
                            row.fields += 1;
                            count(row, &mut lines, &format!("{case} {file} {label} {k}"), k, w, g, low.contains(&label.as_str()));
                        }
                    }
                    continue;
                }
                let w = value.as_str().unwrap_or("");
                let g = doc["fields"][key].as_str().unwrap_or("");
                row.fields += 1;
                let flagged = !low.is_empty();
                count(row, &mut lines, &format!("{case} {file} {key}"), key, w, g, flagged);
            }
            if !low.is_empty() {
                lines.push(format!("{case} {file}: 사람 확인 {low:?}"));
            }
        }
        if case.starts_with("sign-") {
            continue;
        }
        let all: Vec<PageText> = pages.iter().map(|(_, p)| p.clone()).collect();
        let got = extract(&all, &db);
        category_all += 1;
        let want = case.chars().next().unwrap();
        let guess = got.category.map(zorogravity_core::forms::prefix).unwrap_or('?');
        if guess == want {
            category_ok += 1;
        } else {
            lines.push(format!("{case}: 민원 종류 {guess}"));
        }
    }

    println!("{:<14}{:>4}{:>6}{:>7}{:>6}{:>6}{:>6}{:>6}{:>6}", "서식", "장", "종류", "항목", "맞음", "표시", "조용", "과표시", "빠짐");
    let mut total = Row::default();
    for (kind, r) in &rows {
        println!("{kind:<14}{:>4}{:>6}{:>7}{:>6}{:>6}{:>6}{:>6}{:>6}", r.pages, r.kind_ok, r.fields, r.right, r.caught, r.silent, r.needless, r.missing);
        total.pages += r.pages;
        total.kind_ok += r.kind_ok;
        total.fields += r.fields;
        total.right += r.right;
        total.caught += r.caught;
        total.silent += r.silent;
        total.needless += r.needless;
        total.missing += r.missing;
    }
    println!(
        "\n서류 종류 {}/{} · 항목 {}/{} ({:.1}%) · 표시된 오독 {} · 조용한 오독 {} · 과표시 {} · 민원 종류 {category_ok}/{category_all} · 서명 {sign_ok}/{sign_all} (오탐 {sign_fp}, 놓침 {sign_fn})",
        total.kind_ok,
        total.pages,
        total.right,
        total.fields,
        100.0 * total.right as f32 / total.fields.max(1) as f32,
        total.caught,
        total.silent,
        total.needless,
    );
    for l in lines {
        println!("  {l}");
    }
    Ok(())
}

fn count(row: &mut Row, lines: &mut Vec<String>, name: &str, key: &str, want: &str, got: &str, flagged: bool) {
    if same(key, want, got) {
        row.right += 1;
        row.needless += flagged as usize;
        return;
    }
    if got.is_empty() {
        row.missing += 1;
    }
    if flagged {
        row.caught += 1;
    } else {
        row.silent += 1;
    }
    lines.push(format!("{name}: 정답 {want:?} · 읽음 {got:?}{}", if flagged { "" } else { " (조용한 오독)" }));
}
