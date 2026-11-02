//! N:N 시나리오 읽기와 키오스크 입력 만들기. 규칙 층 시험과 실제 모델 평가가 함께 씁니다.

#![allow(dead_code)]

use serde_json::{json, Map, Value};
use zorogravity_core::case::Category;

pub const SCENARIOS: &str = include_str!("../../fixtures/scenarios.json");

pub fn scenarios() -> Vec<Value> {
    let v: Value = serde_json::from_str(SCENARIOS).unwrap();
    v["scenarios"].as_array().unwrap().clone()
}

pub fn modes() -> Vec<String> {
    let v: Value = serde_json::from_str(SCENARIOS).unwrap();
    v["modes"].as_array().unwrap().iter().map(|m| m.as_str().unwrap().to_string()).collect()
}

pub fn category(s: &Value) -> Category {
    match s["category"].as_str().unwrap() {
        "welfare" => Category::Welfare,
        "subsidy" => Category::Subsidy,
        "permit" => Category::Permit,
        "cert" => Category::Cert,
        "grievance" => Category::Grievance,
        other => panic!("{other}"),
    }
}

pub fn scan(s: &Value, kind: &str, doc: &Value) -> Value {
    let real = doc.get("as").and_then(Value::as_str).unwrap_or(kind);
    json!({
        "kind": real,
        "hash": format!("{}-{kind}", s["id"].as_str().unwrap()),
        "signed": doc.get("signed").and_then(Value::as_bool).unwrap_or(false),
        "fields": doc["fields"].clone(),
    })
}

/// 서류 상태별로 스캐너에 넣은 서류. 신분증은 뒤에 붙습니다.
pub fn paper_names(s: &Value, mode: &str) -> Vec<String> {
    let docs = s["docs"].as_object().unwrap();
    let keep = match mode {
        "none" | "id_only" => 0,
        "partial" => docs.len().div_ceil(2),
        "full" => docs.len(),
        other => panic!("{other}"),
    };
    let mut names: Vec<String> = docs.keys().take(keep).cloned().collect();
    if mode != "none" {
        names.push("신분증".to_string());
    }
    names
}

/// 시나리오와 서류 상태로 키오스크 입력을 만듭니다.
pub fn kiosk_values(s: &Value, mode: &str, excuse: Option<&str>) -> Value {
    let mut v: Map<String, Value> = s["form"].as_object().unwrap().clone();
    v.insert("applicant_name".into(), s["applicant"]["name"].clone());
    v.insert("applicant_birth".into(), s["applicant"]["birth"].clone());
    v.insert("consent".into(), json!(true));
    v.insert("signature".into(), json!({ "stroke_count": 8, "bbox_area_ratio": 0.42, "single_line": false }));
    if let Some(e) = excuse {
        v.insert("excuse".into(), json!(e));
    }
    let docs = s["docs"].as_object().unwrap();
    let keep = match mode {
        "none" | "id_only" => 0,
        "partial" => docs.len().div_ceil(2),
        "full" => docs.len(),
        other => panic!("{other}"),
    };
    let mut scanned: Vec<Value> = docs.iter().take(keep).map(|(k, d)| scan(s, k, d)).collect();
    if mode != "none" {
        scanned.push(json!({ "kind": "신분증", "hash": format!("{}-id", s["id"].as_str().unwrap()), "fields": { "name": s["applicant"]["name"], "birth": s["applicant"]["birth"] } }));
    }
    v.insert("scanned".into(), Value::Array(scanned));
    Value::Object(v)
}

