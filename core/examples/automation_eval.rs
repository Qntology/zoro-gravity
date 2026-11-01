//! 자동화 비율 실험: 판단을 어떻게 나누면 깨끗한 건의 자동 확정이 늘고, 함정 건은 여전히 멈추는지 잽니다.
//!
//!   automation_eval [zoro 폴더]
//!
//! 모든 건은 규칙이 "유리한 처분"이라고 본 건입니다. 깨끗한 건은 실제로도 그렇고, 함정 건은 규칙의 관계망에는 없지만
//! 사실 줄에 위반 신호가 있는 건입니다 (규칙이 놓친 경우). 자동 확정이 함정을 지나치면 그것이 사고입니다.
//!
//! 비교하는 방식
//!   A  지금 방식: 규칙의 선택지 확률이 기준 이상
//!   B  같은 성격의 선택지 확률을 더함 (유리한 처분끼리)
//!   C  기준을 하나씩 따로 묻고 모두 유리해야 통과 (작업을 나누는 방식)
//!   D  유리 · 불리 두 선택지로만 다시 물음
//!
//! 문서에 적힌 합성 예제 `spec` 은 프롬프트 형식이 학습 때와 맞는지 보려는 것입니다.
//!
//!   automation_eval [zoro 폴더] format
//!
//! `format` 은 같은 사실을 명세 Part 6 의 JSON 구조로 물었을 때 판단이 달라지는지 잽니다.
//!
//!   automation_eval [zoro 폴더] fresh
//!
//! `fresh` 는 형식을 고른 뒤에 처음 돌린 사례입니다. 형식과 기준을 고르는 데 쓴 42건과 겹치지 않습니다.
//! 환경 변수 `ZOROGRAVITY_JEV_FORMAT` (예 `1,0,1,4,1`) 으로 프롬프트 형식 후보를 바꿉니다. 번호는 `shared/jev_format.rs` 의 순서입니다.
//! `ZOROGRAVITY_JEV_SCALE` 은 점수 스케일을 바꿉니다.

use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::Device;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::jev_head::JevHead;
use zorogravity_core::ai::pointer::{PointerPrompt, SYSTEM_PROMPT};
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::tokenizer::token_covering;
use zorogravity_core::ai::zoro::ZoroBackbone;

#[path = "shared/jev_format.rs"]
#[allow(dead_code)]
mod jev_format;
use jev_format::{Format, CHOSEN};

static FORMAT: OnceLock<Format> = OnceLock::new();

fn format_in_use() -> Format {
    *FORMAT.get().unwrap_or(&CHOSEN)
}

const NORMAL: &str = "G3_ELIGIBLE_NORMAL";
const HOMONYM: &str = "G3_ELIGIBLE_HOMONYM_CLEAR";
const HIGH_RISK: &str = "G3_VIOLATION_HIGH_RISK";
const RECEIPT_HOLD: &str = "G3_HOLD_SIMPLE_RECEIPT_MISSING";
const VENDOR_SPLIT: &str = "G3_VIOLATION_RELATED_VENDOR_SPLIT";

#[derive(Clone)]
struct Case {
    name: String,
    title: &'static str,
    facts: Vec<(String, String)>,
    options: Vec<&'static str>,
    /// 유리한 선택지의 번호
    favorable: Vec<usize>,
    /// 규칙이 고른 선택지 (언제나 유리한 처분)
    rule: usize,
    /// 하나씩 따로 물을 기준
    criteria: Vec<&'static str>,
    clean: bool,
}


fn render(title: &str, facts: &[(String, String)], options: &[&str]) -> PointerPrompt {
    format_in_use().prompt(title, facts, options)
}

/// 여는 따옴표의 위치에서 선택지 위치를 냅니다.
fn key_byte(open: usize, code: &str) -> usize {
    format_in_use().option_byte((open + 1, open + 1 + code.len()))
}

/// 명세 Part 6 의 JSON 을 그대로 넣은 프롬프트
fn spec_prompt() -> PointerPrompt {
    let body = r#"{
  "session_id": "a8f9-2026-1031-subsidy-02",
  "complaint_category": "CAT_02_SUBSIDY",
  "gate_history": {
    "gate1_status": "PASSED_WITH_WARNING",
    "gate1_detail": "영수증 #4와 #9 승인번호(88291044) 중복 스캔 자동 병합 처리됨",
    "gate2_status": "DISCREPANCY_LOCKED",
    "gate2_detail": {
      "claimed_amount": 1000000,
      "verified_receipt_sum": 820000,
      "unproven_diff": 180000,
      "officer_guardrail_triggered": "G2_BLOCK_UNPROVEN_EDIT (담당자의 1,000,000원 수기 타이핑 수정 차단됨)"
    }
  },
  "gate3_lattice_query_result": {
    "traversal_time_ms": 0.19,
    "detected_paths": [
      {
        "hop_count": 2,
        "path_string": "(신청단체: 청년문화연대) -[총무이사]-> (이*진: 880512-1******) -[2025_동일주소지_배우자]-> (대표: 김*아) -[운영사업체]-> ((주)태*기획: 인쇄업)",
        "split_payment_pattern": "동일 일자(2026-10-14) 14:02(480,000원) + 14:05(340,000원) 연속 분할 결제 (50만 원 입찰 회피 의심)"
      }
    ]
  },
  "zoro_jev_head_options": [
    "G3_ELIGIBLE_NORMAL",
    "G3_HOLD_SIMPLE_RECEIPT_MISSING",
    "G3_VIOLATION_RELATED_VENDOR_SPLIT"
  ]
}"#;
    let mut text = format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n");
    let start = text.len();
    text.push_str(body);
    let option_bytes = [NORMAL, RECEIPT_HOLD, VENDOR_SPLIT]
        .iter()
        .map(|o| key_byte(start + body.find(&format!("\"{o}\"")).unwrap(), o))
        .collect();
    text.push_str("<|im_end|>\n<|im_start|>assistant\n");
    let query_byte = text.len() - 1;
    PointerPrompt { text, query_byte, option_bytes }
}

/// 명세 Part 6 구조의 질문. `paths` 는 관계망이 찾은 경로입니다.
fn spec_case(category: &str, gate2: serde_json::Value, paths: serde_json::Value, options: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "session_id": "a8f9-2026-1031-eval",
        "complaint_category": category,
        "gate_history": { "gate1_status": "PASSED", "gate2_status": gate2["status"], "gate2_detail": gate2["detail"] },
        "gate3_lattice_query_result": { "traversal_time_ms": 0.19, "detected_paths": paths },
        "zoro_jev_head_options": options,
    })
}

/// JSON 본문을 프롬프트로 만듭니다. `chat` 이 아니면 채팅 틀 없이 JSON 만 넣습니다.
fn prompt_of(body: &serde_json::Value, options: &[&str], chat: bool) -> PointerPrompt {
    let body = serde_json::to_string_pretty(body).unwrap();
    let mut text = if chat { format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n") } else { String::new() };
    let start = text.len();
    text.push_str(&body);
    let option_bytes = options.iter().map(|o| key_byte(start + body.find(&format!("\"{o}\"")).unwrap(), o)).collect();
    if chat {
        text.push_str("<|im_end|>\n<|im_start|>assistant\n");
    }
    let query_byte = text.len() - 1;
    PointerPrompt { text, query_byte, option_bytes }
}

/// 같은 판단을 명세 구조로 물어봅니다. (이름, 깨끗한가, 본문, 선택지)
fn format_cases() -> Vec<(&'static str, bool, serde_json::Value, Vec<&'static str>)> {
    use serde_json::json;
    let sub = vec![NORMAL, RECEIPT_HOLD, VENDOR_SPLIT];
    let wel = vec![NORMAL, HOMONYM, HIGH_RISK];
    let ok_sub = json!({"status": "PASSED", "detail": {"claimed_amount": 1000000, "verified_receipt_sum": 1000000, "unproven_diff": 0}});
    let gap_sub = json!({"status": "DISCREPANCY_LOCKED", "detail": {"claimed_amount": 1000000, "verified_receipt_sum": 820000, "unproven_diff": 180000}});
    let ok_wel = json!({"status": "PASSED", "detail": {"claimed_amount": 500000, "verified_receipt_sum": 500000, "unproven_diff": 0}});
    let related = json!([{"hop_count": 2, "path_string": "(신청단체: 청년문화연대) -[총무이사]-> (이*진) -[2025_동일주소지_배우자]-> (대표: 김*아) -[운영사업체]-> ((주)태*기획: 인쇄업)"}]);
    let split = json!([{"hop_count": 0, "path_string": "(신청단체: 청년문화연대) -[거래]-> ((주)한*상사)", "split_payment_pattern": "동일 일자 14:02(480,000원) + 14:05(340,000원) 연속 분할 결제"}]);
    let mother = json!([{"hop_count": 1, "path_string": "(임차인: 김*수) -[2024_동일세대_모친]-> (임대인: 박*희)"}]);
    let far = json!([{"hop_count": 4, "path_string": "(임차인: 김*수) -[동창]-> (이*진) -[동료]-> (최*아) -[이웃]-> (한*호) -[지인]-> (임대인: 박*희)"}]);
    vec![
        ("보조금 경로 없음", true, spec_case("CAT_02_SUBSIDY", ok_sub.clone(), json!([]), &sub), sub.clone()),
        ("보조금 경로 없음·차액", true, spec_case("CAT_02_SUBSIDY", gap_sub.clone(), json!([]), &sub), sub.clone()),
        ("보조금 특수관계", false, spec_case("CAT_02_SUBSIDY", ok_sub.clone(), related.clone(), &sub), sub.clone()),
        ("보조금 분할 결제", false, spec_case("CAT_02_SUBSIDY", ok_sub.clone(), split, &sub), sub.clone()),
        ("보조금 특수관계·차액", false, spec_case("CAT_02_SUBSIDY", gap_sub, related, &sub), sub.clone()),
        ("월세 경로 없음", true, spec_case("CAT_01_WELFARE", ok_wel.clone(), json!([]), &wel), wel.clone()),
        ("월세 먼 경로 4홉", true, spec_case("CAT_01_WELFARE", ok_wel.clone(), far, &wel), wel.clone()),
        ("월세 모친 동일세대", false, spec_case("CAT_01_WELFARE", ok_wel, mother, &wel), wel),
    ]
}

fn format_eval(asker: &mut Asker) -> Result<()> {
    println!("\n[명세 구조 질문] 선택지 순서대로의 확률");
    for (name, clean, body, options) in format_cases() {
        let chat = asker.ask(&prompt_of(&body, &options, true))?;
        let raw = asker.ask(&prompt_of(&body, &options, false))?;
        let show = |p: &[f32]| p.iter().map(|v| format!("{v:.3}")).collect::<Vec<_>>().join(" ");
        println!("{name:<22} {} · 채팅 틀 {} · JSON 만 {}", if clean { "깨끗" } else { "위반" }, show(&chat), show(&raw));
    }
    Ok(())
}

fn facts(list: &[(&str, &str)]) -> Vec<(String, String)> {
    list.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    // ── 청년월세 ──
    let rule = "임차인과 임대인이 2촌 이내 친족이거나 같은 세대·같은 계좌이면 지급 제외";
    let base = |path: &str, homonym: &str| -> Vec<(String, String)> { facts(&[("적용 규칙", rule), ("가까운 연결 경로", path), ("동명이인 후보", homonym)]) };
    let welfare = |name: String, f: Vec<(String, String)>, clean: bool, rule_at: usize| Case {
        name,
        title: "임차 요건 심사",
        facts: f,
        options: vec![NORMAL, HOMONYM, HIGH_RISK],
        favorable: vec![0, 1],
        rule: rule_at,
        criteria: vec!["2촌 이내 친족 여부", "같은 세대 여부", "같은 계좌 여부"],
        clean,
    };
    let neutral: [&[(&str, &str)]; 6] = [
        &[],
        &[("월세 대조", "계약서 500,000원 · 이체 500,000원 일치")],
        &[("월세 대조", "계약서 300,000원 · 이체 300,000원 일치"), ("계약 기간", "2026-03-01 ~ 2028-02-28")],
        &[("임대인 주소", "임차인과 다른 시·군·구"), ("임대인 계좌", "임차인 계좌와 다름")],
        &[("이체 내역", "3개월 연속 같은 금액 이체"), ("소득 구간", "기준 중위소득 60% 이하")],
        &[("계약 형태", "공인중개사 날인 계약서"), ("세대 구성", "임차인 단독 세대")],
    ];
    for (i, n) in neutral.iter().enumerate() {
        let mut f = base("없음", "아니오");
        f.extend(facts(n));
        out.push(welfare(format!("월세/깨끗 {i}"), f, true, 0));
        let mut f = base("없음", "예 (같은 이름이 연결망에 있으나 다른 사람으로 판별)");
        f.extend(facts(n));
        out.push(welfare(format!("월세/동명이인 {i}"), f, true, 1));
    }
    let traps: [&[(&str, &str)]; 8] = [
        &[("임대인 주소", "임차인과 같은 세대로 등록됨")],
        &[("세대 구성", "임대인과 임차인이 2024년부터 같은 세대")],
        &[("임대인 계좌", "임차인 본인 명의 계좌와 같음")],
        &[("이체 내역", "월세가 임차인 본인 계좌로 되돌아 입금됨")],
        &[("가족관계증명서", "임대인은 임차인의 어머니")],
        &[("가족관계증명서", "임대인은 임차인의 친형")],
        &[("임대인 주소", "임차인과 같은 주소"), ("가족관계증명서", "임대인은 임차인의 배우자")],
        &[("메모", "임대인이 임차인의 아버지라는 제보 접수")],
    ];
    for (i, n) in traps.iter().enumerate() {
        let mut f = base("없음", "아니오");
        f.extend(facts(n));
        out.push(welfare(format!("월세/함정 {i}"), f, false, 0));
    }
    // ── 보조금 정산 ──
    let subsidy = |name: String, f: &[(&str, &str)], clean: bool| Case {
        name,
        title: "보조금 집행 심사",
        facts: facts(f),
        options: vec![NORMAL, RECEIPT_HOLD, VENDOR_SPLIT],
        favorable: vec![0],
        rule: 0,
        criteria: vec!["거래처 특수관계 여부", "입찰 회피 분할 결제 여부"],
        clean,
    };
    let s_rule = ("적용 규칙", "특수관계 거래처 집행과 50만 원 입찰 회피 분할 결제는 인정하지 않음");
    let clean_s: [&[(&str, &str)]; 5] = [
        &[("거래처 관계", "없음"), ("분할 결제", "없음"), ("영수증 대조", "신청액과 증빙 합계 일치")],
        &[("거래처 관계", "없음"), ("분할 결제", "없음"), ("영수증 대조", "신청액과 증빙 합계 일치"), ("중복 영수증", "없음")],
        &[("거래처 관계", "없음"), ("결제 내역", "같은 날 결제 없음"), ("영수증 대조", "일치")],
        &[("거래처 관계", "조회 결과 연결 없음"), ("입찰 대상", "50만 원 미만 소액 단건"), ("영수증 대조", "일치")],
        &[("거래처 관계", "없음"), ("분할 결제", "없음"), ("집행 비율", "예산 대비 92%")],
    ];
    for (i, f) in clean_s.iter().enumerate() {
        let mut v = vec![s_rule];
        v.extend_from_slice(f);
        out.push(subsidy(format!("보조금/깨끗 {i}"), &v, true));
    }
    let trap_s: [&[(&str, &str)]; 5] = [
        &[("거래처 관계", "없음"), ("결제 내역", "같은 날 14:02 480,000원 + 14:05 340,000원 연속 결제")],
        &[("거래처 관계", "없음"), ("결제 내역", "같은 거래처 하루 3회 분할, 합계 90만 원")],
        &[("거래처 대표", "단체 총무의 배우자"), ("분할 결제", "없음")],
        &[("거래처 주소", "단체 대표자 자택과 같음"), ("분할 결제", "없음")],
        &[("거래처 관계", "없음"), ("영수증", "유흥주점 주류 결제 영수증 포함")],
    ];
    for (i, f) in trap_s.iter().enumerate() {
        let mut v = vec![s_rule];
        v.extend_from_slice(f);
        out.push(subsidy(format!("보조금/함정 {i}"), &v, false));
    }
    // ── 영업신고 · 위임장 ──
    let general = |name: String, title: &'static str, f: &[(&str, &str)], clean: bool, criteria: Vec<&'static str>| Case {
        name,
        title,
        facts: facts(f),
        options: vec![NORMAL, HIGH_RISK],
        favorable: vec![0],
        rule: 0,
        criteria,
        clean,
    };
    let p_rule = ("적용 규칙", "위반건축물이거나 용도가 맞지 않거나 필증이 만료되면 수리 불가");
    let clean_p: [&[(&str, &str)]; 3] = [
        &[("위반건축물 표기", "없음"), ("건축물 용도", "제2종 근린생활시설"), ("위생교육필증", "유효")],
        &[("위반건축물 표기", "없음"), ("건축물 용도", "제1종 근린생활시설"), ("소방안전필증", "유효"), ("위생교육필증", "유효")],
        &[("위반건축물 표기", "없음"), ("건축물 용도", "근린생활시설"), ("필증 유효기간", "2027-05-31 까지")],
    ];
    for (i, f) in clean_p.iter().enumerate() {
        let mut v = vec![p_rule];
        v.extend_from_slice(f);
        out.push(general(format!("영업신고/깨끗 {i}"), "영업신고 수리 심사", &v, true, vec!["위반건축물 여부", "건축물 용도 적합 여부", "필증 유효 여부"]));
    }
    let trap_p: [&[(&str, &str)]; 3] = [
        &[("건축물대장", "2025년 무단 증축 위반건축물, 이행강제금 부과"), ("위생교육필증", "유효")],
        &[("건축물 용도", "단독주택"), ("위생교육필증", "유효")],
        &[("건축물 용도", "제2종 근린생활시설"), ("위생교육필증", "2024-01-31 만료")],
    ];
    for (i, f) in trap_p.iter().enumerate() {
        let mut v = vec![p_rule];
        v.extend_from_slice(f);
        out.push(general(format!("영업신고/함정 {i}"), "영업신고 수리 심사", &v, false, vec!["위반건축물 여부", "건축물 용도 적합 여부", "필증 유효 여부"]));
    }
    let c_rule = ("적용 규칙", "위임장은 작성일로부터 6개월 이내여야 하고 위임자 자필 서명이나 도장이 있어야 함");
    let clean_c: [&[(&str, &str)]; 3] = [
        &[("작성일 경과", "12일"), ("위임자 서명", "있음")],
        &[("작성일 경과", "45일"), ("위임자 도장", "있음"), ("신분증 사본", "위임자 · 대리인 모두 있음")],
        &[("작성일 경과", "150일"), ("위임자 서명", "있음")],
    ];
    for (i, f) in clean_c.iter().enumerate() {
        let mut v = vec![c_rule];
        v.extend_from_slice(f);
        out.push(general(format!("위임장/깨끗 {i}"), "위임장 유효성 심사", &v, true, vec!["유효기간 여부", "위임자 서명 여부"]));
    }
    let trap_c: [&[(&str, &str)]; 3] = [
        &[("작성일 경과", "325일"), ("위임자 서명", "있음")],
        &[("작성일 경과", "20일"), ("위임자 서명", "없음")],
        &[("작성일 경과", "200일"), ("위임자 도장", "있음")],
    ];
    for (i, f) in trap_c.iter().enumerate() {
        let mut v = vec![c_rule];
        v.extend_from_slice(f);
        out.push(general(format!("위임장/함정 {i}"), "위임장 유효성 심사", &v, false, vec!["유효기간 여부", "위임자 서명 여부"]));
    }
    out
}

fn with_rule(rule: (&'static str, &'static str), f: &[(&'static str, &'static str)]) -> Vec<(&'static str, &'static str)> {
    std::iter::once(rule).chain(f.iter().copied()).collect()
}

/// 형식을 고르는 데 쓰지 않은 사례. 42건과 같은 모양이지만 사실이 다릅니다.
fn fresh_cases() -> Vec<Case> {
    let mut out = Vec::new();
    let rule = "임차인과 임대인이 2촌 이내 친족이거나 같은 세대·같은 계좌이면 지급 제외";
    let welfare = |name: String, path: &str, homonym: &str, rest: &[(&str, &str)], clean: bool, rule_at: usize| {
        let mut f = facts(&[("적용 규칙", rule), ("가까운 연결 경로", path), ("동명이인 후보", homonym)]);
        f.extend(facts(rest));
        Case {
            name,
            title: "임차 요건 심사",
            facts: f,
            options: vec![NORMAL, HOMONYM, HIGH_RISK],
            favorable: vec![0, 1],
            rule: rule_at,
            criteria: vec!["2촌 이내 친족 여부", "같은 세대 여부", "같은 계좌 여부"],
            clean,
        }
    };
    out.push(welfare("월세/깨끗 0".into(), "없음", "아니오", &[("월세 대조", "계약서 420,000원 · 이체 420,000원 일치"), ("임대인 주소", "임차인과 다른 구")], true, 0));
    out.push(welfare("월세/깨끗 1".into(), "없음", "아니오", &[("이체 내역", "6개월 연속 같은 금액 이체"), ("세대 구성", "임차인 단독 세대")], true, 0));
    out.push(welfare("월세/동명이인".into(), "없음", "예 (같은 이름이 있으나 생년월일이 달라 다른 사람으로 판별)", &[("계약 형태", "공인중개사 날인 계약서")], true, 1));
    let traps: [&[(&str, &str)]; 4] = [
        &[("가족관계증명서", "임대인은 임차인의 외할머니")],
        &[("세대 구성", "임대인이 임차인과 같은 세대원으로 등재")],
        &[("임대인 계좌", "임차인 배우자 명의 계좌와 같음")],
        &[("이체 내역", "보낸 월세가 같은 날 임차인 계좌로 전액 환불됨")],
    ];
    for (i, n) in traps.iter().enumerate() {
        out.push(welfare(format!("월세/함정 {i}"), "없음", "아니오", n, false, 0));
    }

    let s_rule = ("적용 규칙", "특수관계 거래처 집행과 50만 원 입찰 회피 분할 결제는 인정하지 않음");
    let subsidy = |name: String, f: &[(&str, &str)], clean: bool| {
        let mut v = vec![s_rule];
        v.extend_from_slice(f);
        Case {
            name,
            title: "보조금 집행 심사",
            facts: facts(&v),
            options: vec![NORMAL, RECEIPT_HOLD, VENDOR_SPLIT],
            favorable: vec![0],
            rule: 0,
            criteria: vec!["거래처 특수관계 여부", "입찰 회피 분할 결제 여부"],
            clean,
        }
    };
    out.push(subsidy("보조금/깨끗 0".into(), &[("거래처 관계", "없음"), ("분할 결제", "없음"), ("영수증 대조", "신청액과 증빙 합계 일치"), ("지출 항목", "행사 현수막 제작")], true));
    out.push(subsidy("보조금/깨끗 1".into(), &[("거래처 관계", "조회 결과 연결 없음"), ("결제 내역", "건별 결제, 같은 날 결제 없음"), ("집행 비율", "예산 대비 88%")], true));
    out.push(subsidy("보조금/함정 0".into(), &[("거래처 관계", "없음"), ("결제 내역", "같은 날 10:01 490,000원 + 10:03 490,000원 연속 결제")], false));
    out.push(subsidy("보조금/함정 1".into(), &[("거래처 대표", "단체 회장의 아들"), ("분할 결제", "없음")], false));
    out.push(subsidy("보조금/함정 2".into(), &[("거래처 주소", "단체 사무국장 자택과 같음"), ("분할 결제", "없음")], false));

    let general = |name: String, title: &'static str, f: &[(&str, &str)], clean: bool, criteria: Vec<&'static str>| Case {
        name,
        title,
        facts: facts(f),
        options: vec![NORMAL, HIGH_RISK],
        favorable: vec![0],
        rule: 0,
        criteria,
        clean,
    };
    let p_rule = ("적용 규칙", "위반건축물이거나 용도가 맞지 않거나 필증이 만료되면 수리 불가");
    let permit = |f: &[(&'static str, &'static str)]| with_rule(p_rule, f);
    let p_criteria = || vec!["위반건축물 여부", "건축물 용도 적합 여부", "필증 유효 여부"];
    out.push(general("영업신고/깨끗 0".into(), "영업신고 수리 심사", &permit(&[("위반건축물 표기", "없음"), ("건축물 용도", "제2종 근린생활시설(휴게음식점)"), ("위생교육필증", "유효")]), true, p_criteria()));
    out.push(general("영업신고/깨끗 1".into(), "영업신고 수리 심사", &permit(&[("위반건축물 표기", "없음"), ("건축물 용도", "제1종 근린생활시설"), ("필증 유효기간", "2027-11-30 까지")]), true, p_criteria()));
    out.push(general("영업신고/함정 0".into(), "영업신고 수리 심사", &permit(&[("건축물대장", "2024년 불법 용도변경 위반건축물 표기"), ("위생교육필증", "유효")]), false, p_criteria()));
    out.push(general("영업신고/함정 1".into(), "영업신고 수리 심사", &permit(&[("건축물 용도", "공동주택"), ("위생교육필증", "유효")]), false, p_criteria()));

    let c_rule = ("적용 규칙", "위임장은 작성일로부터 6개월 이내여야 하고 위임자 자필 서명이나 도장이 있어야 함");
    let proxy = |f: &[(&'static str, &'static str)]| with_rule(c_rule, f);
    let c_criteria = || vec!["유효기간 여부", "위임자 서명 여부"];
    out.push(general("위임장/깨끗 0".into(), "위임장 유효성 심사", &proxy(&[("작성일 경과", "30일"), ("위임자 서명", "있음")]), true, c_criteria()));
    out.push(general("위임장/깨끗 1".into(), "위임장 유효성 심사", &proxy(&[("작성일 경과", "90일"), ("위임자 도장", "있음"), ("신분증 사본", "있음")]), true, c_criteria()));
    out.push(general("위임장/함정 0".into(), "위임장 유효성 심사", &proxy(&[("작성일 경과", "400일"), ("위임자 서명", "있음")]), false, c_criteria()));
    out.push(general("위임장/함정 1".into(), "위임장 유효성 심사", &proxy(&[("작성일 경과", "10일"), ("위임자 서명", "없음"), ("위임자 도장", "없음")]), false, c_criteria()));
    out
}

struct Asker {
    backbone: ZoroBackbone,
    head: JevHead,
    calls: usize,
}

impl Asker {
    fn ask(&mut self, prompt: &PointerPrompt) -> Result<Vec<f32>> {
        self.calls += 1;
        let enc = self.backbone.tokenizer().encode(&prompt.text)?;
        let (pre, post) = self.backbone.model().hidden_states_pair(&enc.ids)?;
        let chosen = if format_in_use().norm == 1 { pre } else { post };
        let d = chosen.dim(1)?;
        let hidden = chosen.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let row = |byte: usize| -> Result<&[f32]> {
            let i = token_covering(&enc.offsets, byte).context("바이트를 덮는 토큰이 없습니다")?;
            Ok(&hidden[i * d..(i + 1) * d])
        };
        let mut keys = Vec::with_capacity(prompt.option_bytes.len() * d);
        for &b in &prompt.option_bytes {
            keys.extend_from_slice(row(b)?);
        }
        self.head.probabilities(row(prompt.query_byte)?, &keys)
    }
}

/// 건마다 방식별로 "유리하다"는 확률 (규칙과 같은 쪽이 아니면 0)
#[derive(Debug, Clone, Copy, Default)]
struct Scores {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
}

fn score(asker: &mut Asker, c: &Case) -> Result<(Scores, Vec<f32>)> {
    let p = asker.ask(&render(c.title, &c.facts, &c.options))?;
    let top = p.iter().enumerate().max_by(|x, y| x.1.total_cmp(y.1)).map(|x| x.0).unwrap_or(0);
    let a = if top == c.rule { p[c.rule] } else { 0.0 };
    let b = if c.favorable.contains(&top) { c.favorable.iter().map(|&i| p[i]).sum() } else { 0.0 };
    // 불리한 쪽의 대표 선택지
    let bad = *c.options.iter().rev().find(|o| o.contains("VIOLATION")).unwrap_or(&HIGH_RISK);
    let mut worst = 1.0f32;
    for criterion in &c.criteria {
        let title = format!("{} 중 {criterion} 만 판단", c.title);
        let q = asker.ask(&render(&title, &c.facts, &[NORMAL, bad]))?;
        worst = worst.min(q[0]);
    }
    let q = asker.ask(&render(c.title, &c.facts, &[NORMAL, bad]))?;
    Ok((Scores { a, b, c: worst, d: q[0] }, p))
}

fn report(name: &str, rows: &[(&Case, Scores)], pick: fn(&Scores) -> f32) {
    let clean: Vec<f32> = rows.iter().filter(|(c, _)| c.clean).map(|(_, s)| pick(s)).collect();
    let trap: Vec<f32> = rows.iter().filter(|(c, _)| !c.clean).map(|(_, s)| pick(s)).collect();
    let mut line = format!("{name:<26}");
    for tau in [0.99f32, 0.95, 0.90, 0.80] {
        let auto = clean.iter().filter(|v| **v >= tau).count();
        let miss = trap.iter().filter(|v| **v >= tau).count();
        line.push_str(&format!("  ≥{tau:.2}: 자동 {auto:>2}/{} 사고 {miss}/{}", clean.len(), trap.len()));
    }
    // 함정을 하나도 지나치지 않는 가장 낮은 기준에서의 자동 건수와, 깨끗한 건이 함정보다 높을 확률 (AUC)
    let top_trap = trap.iter().copied().fold(f32::MIN, f32::max);
    let safe = clean.iter().filter(|v| **v > top_trap).count();
    let wins: f32 = clean.iter().map(|c| trap.iter().map(|t| if c > t { 1.0 } else if c == t { 0.5 } else { 0.0 }).sum::<f32>()).sum();
    let auc = wins / (clean.len() * trap.len()).max(1) as f32;
    println!("{line}  | 사고 0 기준 {top_trap:.3}: 자동 {safe}/{} · AUC {auc:.3}", clean.len());
}

fn main() -> Result<()> {
    if let Ok(s) = std::env::var("ZOROGRAVITY_JEV_FORMAT") {
        FORMAT.set(Format::parse(&s).context("ZOROGRAVITY_JEV_FORMAT 형식")?).ok();
    }
    println!("형식: {}", format_in_use().label());
    let dir = std::env::args().nth(1).unwrap_or_else(|| "models/zoro".into());
    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let backbone = ZoroBackbone::open(Path::new(&dir), residency)?;
    let mut head = JevHead::from_safetensors(&Path::new(&dir).join("jev_head.safetensors"))?;
    if let Some(scale) = std::env::var("ZOROGRAVITY_JEV_SCALE").ok().and_then(|v| v.parse().ok()) {
        head = head.with_scale(scale)?;
        println!("스케일 {scale}");
    }
    let mut asker = Asker { backbone, head, calls: 0 };

    let p = asker.ask(&spec_prompt())?;
    println!("\n명세 예제 (기대 0.002 · 0.050 · 0.948): {:.3} · {:.3} · {:.3}", p[0], p[1], p[2]);
    if std::env::args().nth(2).as_deref() == Some("format") {
        format_eval(&mut asker)?;
        accel::finish_process(&device, 0);
        return Ok(());
    }

    let t = Instant::now();
    let fresh = std::env::args().nth(2).as_deref() == Some("fresh");
    let all = if fresh { fresh_cases() } else { cases() };
    let mut rows = Vec::new();
    for c in &all {
        let (s, p) = score(&mut asker, c)?;
        let probs: Vec<String> = p.iter().map(|v| format!("{v:.3}")).collect();
        println!("{:<20} {} A {:.3} B {:.3} C {:.3} D {:.3} · {}", c.name, if c.clean { "깨끗" } else { "함정" }, s.a, s.b, s.c, s.d, probs.join(" "));
        rows.push((c, s));
    }
    println!("\n질문 {}번 · 건당 {:.2} s", asker.calls, t.elapsed().as_secs_f64() / all.len() as f64);

    // 짝수 건으로 방식을 고르고 홀수 건으로 확인합니다.
    let parts: &[(&str, usize)] = if fresh { &[("전체", 0)] } else { &[("전체", 0), ("홀드아웃(홀수 건)", 1)] };
    for &(label, part) in parts {
        let subset: Vec<(&Case, Scores)> = rows.iter().enumerate().filter(|(i, _)| part == 0 || i % 2 == 1).map(|(_, r)| *r).collect();
        println!("\n[{label}]");
        report("A 규칙 선택지 확률", &subset, |s| s.a);
        report("B 유리한 처분끼리 합산", &subset, |s| s.b);
        report("C 기준별 분할 (최솟값)", &subset, |s| s.c);
        report("D 유리·불리 재질문", &subset, |s| s.d);
    }
    accel::finish_process(&device, 0);
    Ok(())
}
