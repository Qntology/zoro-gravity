//! 판단 경로 점검: 실제 zoro 백본 + 실제 `jev_head` 로 Gate 3 모양의 질문에 선택지별 확률을 냅니다.
//!
//! 확인하는 것
//!
//! 1. 프롬프트의 질의 · 선택지 바이트 위치가 실제 토크나이저에서 의도한 토큰에 닿는가 (코드 문자열의 마지막 글자)
//! 2. 같은 질문은 같은 확률을 내는가 (결정적), 호출당 시간
//! 3. 사실이 바뀌면 확률이 규칙이 말하는 쪽으로 움직이는가
//!
//! 3번은 위치 · 스케일 약속(docs/jev-head.md)이 학습 형식과 맞는지에 대한 간접 증거일 뿐, 증명은 아닙니다.
//!
//!   cargo run -p zorogravity-core --example decide_probe
//!   ZOROGRAVITY_DEVICE=cpu cargo run -p zorogravity-core --example decide_probe

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::jev_head::JevHead;
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::pointer::{Backbone, PointerHidden, PointerModel, PointerPrompt};
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::tokenizer::token_covering;
use zorogravity_core::ai::zoro::ZoroBackbone;
use zorogravity_core::decision::{DecisionModel, DecisionTask, Fact};
use zorogravity_core::state_machine::gate3_substantive::OPTIONS;

/// 진단을 위해 백본을 공유합니다 (`PointerModel` 에 넘기면서 토크나이저도 계속 쓰려고)
struct Shared(Arc<ZoroBackbone>);

impl Backbone for Shared {
    fn encode(&self, prompt: &PointerPrompt) -> Result<PointerHidden> {
        self.0.encode(prompt)
    }
}

fn task(path: &str, homonym: &str) -> DecisionTask {
    DecisionTask {
        title: "임차 요건 심사".into(),
        facts: vec![
            Fact::new("적용 규칙", "임차인과 임대인이 2촌 이내 친족이거나 같은 세대·같은 계좌이면 지급 제외"),
            Fact::new("가까운 연결 경로", path),
            Fact::new("동명이인 후보", homonym),
        ],
        options: OPTIONS.to_vec(),
    }
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| model_dir("zoro").display().to_string());
    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let backbone = Arc::new(ZoroBackbone::open(Path::new(&dir), residency.clone())?);
    let head = JevHead::from_safetensors(&Path::new(&dir).join("jev_head.safetensors"))?;
    let model = PointerModel::new(Box::new(Shared(Arc::clone(&backbone))), head);

    // 1. 위치 점검
    let probe = task("있음 (1홉)", "아니오");
    let prompt = PointerPrompt::render(&probe);
    let enc = backbone.tokenizer().encode(&prompt.text)?;
    println!("\n[1] 프롬프트 {}자 · 토큰 {}개", prompt.text.chars().count(), enc.ids.len());
    let show = |label: &str, byte: usize| -> Result<()> {
        let i = token_covering(&enc.offsets, byte).expect("위치를 덮는 토큰");
        let from = i.saturating_sub(4);
        let context = backbone.tokenizer().decode(&enc.ids[from..=i])?;
        let own = backbone.tokenizer().decode(&enc.ids[i..=i])?;
        println!("  {label:<8} 토큰 #{i} {own:?} (앞 문맥 {context:?})");
        Ok(())
    };
    show("질의", prompt.query_byte)?;
    for (i, v) in OPTIONS.iter().enumerate() {
        show(&format!("선택지{i}"), prompt.option_bytes[i])?;
        println!("           코드 {}", v.code());
    }

    // 2. 결정성과 시간
    println!("\n[2] 같은 질문을 3번 (첫 호출은 가중치 적재 포함)");
    let mut first: Option<Vec<f32>> = None;
    for n in 0..3 {
        let t = Instant::now();
        let p = model.probabilities(&probe)?;
        println!("  호출 {n}: {:.2} s · {:?}", t.elapsed().as_secs_f64(), p.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
        match &first {
            None => first = Some(p),
            Some(f) => {
                let d = f.iter().zip(&p).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                println!("          첫 호출과의 최대 차이 {d:.2e}");
            }
        }
    }

    // 3. 사실에 따른 움직임
    println!("\n[3] 사실에 따른 확률 (선택지 순서: {})", OPTIONS.iter().map(|v| v.code()).collect::<Vec<_>>().join(" / "));
    let cases = [
        ("1홉 연결, 동명이인 아님", "있음 (1홉)", "아니오", "위반 쪽이어야 함"),
        ("2홉 연결, 동명이인 아님", "있음 (2홉)", "아니오", "위반 쪽이어야 함"),
        ("연결 없음, 동명이인 아님", "없음", "아니오", "정상 쪽이어야 함"),
        ("연결 없음, 동명이인", "없음", "예 (같은 이름이 연결망에 있으나 다른 사람으로 판별)", "동명이인 해소 쪽이어야 함"),
    ];
    for (label, path, homonym, expect) in cases {
        let t = Instant::now();
        let p = model.probabilities(&task(path, homonym))?;
        let best = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| OPTIONS[i].code()).unwrap_or("-");
        println!(
            "  {label:<22} {} · 최대 {best} · {:.2} s  ({expect})",
            p.iter().map(|x| format!("{:5.1}%", x * 100.0)).collect::<Vec<_>>().join(" "),
            t.elapsed().as_secs_f64()
        );
    }

    let res = residency.lock().unwrap();
    println!("\n상주: 모델 {:?} · 묶음 {:?}", res.resident_model(), res.resident_groups());
    accel::finish_process(&device, 0);
    Ok(())
}
