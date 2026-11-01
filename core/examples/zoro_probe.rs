//! zoro 백본 점검: 실제 q8 가중치로 프롬프트를 읽고 다음 토큰을 확인합니다.
//!
//! 구조에 대한 가정(정규화 방식, 출력 게이트, RoPE 등)이 맞는지는 독립된 기준 구현이 없어서 이 점검으로만 알 수 있습니다.
//! 맞다면 "The capital of France is" 다음에 " Paris" 같은 말이 되는 토큰이 높은 확률로 나와야 합니다.
//!
//! 다음 토큰 확률은 임베딩 표를 출력 투영으로 다시 쓰는 방식(연결 가중치)으로 계산합니다. 이 점검에서만 표 전체(약 640 MB)를 CPU 에 읽습니다.
//! 판단 경로(`ZoroBackbone::encode`)는 표를 올리지 않습니다.
//!
//!   cargo run --release -p zorogravity-core --example zoro_probe -- models/zoro "The capital of France is" 8
//!   ZOROGRAVITY_DEVICE=cpu cargo run -p zorogravity-core --example zoro_probe -- models/zoro
//!   ZOROGRAVITY_ZORO_STREAM_BLOCKS=8 ...   (블록 8 개씩 올렸다 내리는 스트리밍)

use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::quantized::QMatMul;
use candle_core::{Device, Module};
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::zoro::ZoroBackbone;
use zorogravity_core::ai::accel;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "models/zoro".into());
    let prompt = args.next().unwrap_or_else(|| "The capital of France is".into());
    let steps: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(6);

    let device = accel::select_device()?;
    let budget = Budget::auto(&device);
    println!(
        "장치 {} · 가중치 예산 {} MB · 여유 {:?} MB",
        accel::backend_name(&device),
        budget.max_bytes().map_or("제한 없음".to_string(), |b| format!("{:.0}", mb(b))),
        device.mem_info().ok().flatten().map(|(f, _)| mb(f as u64) as u64),
    );
    let residency = Residency::shared(device.clone(), budget);
    let backbone = ZoroBackbone::open(Path::new(&dir), residency.clone())?;
    let plan: Vec<String> = backbone.model().plan().iter().map(|p| p.name.clone()).collect();
    println!("단계 계획: {}", plan.join(" → "));

    // 출력 투영 = 임베딩 표 (CPU)
    let t = Instant::now();
    let emb = backbone.model().spec().file.read_qtensor("token_embd.weight", &Device::Cpu)?;
    let lm_head = QMatMul::from_qtensor(emb)?;
    println!("임베딩 표 읽기 (점검 전용): {:.1} s", t.elapsed().as_secs_f64());

    let tk = backbone.tokenizer();
    let enc = tk.encode(&prompt)?;
    let mut ids = enc.ids.clone();
    println!("프롬프트 {:?} → 토큰 {}개 {:?}", prompt, ids.len(), ids);

    let eos = [248046u32, 248044];
    let mut generated = String::new();
    for step in 0..steps {
        let t = Instant::now();
        let hidden = backbone.model().hidden_states(&ids)?;
        let last = hidden.narrow(0, ids.len() - 1, 1)?.to_device(&Device::Cpu)?;
        let forward_s = t.elapsed().as_secs_f64();
        let logits = lm_head.forward(&last)?.squeeze(0)?.to_vec1::<f32>()?;
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|a, b| logits[*b].total_cmp(&logits[*a]));
        let max = logits[order[0]];
        let z: f32 = logits.iter().map(|l| (l - max).exp()).sum();
        let top: Vec<String> = order[..5]
            .iter()
            .map(|&i| format!("{:?} {:.1}% ({:.1})", tk.decode(&[i as u32]).unwrap_or_default(), 100.0 * (logits[i] - max).exp() / z, logits[i]))
            .collect();
        println!("[{step}] 순전파 {forward_s:.2} s · 상위: {}", top.join(" | "));
        let next = order[0] as u32;
        if eos.contains(&next) {
            println!("종료 토큰");
            break;
        }
        generated.push_str(&tk.decode(&[next])?);
        ids.push(next);
    }
    println!("\n생성: {prompt}{generated}");

    let res = residency.lock().unwrap();
    let snap = res.snapshot();
    println!(
        "\n상주: 모델 {:?} · 묶음 {:?} · {:.0} MB · 장치 여유 {:?} MB",
        snap.model,
        snap.groups.iter().map(|(g, b)| format!("{g}={:.0}MB", mb(*b))).collect::<Vec<_>>(),
        mb(snap.resident_bytes),
        snap.free_bytes.map(|b| mb(b) as u64),
    );
    for tr in res.history() {
        println!(
            "  {:?} {} {:.0} MB · {} ms · 여유 {:?} → {:?} MB",
            tr.action,
            tr.group,
            mb(tr.bytes),
            tr.millis,
            tr.free_before.map(|b| mb(b) as u64),
            tr.free_after.map(|b| mb(b) as u64),
        );
    }
    accel::finish_process(&device, 0);
    Ok(())
}
