//! 가중치 상주 점검: 실제 GGUF 와 실제 장치로 "모델은 한 번에 하나, 단계마다 필요한 묶음만" 을 확인합니다.
//!
//! 순서 (logis-center 의 단계 전환과 같은 모양):
//!
//! 1. gocr 비전 단계 — 비전 타워만 올림 (언어 가중치 없음)
//! 2. gocr 언어 단계 — 비전을 내리고 언어 가중치만 올림
//! 3. zoro 텍스트 단계 — gocr 를 모두 내리고 zoro 텍스트 블록만 올림 (비전 · 임베딩 표 · 출력 투영은 올리지 않음)
//! 4. 다시 gocr 비전 — zoro 를 모두 내림
//! 5. 모두 내림 — 장치 여유 메모리가 처음 값 근처로 돌아와야 함
//!
//! 각 단계에서 장치의 여유 메모리를 재서, 내린 묶음의 메모리가 실제로 돌아왔는지 봅니다.
//! 장치 전체의 여유를 재므로 다른 GPU 프로그램이 돌고 있으면 값이 흔들립니다.
//!
//!   cargo run -p zorogravity-core --example residency_check
//!   ZOROGRAVITY_DEVICE=rocm cargo run -p zorogravity-core --no-default-features --features rocm --example residency_check

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};
use zorogravity_core::ai::accel;
use zorogravity_core::ai::gguf_model::load_active;
use zorogravity_core::ai::residency::{Budget, Group, ModelId, ModelSpec, Phase, Residency};
use zorogravity_core::ai::weights::WeightFile;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn spec(dir: &str, id: ModelId) -> Result<ModelSpec> {
    let active = load_active(Path::new(dir))?;
    let spec = ModelSpec::new(id, Arc::new(WeightFile::open(&active.path)?));
    Ok(match active.open_mmproj()? {
        Some(mmproj) => spec.with_mmproj(mmproj),
        None => spec,
    })
}

fn main() -> Result<()> {
    let device = accel::select_device()?;
    let zoro = spec("models/zoro", ModelId::Zoro)?;
    let gocr = spec("models/gocr", ModelId::Gocr)?;
    let zoro_blocks = zoro.file.meta_usize("zoro.block_count")? as usize;
    let gocr_blocks = gocr.file.meta_usize("gocr.block_count")? as usize;

    let mut res = Residency::new(device.clone(), Budget::auto(&device));
    println!(
        "장치 {} · 예산 {} MB",
        accel::backend_name(&device),
        res.budget().max_bytes().map_or("제한 없음".into(), |b| format!("{:.0}", mb(b)))
    );
    let free0 = res.free_bytes();
    println!("처음 장치 여유: {:?} MB\n", free0.map(mb));

    let gocr_vision = Phase::new("gocr:비전", [Group::Vision]);
    let gocr_language = Phase::new("gocr:언어", [Group::Embedding, Group::blocks(0..gocr_blocks), Group::Output]);
    let zoro_text = Phase::new("zoro:텍스트", [Group::blocks(0..zoro_blocks), Group::Output]);

    let mut single_model_always = true;
    let mut step = |res: &mut Residency, name: &str, spec: &ModelSpec, phase: &Phase| -> Result<()> {
        let handles = res.enter(spec, phase)?;
        let used: u64 = handles.iter().map(|h| h.device_bytes()).sum();
        drop(handles);
        let groups: Vec<String> = res.resident_groups().iter().map(|g| g.to_string()).collect();
        // 올라온 모델이 하나뿐인지: 상주 관리자는 모델 하나만 들고 있습니다.
        single_model_always &= res.resident_model() == Some(spec.id);
        println!(
            "{name:<22} 모델 {:?} · 묶음 [{}] · 상주 {:.0} MB (이 단계 {:.0} MB) · 장치 여유 {:?} MB · 일반 텐서 버퍼 {:?} MB",
            res.resident_model().map(|m| m.name()),
            groups.join(", "),
            mb(res.resident_bytes()),
            mb(used),
            res.free_bytes().map(mb).map(|v| v as u64),
            res.tensor_buffer_bytes().map(mb).map(|v| v as u64),
        );
        Ok(())
    };

    step(&mut res, "1 gocr 비전", &gocr, &gocr_vision)?;
    let vision_only = res.resident_groups();
    if vision_only != vec![Group::Vision] {
        bail!("비전 단계에서 비전만 올라와야 합니다: {vision_only:?}");
    }
    step(&mut res, "2 gocr 언어", &gocr, &gocr_language)?;
    if res.resident_groups().contains(&Group::Vision) {
        bail!("언어 단계에서는 비전이 내려가 있어야 합니다");
    }
    step(&mut res, "3 zoro 텍스트", &zoro, &zoro_text)?;
    if res.resident_model() != Some(ModelId::Zoro) || res.resident_groups().iter().any(|g| matches!(g, Group::Vision | Group::Embedding)) {
        bail!("zoro 단계에는 zoro 의 텍스트 블록과 출력만 있어야 합니다: {:?}", res.resident_groups());
    }
    step(&mut res, "4 gocr 비전 (다시)", &gocr, &gocr_vision)?;
    res.release_all()?;
    println!(
        "{:<22} 상주 {:.0} MB · 장치 여유 {:?} MB",
        "5 모두 내림",
        mb(res.resident_bytes()),
        res.free_bytes().map(mb).map(|v| v as u64)
    );

    println!("\n[올리고 내린 기록] 여유 메모리는 그 동작 전 → 후");
    for t in res.history() {
        println!(
            "  {:<7} {:<6} {:<12} {:>7.0} MB · {:>5} ms · 장치 여유 {:?} → {:?} MB · 일반 텐서 버퍼 {:?} → {:?} MB",
            format!("{:?}", t.action),
            t.model.name(),
            t.group.to_string(),
            mb(t.bytes),
            t.millis,
            t.free_before.map(|b| mb(b) as u64),
            t.free_after.map(|b| mb(b) as u64),
            t.buffers_before.map(|b| mb(b) as u64),
            t.buffers_after.map(|b| mb(b) as u64),
        );
    }

    if !single_model_always {
        bail!("한 단계에서 두 모델이 함께 올라와 있었습니다");
    }
    if let (Some(f0), Some(f1)) = (free0, res.free_bytes()) {
        let lost = f0 as i128 - f1 as i128;
        println!("\n처음과 비교한 장치 여유 차이: {:+.0} MB (0 에 가까울수록 모두 돌려받음)", -(lost as f64) / (1u64 << 20) as f64);
        if lost > (256 << 20) {
            bail!("모두 내린 뒤에도 장치 메모리가 {:.0} MB 덜 돌아왔습니다", lost as f64 / (1u64 << 20) as f64);
        }
    }
    println!("확인 완료");
    accel::finish_process(&device, 0);
    Ok(())
}
