//! 선택된 GGUF 의 가중치 묶음별 크기와 주요 텐서의 형식·통계를 봅니다. 아무것도 장치에 올리지 않고 목차와 작은 텐서만 읽습니다.
//!
//!   cargo run -p zorogravity-core --example weight_groups -- models/zoro
//!   cargo run -p zorogravity-core --example weight_groups -- models/gocr

use std::path::Path;

use anyhow::Result;
use candle_core::{DType, Device};
use zorogravity_core::ai::gguf_model::load_active;
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::residency::{estimate_group_bytes, Group};
use zorogravity_core::ai::weights::WeightFile;

fn mb(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 20) as f64
}

fn stats(file: &WeightFile, name: &str) -> Result<String> {
    if !file.has(name) {
        return Ok(format!("{name}: 없음"));
    }
    let meta = file.info(name)?;
    let v = file.read_dense(name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
    let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
    Ok(format!("{name:<34} {:?} {:?} 평균 {mean:+.4} 표준편차 {std:.4} 최소 {lo:+.4} 최대 {hi:+.4}", meta.dtype, meta.dims))
}

fn main() -> Result<()> {
    let dir = std::env::args().nth(1).unwrap_or_else(|| model_dir("zoro").display().to_string());
    let active = load_active(Path::new(&dir))?;
    let file = WeightFile::open(&active.path)?;
    println!("{} ({}) · 텐서 {}개 · 파일 안 합계 {:.0} MB", active.path.display(), active.quant, file.tensor_count(), mb(file.total_bytes()));

    let cpu = Device::Cpu;
    println!("\n[묶음별 크기, 장치에 올렸을 때 (CPU 기준 dtype)]");
    let mmproj = active.open_mmproj()?;
    for group in [Group::Embedding, Group::Output, Group::Vision] {
        let holder = if group == Group::Vision { mmproj.as_deref().unwrap_or(&file) } else { &file };
        match estimate_group_bytes(holder, group, &cpu) {
            Ok(b) => println!("  {group:<14} {:>9.1} MB", mb(b)),
            Err(_) => println!("  {group:<14} 해당 텐서 없음"),
        }
    }
    let blocks = active.block_count as usize;
    let mut text_total = 0u64;
    for i in 0..blocks {
        let b = estimate_group_bytes(&file, Group::blocks(i..i + 1), &cpu)?;
        text_total += b;
        if i < 5 || i + 1 == blocks {
            println!("  {:<14} {:>9.1} MB", Group::blocks(i..i + 1).to_string(), mb(b));
        } else if i == 5 {
            println!("  ...");
        }
    }
    println!("  텍스트 블록 합계 {:.1} MB", mb(text_total));

    println!("\n[블록 0 · 블록 3 텐서]");
    for blk in [0usize, 3] {
        for n in file.names_with_prefix(&format!("blk.{blk}.")) {
            let m = file.info(&n)?;
            println!("  {n:<38} {:?} {:?} {:.2} MB", m.dtype, m.dims, mb(m.bytes));
        }
    }
    println!("\n[비전 텐서 앞 6개]");
    for n in file.names_with_prefix("model.visual.").into_iter().take(6) {
        let m = file.info(&n)?;
        println!("  {n:<62} {:?} {:?}", m.dtype, m.dims);
    }

    println!("\n[작은 텐서 통계]");
    for n in [
        "blk.0.attn_norm.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ssm_norm.weight",
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_conv1d.weight",
        "blk.3.attn_norm.weight",
        "blk.3.attn_q_norm.weight",
        "blk.3.attn_k_norm.weight",
        "blk.16.attn_norm.weight",
        "blk.31.post_attention_norm.weight",
        "output_norm.weight",
    ] {
        println!("  {}", stats(&file, n)?);
    }
    Ok(())
}
