//! 페이지를 끝까지 읽어 봅니다: 기울기 보정 → SigLIP2 크롭 계획 → gocr.
//!
//!   read_page <이미지> [정답.json]
//!
//! 정답 파일이 있으면 값 문자열이 읽은 글에 그대로 들어 있는 비율을 함께 보여 줍니다.

use std::path::Path;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::document::DocumentReader;
use zorogravity_core::ai::residency::{Budget, Residency};

fn squeeze(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let reader = DocumentReader::open(Path::new("models"), residency.clone())?;
    let img = image::open(&args[0])?.to_rgb8();
    let r = reader.read(&img)?;
    println!(
        "기울기 {:+.1}° · 조각 {}개 · 버린 조각 {}개 · 면적 {:.2} · 패치 {}",
        r.skew,
        r.crops.len(),
        r.plan.dropped.len(),
        r.plan.kept_area(),
        r.plan.crops.iter().map(|c| c.patches()).sum::<u32>()
    );
    println!("시간: 배치 {:.0} ms · SigLIP2 {:.0} ms · OCR {:.0} ms", r.timings.layout_ms, r.timings.siglip2_ms, r.timings.ocr_ms);
    let mut all = String::new();
    for c in &r.crops {
        println!(
            "[{}] {:?} 평균 {:.3} 최소 {:.3} · 이미지 토큰 {} · 토큰 {} · 초안 {} 검증 {}{} · 비전 {:.0} 프리필 {:.0} 디코드 {:.0} ms\n{}",
            c.kind.name(),
            c.bbox,
            c.output.mean_confidence,
            c.output.min_confidence,
            c.output.image_tokens,
            c.output.tokens.len(),
            c.output.drafts,
            c.output.verifies,
            if c.output.truncated { " · 잘림" } else { "" },
            c.output.vision_ms,
            c.output.prefill_ms,
            c.output.decode_ms,
            c.output.text
        );
        all.push_str(&c.output.text);
        all.push('\n');
    }
    if let Some(gt) = args.get(1) {
        let gt: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(gt)?)?;
        let text = squeeze(&all);
        let (mut hit, mut total) = (0, 0);
        for g in gt["gt"].as_array().unwrap() {
            let (kind, want) = (g["kind"].as_str().unwrap(), squeeze(g["text"].as_str().unwrap_or("")));
            if want.is_empty() || !matches!(kind, "value" | "title") {
                continue;
            }
            total += 1;
            if text.contains(&want) {
                hit += 1;
            } else {
                println!("  못 읽음 [{kind}] {want}");
            }
        }
        println!("값 · 제목 그대로 읽힘 {hit}/{total}");
    }
    println!();
    let res = residency.lock().unwrap();
    for t in res.history() {
        println!("  {:?} {} {} {:.0} MB · {} ms", t.action, t.model, t.group, t.bytes as f64 / (1u64 << 20) as f64, t.millis);
    }
    accel::finish_process(&device, 0);
    Ok(())
}
