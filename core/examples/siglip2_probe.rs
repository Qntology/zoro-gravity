//! siglip2 점검: 이미지와 문구를 실제 가중치로 비교합니다.
//!
//!   cargo run -p zorogravity-core --example siglip2_probe -- 이미지.png [최대 패치] [문구 ...]
//!
//! 이미지 전체 일치도(로짓 → 확률)와, 문구마다 패치 코사인의 격자를 찍습니다.

use std::time::Instant;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::siglip2::image::prepare;
use zorogravity_core::ai::siglip2::Siglip2;

fn mb(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("이미지 경로");
    let max_patches: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(256);
    let phrases: Vec<String> = args.collect();

    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let model = Siglip2::open(&model_dir("siglip2"), residency.clone())?;
    let img = image::open(&path)?.to_rgb8();

    let t = Instant::now();
    let input = prepare(&img, model.config().patch, max_patches)?;
    println!("이미지 {}×{} → 격자 {}×{} · 전처리 {:.0} ms", img.width(), img.height(), input.rows, input.cols, t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let grid = model.encode_image(&input)?;
    println!("비전 {:.2} s", t.elapsed().as_secs_f64());
    let t = Instant::now();
    let grid2 = model.encode_image(&input)?;
    println!("비전(두 번째) {:.2} s · 재현 차이 {:.2e}", t.elapsed().as_secs_f64(), grid.patches.iter().zip(&grid2.patches).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max));

    let t = Instant::now();
    let vectors = model.encode_phrases(&phrases)?;
    println!("텍스트 {}개 {:.2} s", phrases.len(), t.elapsed().as_secs_f64());
    let (scale, bias) = model.logit_params()?;
    println!("logit_scale {scale:.3} · logit_bias {bias:.3}");

    for (p, v) in phrases.iter().zip(&vectors) {
        let cos: f32 = grid.pooled.iter().zip(v).map(|(a, b)| a * b).sum();
        let logit = cos * scale.exp() + bias;
        println!("\n{p}: 전체 코사인 {cos:+.4} · 확률 {:.1}%", 100.0 / (1.0 + (-logit).exp()));
        let scores: Vec<f32> = (0..grid.len()).map(|i| grid.patch(i).iter().zip(v).map(|(a, b)| a * b).sum()).collect();
        let (lo, hi) = scores.iter().fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
        println!("  패치 코사인 {lo:+.3} ~ {hi:+.3}");
        const SHADE: &[u8] = b" .:-=+*#%@";
        for r in 0..grid.rows {
            let row: String = (0..grid.cols)
                .map(|c| {
                    let x = (scores[r * grid.cols + c] - lo) / (hi - lo).max(1e-6);
                    SHADE[((x * 9.0) as usize).min(9)] as char
                })
                .collect();
            println!("  |{row}|");
        }
    }
    println!();
    let r = residency.lock().unwrap();
    for tr in r.history() {
        println!("  {:?} {} {:.0} MB · {} ms", tr.action, tr.group, mb(tr.bytes), tr.millis);
    }
    accel::finish_process(&device, 0);
    Ok(())
}
