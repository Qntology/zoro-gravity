//! gocr 점검: 이미지 한 장을 실제 가중치로 읽습니다.
//!
//!   cargo run -p zorogravity-core --example gocr_probe -- 이미지.png
//!   cargo run -p zorogravity-core --example gocr_probe -- 이미지.png "Text Recognition:" 200

use std::time::Instant;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::gocr::image::prepare;
use zorogravity_core::ai::gocr::{open, GocrInput, Strategy, EARLY_EXIT, TEXT_PROMPT};
use zorogravity_core::ai::model_store::model_dir;
use zorogravity_core::ai::residency::{Budget, Residency};

fn mb(b: u64) -> f64 {
    b as f64 / (1u64 << 20) as f64
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("이미지 경로");
    let prompt = args.next().unwrap_or_else(|| TEXT_PROMPT.to_string());
    let max_tokens: usize = args.next().map(|s| s.parse()).transpose()?.unwrap_or(256);

    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let img = image::open(&path)?.to_rgb8();
    let t = Instant::now();
    let prepared = prepare(&img, 14, 2)?;
    println!(
        "이미지 {}×{} → 패치 {}×{} ({} 패치, 병합 토큰 {}) · 전처리 {:.0} ms",
        img.height(),
        img.width(),
        prepared.grid_h,
        prepared.grid_w,
        prepared.n_patches(),
        prepared.n_patches() / 4,
        t.elapsed().as_secs_f64() * 1e3
    );
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    // 모드: 환경 변수 ZOROGRAVITY_GOCR_MODE = ar | verify | early (기본 early)
    let strategy = match std::env::var("ZOROGRAVITY_GOCR_MODE").as_deref() {
        Ok("ar") => Strategy::Autoregressive,
        Ok("verify") => Strategy::Speculative { early_exit: None },
        _ => Strategy::Speculative { early_exit: Some(EARLY_EXIT) },
    };
    let engine = open(&model_dir("gocr"), residency.clone(), max_tokens, strategy)?;
    let t = Instant::now();
    let out = engine.recognize(&[GocrInput { image: prepared, prompt }])?;
    let secs = t.elapsed().as_secs_f64();
    let o = &out[0];
    println!(
        "\n--- 인식 결과 ({} 토큰, 초안 {} · 검증 {} · 조기 종료 {}, 평균 확신도 {:.3}, 최소 {:.3}, 잘림 {}, 전체 {secs:.2} s) ---\n{}\n---",
        o.tokens.len(), o.drafts, o.verifies, o.early_exit, o.mean_confidence, o.min_confidence, o.truncated, o.text
    );
    let r = residency.lock().unwrap();
    for tr in r.history() {
        println!("  {:?} {} {:.0} MB · {} ms", tr.action, tr.group, mb(tr.bytes), tr.millis);
    }
    accel::finish_process(&device, 0);
    Ok(())
}
