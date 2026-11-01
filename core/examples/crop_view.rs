//! 크롭 계획을 그림으로 봅니다. `siglip2_dump` 가 뽑은 특징을 쓰므로 GPU 가 필요 없습니다.
//!
//!   crop_view <이미지> <특징.bin> <앵커.bin | none> <출력.png>
//!
//! 제목 빨강 · 입력란 파랑 · 표 초록 · 서명 보라 · 버린 조각 회색

use std::fs;
use std::path::Path;

use anyhow::Result;
use image::{GrayImage, Rgb, RgbImage};
use zorogravity_core::ai::siglip2::ink::{deskew, estimate_skew, row_bands, Bbox, InkPage};
use zorogravity_core::ai::siglip2::layout::{blocks, runs};
use zorogravity_core::ai::siglip2::plan::{plan_page, plan_unfiltered, CropKind, PlanParams, Scores};
use zorogravity_core::ai::siglip2::probe::ProbeBank;
use zorogravity_core::ai::siglip2::PatchGrid;

fn frame(img: &mut RgbImage, (x0, y0, x1, y1): Bbox, color: Rgb<u8>) {
    let (w, h) = img.dimensions();
    for t in 0..3 {
        for x in x0..x1.min(w) {
            img.put_pixel(x, (y0 + t).min(h - 1), color);
            img.put_pixel(x, (y1.saturating_sub(1 + t)).min(h - 1), color);
        }
        for y in y0..y1.min(h) {
            img.put_pixel((x0 + t).min(w - 1), y, color);
            img.put_pixel((x1.saturating_sub(1 + t)).min(w - 1), y, color);
        }
    }
}

fn gray_of(img: &RgbImage) -> GrayImage {
    image::DynamicImage::ImageRgb8(img.clone()).to_luma8()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let img = image::open(&args[0])?.to_rgb8();
    let skew = estimate_skew(&gray_of(&img));
    let mut straight = deskew(&img, skew);
    let ink = InkPage::new(&gray_of(&straight));
    let params = PlanParams::default();
    let plan = if args[2] == "none" {
        plan_unfiltered(&ink, &params)
    } else {
        let raw = fs::read(&args[1])?;
        let u = |i: usize| u32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]) as usize;
        let (rows, cols, dim) = (u(0), u(4), u(8));
        let floats: Vec<f32> = raw[12..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let (patches, pooled) = floats.split_at(rows * cols * dim);
        let grid = PatchGrid { rows, cols, dim, patches: patches.to_vec(), pooled: pooled.to_vec(), width: img.width(), height: img.height() };
        plan_page(&ink, &Scores::new(&ProbeBank::from_file(Path::new(&args[2]))?, &grid)?, &params)
    };
    println!("기울기 {skew:+.1}° · 조각 {} · 버림 {} · 면적 {:.2}", plan.crops.len(), plan.dropped.len(), plan.kept_area());
    for b in blocks(&ink) {
        println!("  덩어리 {b:?}");
        for r in runs(&ink, b) {
            println!("    {:?} 문단 {} 행 {:?}", r.bbox, r.paragraph, row_bands(&ink, r.bbox));
        }
    }
    for c in &plan.crops {
        println!("  {} {:?} ×{:.2} 패치 {} 안내문 차 {:+.2}", c.kind.name(), c.bbox, c.upscale, c.patches(), c.guide_margin.unwrap_or(f32::NAN));
        let color = match c.kind {
            CropKind::Title => Rgb([220, 30, 30]),
            CropKind::Form => Rgb([30, 80, 220]),
            CropKind::Table => Rgb([20, 160, 60]),
            CropKind::Sign => Rgb([170, 40, 200]),
        };
        frame(&mut straight, c.bbox, color);
    }
    for d in &plan.dropped {
        println!("  버림 {d:?}");
        frame(&mut straight, *d, Rgb([150, 150, 150]));
    }
    straight.save(&args[3])?;
    Ok(())
}
