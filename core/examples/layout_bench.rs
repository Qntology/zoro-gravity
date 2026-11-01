//! 읽기 전 CPU 단계의 시간과 크롭 격자 분포를 잽니다. GPU 를 쓰지 않습니다.
//!
//!   layout_bench <서식 폴더> <특징 폴더> <앵커.bin> [장 수]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use image::{GrayImage, RgbImage};
use zorogravity_core::ai::document::cut;
use zorogravity_core::ai::gocr::image::{prepare, smart_resize};
use zorogravity_core::ai::siglip2::ink::{deskew, estimate_skew, InkPage};
use zorogravity_core::ai::siglip2::plan::{plan_page, CropKind, PlanParams, Scores};
use zorogravity_core::ai::siglip2::probe::ProbeBank;
use zorogravity_core::ai::siglip2::PatchGrid;

fn read_grid(path: &Path, width: u32, height: u32) -> Result<PatchGrid> {
    let raw = fs::read(path)?;
    let u = |i: usize| u32::from_le_bytes([raw[i], raw[i + 1], raw[i + 2], raw[i + 3]]) as usize;
    let (rows, cols, dim) = (u(0), u(4), u(8));
    let floats: Vec<f32> = raw[12..].as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
    let (patches, pooled) = floats.split_at(rows * cols * dim);
    Ok(PatchGrid { rows, cols, dim, patches: patches.to_vec(), pooled: pooled.to_vec(), width, height })
}

fn gray_of(img: &RgbImage) -> GrayImage {
    image::DynamicImage::ImageRgb8(img.clone()).to_luma8()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (forms, feats) = (Path::new(&args[0]), Path::new(&args[1]));
    let bank = ProbeBank::from_file(Path::new(&args[2]))?;
    let limit: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);
    let mut stems: Vec<String> = fs::read_dir(forms)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    stems.sort();
    stems.truncate(limit);

    let params = PlanParams::default();
    let mut spent = [0f64; 6];
    let names = ["그림 읽기(JPEG)", "기울기 찾기", "곧게 펴기", "잉크 지도", "계획", "크롭 자르기 · 패치"];
    let mut grids: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut per_page_groups = Vec::new();
    let (mut crops_total, mut patches_total) = (0usize, 0usize);
    for stem in &stems {
        let mut lap = {
            let mut t = Instant::now();
            move |slot: &mut f64| {
                *slot += t.elapsed().as_secs_f64() * 1e3;
                t = Instant::now();
            }
        };
        let img = image::open(forms.join(format!("{stem}.jpg")))?.to_rgb8();
        lap(&mut spent[0]);
        let deg = estimate_skew(&gray_of(&img));
        lap(&mut spent[1]);
        let straight = deskew(&img, deg);
        lap(&mut spent[2]);
        let ink = InkPage::new(&gray_of(&straight));
        lap(&mut spent[3]);
        let grid = read_grid(&feats.join(format!("{stem}.p256.bin")), img.width(), img.height())?;
        let mut t0 = Instant::now();
        let plan = plan_page(&ink, &Scores::new(&bank, &grid)?, &params);
        spent[4] += t0.elapsed().as_secs_f64() * 1e3;
        t0 = Instant::now();
        let mut page: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for c in plan.crops.iter().filter(|c| c.kind != CropKind::Sign) {
            let piece = cut(&straight, c.bbox, c.upscale);
            let (h, w) = smart_resize(piece.height() as usize, piece.width() as usize, 28)?;
            let prepared = prepare(&piece, 14, 2)?;
            *grids.entry((h, w)).or_default() += 1;
            *page.entry((h, w)).or_default() += 1;
            crops_total += 1;
            patches_total += prepared.n_patches();
        }
        spent[5] += t0.elapsed().as_secs_f64() * 1e3;
        per_page_groups.push((page.values().copied().sum::<usize>(), page.len()));
    }
    let n = stems.len() as f64;
    println!("{} 장 · 장당 평균", stems.len());
    for (name, v) in names.iter().zip(&spent) {
        println!("  {name:<20} {:>8.1} ms", v / n);
    }
    println!("  합 {:.1} ms", spent.iter().sum::<f64>() / n);
    println!("\n크롭 {crops_total}개 · 패치 {patches_total} · 격자 {}가지", grids.len());
    let same: usize = grids.values().filter(|v| **v > 1).sum();
    println!("다른 크롭과 격자가 같은 크롭 {same}개 ({:.0}%) (전체 페이지 기준)", 100.0 * same as f32 / crops_total.max(1) as f32);
    let (c, g): (usize, usize) = per_page_groups.iter().fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    println!("한 장 안에서: 크롭 {c}개 · 묶음 {g}개 (묶음당 {:.2}개)", c as f32 / g.max(1) as f32);
    let mut top: Vec<_> = grids.iter().collect();
    top.sort_by_key(|(_, v)| std::cmp::Reverse(**v));
    let list: Vec<String> = top.iter().take(12).map(|((h, w), v)| format!("{h}×{w}:{v}")).collect();
    println!("가장 많은 격자: {}", list.join(" · "));
    Ok(())
}
