//! 크롭 계획 평가: 합성 서식의 정답 상자로 놓친 내용과 줄어든 면적을 잽니다. `siglip2_dump` 가 뽑은 특징을 쓰므로 GPU 가 필요 없습니다.
//!
//!   crop_eval <서식 폴더> <특징 폴더> <앵커.bin | none> [앵커 폴더: 서식 종류별 교차 검증]
//!
//! 기준값은 환경 변수로 바꿉니다: ZOROGRAVITY_DROP_MARGIN, ZOROGRAVITY_SIGN_GATE, ZOROGRAVITY_TEXT_PX, ZOROGRAVITY_TILE_PATCHES, ZOROGRAVITY_ARENA=1, ZOROGRAVITY_DROP_SHARE
//! 정답 상자 안 잉크의 90% 이상이 크롭에 들어가면 찾은 것으로 칩니다 (표는 테두리가 잘려 나가므로 80%). 상자에 잉크가 거의 없으면 세지 않습니다.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{bail, Result};
use image::{GrayImage, RgbImage};
use serde_json::Value;
use zorogravity_core::ai::siglip2::ink::{deskew, estimate_skew, InkPage};
use zorogravity_core::ai::siglip2::plan::{plan_page, plan_unfiltered, CropKind, PagePlan, PlanParams, Scores};
use zorogravity_core::ai::siglip2::probe::ProbeBank;
use zorogravity_core::ai::siglip2::PatchGrid;

const CAUGHT: f32 = 0.9;
const CAUGHT_TABLE: f32 = 0.8;
const COARSE: u32 = 4;

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

/// 기운 페이지를 곧게 폈을 때의 정답 상자
fn straighten(b: [f32; 4], deg: f32, w: f32, h: f32) -> [f32; 4] {
    let (sin, cos) = deg.to_radians().sin_cos();
    let (cx, cy) = (w / 2.0, h / 2.0);
    let (mut lx, mut ly, mut hx, mut hy) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for (x, y) in [(b[0], b[1]), (b[2], b[1]), (b[0], b[3]), (b[2], b[3])] {
        let (dx, dy) = (x - cx, y - cy);
        let (ox, oy) = (cx + dx * cos + dy * sin, cy - dx * sin + dy * cos);
        lx = lx.min(ox);
        ly = ly.min(oy);
        hx = hx.max(ox);
        hy = hy.max(oy);
    }
    [lx, ly, hx, hy]
}

/// 크롭들이 덮는 칸 지도
fn coverage(plan: &PagePlan) -> (Vec<bool>, usize) {
    let (w, h) = ((plan.width / COARSE + 1) as usize, (plan.height / COARSE + 1) as usize);
    let mut map = vec![false; w * h];
    for c in &plan.crops {
        for y in c.bbox.1 / COARSE..c.bbox.3 / COARSE {
            for x in c.bbox.0 / COARSE..c.bbox.2 / COARSE {
                map[y as usize * w + x as usize] = true;
            }
        }
    }
    (map, w)
}

/// 상자 안 잉크 중 크롭에 들어간 비율. 상자에 잉크가 없으면 None.
fn covered(map: &[bool], w: usize, ink: &InkPage, b: [f32; 4]) -> Option<f32> {
    let (x0, y0, x1, y1) = ((b[0].max(0.0)) as u32 / COARSE, (b[1].max(0.0)) as u32 / COARSE, b[2] as u32 / COARSE, b[3] as u32 / COARSE);
    let h = (map.len() / w) as u32;
    let (mut hit, mut all) = (0u64, 0u64);
    for y in y0..=y1.min(h - 1) {
        for x in x0..=x1.min(w as u32 - 1) {
            let cell = (x * COARSE, y * COARSE, ((x + 1) * COARSE).min(ink.width), ((y + 1) * COARSE).min(ink.height));
            if cell.2 <= cell.0 || cell.3 <= cell.1 {
                continue;
            }
            let n = ink.ink_in(cell);
            all += n;
            if map[y as usize * w + x as usize] {
                hit += n;
            }
        }
    }
    (all >= 16).then(|| hit as f32 / all as f32)
}

#[derive(Default)]
struct Row {
    pages: usize,
    found: BTreeMap<String, (usize, usize)>,
    guide_kept: (f32, usize),
    area: f32,
    crops: usize,
    patches: u64,
    dropped: usize,
    title_hit: usize,
    title_pages: usize,
    sign_hit: usize,
    sign_total: usize,
    sign_crops: usize,
    sign_false: usize,
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<()> {
    let d = PlanParams::default();
    let params = PlanParams {
        drop_margin: env_or("ZOROGRAVITY_DROP_MARGIN", d.drop_margin),
        sign_gate: env_or("ZOROGRAVITY_SIGN_GATE", d.sign_gate),
        text_px: env_or("ZOROGRAVITY_TEXT_PX", d.text_px),
        max_tile_patches: env_or("ZOROGRAVITY_TILE_PATCHES", d.max_tile_patches),
        use_arena: env_or("ZOROGRAVITY_ARENA", d.use_arena as u8) == 1,
        drop_share: env_or("ZOROGRAVITY_DROP_SHARE", d.drop_share),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 3 {
        bail!("사용법: crop_eval <서식 폴더> <특징 폴더> <앵커.bin|none> [앵커 폴더]");
    }
    let (forms, feats) = (Path::new(&args[0]), Path::new(&args[1]));
    let common = if args[2] == "none" { None } else { Some(ProbeBank::from_file(Path::new(&args[2]))?) };
    let per_doc = args.get(3).map(Path::new);

    let mut stems: Vec<String> = fs::read_dir(forms)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().to_string())
        .collect();
    stems.sort();

    let mut by_doc: BTreeMap<String, Row> = BTreeMap::new();
    let mut misses: Vec<String> = Vec::new();
    for stem in &stems {
        let gt: Value = serde_json::from_str(&fs::read_to_string(forms.join(format!("{stem}.json")))?)?;
        let doc = gt["doc"].as_str().unwrap().to_string();
        let img = image::open(forms.join(format!("{stem}.jpg")))?.to_rgb8();
        let deg = estimate_skew(&gray_of(&img));
        let straight = deskew(&img, deg);
        let ink = InkPage::new(&gray_of(&straight));
        let grid = read_grid(&feats.join(format!("{stem}.p256.bin")), img.width(), img.height())?;
        let loo = per_doc.map(|d| ProbeBank::from_file(&d.join(format!("crop_probe.{doc}.bin")))).transpose()?;
        let plan = match loo.as_ref().or(common.as_ref()) {
            Some(bank) => plan_page(&ink, &Scores::new(bank, &grid)?, &params),
            None => plan_unfiltered(&ink, &params),
        };
        let (map, w) = coverage(&plan);
        let row = by_doc.entry(doc.clone()).or_default();
        row.pages += 1;
        row.area += plan.kept_area();
        row.crops += plan.crops.iter().filter(|c| c.kind != CropKind::Sign).count();
        row.patches += plan.crops.iter().filter(|c| c.kind != CropKind::Sign).map(|c| c.patches() as u64).sum::<u64>();
        row.dropped += plan.dropped.len();
        // 서명 크롭: 정답 서명과 겹치면 찾은 것, 아무 정답과도 겹치지 않으면 헛것입니다.
        let boxes_of = |want: &str| -> Vec<[f32; 4]> {
            gt["gt"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|g| g["kind"] == want)
                .map(|g| {
                    let b: Vec<f32> = g["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
                    straighten([b[0], b[1], b[2], b[3]], deg, img.width() as f32, img.height() as f32)
                })
                .collect()
        };
        let signs = boxes_of("sign");
        let meets = |c: &zorogravity_core::ai::siglip2::plan::CropPlan, b: &[f32; 4]| c.bbox.0 as f32 <= b[2] && b[0] <= c.bbox.2 as f32 && c.bbox.1 as f32 <= b[3] && b[1] <= c.bbox.3 as f32;
        let sign_crops: Vec<_> = plan.crops.iter().filter(|c| c.kind == CropKind::Sign).collect();
        row.sign_total += signs.len();
        row.sign_hit += signs.iter().filter(|b| sign_crops.iter().any(|c| meets(c, b))).count();
        row.sign_crops += sign_crops.len();
        row.sign_false += sign_crops.iter().filter(|c| !signs.iter().any(|b| meets(c, b))).count();
        for g in gt["gt"].as_array().unwrap() {
            let kind = g["kind"].as_str().unwrap();
            let b: Vec<f32> = g["box"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let b = straighten([b[0], b[1], b[2], b[3]], deg, img.width() as f32, img.height() as f32);
            let Some(c) = covered(&map, w, &ink, b) else { continue };
            if kind == "guide" {
                row.guide_kept.0 += c;
                row.guide_kept.1 += 1;
            } else if kind != "label" {
                let e = row.found.entry(kind.to_string()).or_default();
                e.1 += 1;
                if c >= if kind == "table" { CAUGHT_TABLE } else { CAUGHT } {
                    e.0 += 1;
                } else if misses.len() < 40 {
                    misses.push(format!("{stem} {kind} {:.0}% {:?} {}", c * 100.0, b.map(|v| v as i32), g["text"].as_str().unwrap_or("")));
                }
                if kind == "title" {
                    let hit = plan.crops.iter().any(|p| {
                        p.kind == CropKind::Title && p.bbox.0 as f32 <= b[2] && b[0] <= p.bbox.2 as f32 && p.bbox.1 as f32 <= b[3] && b[1] <= p.bbox.3 as f32
                    });
                    row.title_hit += hit as usize;
                    row.title_pages += 1;
                }
            }
        }
    }

    println!("{:<18}{:>7}{:>7}{:>7}{:>7}{:>9}{:>8}{:>6}{:>8}{:>7}", "서식", "값", "서명", "표", "제목", "제목표시", "안내남음", "면적", "조각", "패치");
    let mut total = Row::default();
    let pct = |p: Option<&(usize, usize)>| p.filter(|v| v.1 > 0).map_or("-".to_string(), |v| format!("{:.0}%", 100.0 * v.0 as f32 / v.1 as f32));
    for (doc, r) in &by_doc {
        println!(
            "{doc:<18}{:>7}{:>7}{:>7}{:>7}{:>7}/{:<2}{:>7.0}%{:>6.2}{:>8.1}{:>7}",
            pct(r.found.get("value")),
            pct(r.found.get("sign")),
            pct(r.found.get("table")),
            pct(r.found.get("title")),
            r.title_hit,
            r.title_pages,
            100.0 * r.guide_kept.0 / r.guide_kept.1.max(1) as f32,
            r.area / r.pages as f32,
            r.crops as f32 / r.pages as f32,
            r.patches / r.pages as u64,
        );
        total.pages += r.pages;
        for (k, v) in &r.found {
            let e = total.found.entry(k.clone()).or_default();
            e.0 += v.0;
            e.1 += v.1;
        }
        total.guide_kept.0 += r.guide_kept.0;
        total.guide_kept.1 += r.guide_kept.1;
        total.area += r.area;
        total.crops += r.crops;
        total.patches += r.patches;
        total.title_hit += r.title_hit;
        total.title_pages += r.title_pages;
        total.sign_hit += r.sign_hit;
        total.sign_total += r.sign_total;
        total.sign_crops += r.sign_crops;
        total.sign_false += r.sign_false;
    }
    println!(
        "{:<18}{:>7}{:>7}{:>7}{:>7}{:>7}/{:<2}{:>7.0}%{:>6.2}{:>8.1}{:>7}",
        "전체",
        pct(total.found.get("value")),
        pct(total.found.get("sign")),
        pct(total.found.get("table")),
        pct(total.found.get("title")),
        total.title_hit,
        total.title_pages,
        100.0 * total.guide_kept.0 / total.guide_kept.1.max(1) as f32,
        total.area / total.pages as f32,
        total.crops as f32 / total.pages as f32,
        total.patches / total.pages as u64,
    );
    println!("서명 크롭: 찾음 {}/{} · 크롭 {}개 중 헛것 {}개", total.sign_hit, total.sign_total, total.sign_crops, total.sign_false);
    if !misses.is_empty() {
        println!("\n놓친 것 (앞 {}건):", misses.len());
        for m in &misses {
            println!("  {m}");
        }
    }
    Ok(())
}
