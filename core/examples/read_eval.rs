//! 페이지 묶음을 끝까지 읽고 값이 얼마나 그대로 읽혔는지 잽니다.
//!
//!   read_eval <서식 폴더> [이름에 들어갈 글자] [최대 장수]
//!
//! 기준값은 환경 변수로 바꿉니다: ZOROGRAVITY_TEXT_PX, ZOROGRAVITY_TILE_PATCHES, ZOROGRAVITY_DROP_MARGIN, ZOROGRAVITY_FORM_TABLE=0|1, ZOROGRAVITY_ARENA=1, ZOROGRAVITY_DROP_SHARE
//! `ZOROGRAVITY_BUNDLE` 은 한 번에 읽는 장 수입니다 (기본 1). 민원 한 건의 서류 묶음에 해당합니다. `ZOROGRAVITY_VISION_BATCH=0` 은 비전 묶음을 끕니다.
//! 비교는 공백을 뺀 문자열로 합니다. 표 프롬프트의 태그는 지우고 비교합니다.
//! 그대로 읽은 값, 틀린 글자가 값 길이의 20% 이하인 값(오독), 그보다 많이 다른 값(건너뜀)으로 나눕니다.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::document::{DocumentReader, ReadOptions};
use zorogravity_core::ai::model_store::models_dir;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::siglip2::plan::PlanParams;

fn squeeze(s: &str) -> String {
    // 표 태그를 지웁니다.
    let mut out = String::new();
    let mut tag = false;
    for c in s.chars() {
        match c {
            '<' => tag = true,
            '>' => tag = false,
            c if !tag && !c.is_whitespace() => out.push(c),
            _ => {}
        }
    }
    out
}

/// `want` 와 `text` 의 어떤 부분 문자열 사이의 가장 작은 편집 거리
fn best_distance(want: &[char], text: &[char]) -> usize {
    // 열은 패턴 위치, 첫 행은 0 이라서 글의 어디에서든 시작할 수 있습니다.
    let mut col: Vec<usize> = (0..=want.len()).collect();
    let mut best = want.len();
    for &c in text {
        let mut diag = col[0];
        col[0] = 0;
        for (i, &w) in want.iter().enumerate() {
            let next = (diag + (w != c) as usize).min(col[i] + 1).min(col[i + 1] + 1);
            diag = col[i + 1];
            col[i + 1] = next;
        }
        best = best.min(col[want.len()]);
    }
    best
}

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(&args[0]);
    let filter = args.get(1).cloned().unwrap_or_default();
    let limit: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(usize::MAX);
    let d = PlanParams::default();
    let options = ReadOptions {
        plan: PlanParams {
            drop_margin: env_or("ZOROGRAVITY_DROP_MARGIN", d.drop_margin),
            text_px: env_or("ZOROGRAVITY_TEXT_PX", d.text_px),
            max_tile_patches: env_or("ZOROGRAVITY_TILE_PATCHES", d.max_tile_patches),
            use_arena: env_or("ZOROGRAVITY_ARENA", d.use_arena as u8) == 1,
            drop_share: env_or("ZOROGRAVITY_DROP_SHARE", d.drop_share),
            ..d
        },
        forms_as_tables: env_or("ZOROGRAVITY_FORM_TABLE", 1) == 1,
    };

    let device = accel::select_device()?;
    println!("장치 {} · {options:?}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let reader = DocumentReader::open(&models_dir(), residency)?.with_options(options);

    let mut stems: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().unwrap().to_string_lossy().to_string())
        .filter(|s| s.contains(&filter))
        .collect();
    stems.sort();
    stems.truncate(limit);

    let bundle: usize = env_or("ZOROGRAVITY_BUNDLE", 1);
    let t = Instant::now();
    let mut phases = [0f64; 3]; // 비전 · 프리필 · 디코딩 (ms)
    let mut read_tokens = 0usize;
    let mut by_doc: BTreeMap<String, (usize, usize, usize, f64, f64, u64)> = BTreeMap::new();
    let (mut hit, mut total, mut title_hit, mut title_total) = (0, 0, 0, 0);
    let (mut misread, mut skipped) = (0, 0);
    let mut misses = Vec::new();
    let mut readings = Vec::new();
    for names in stems.chunks(bundle) {
        let images: Vec<image::RgbImage> = names.iter().map(|s| Ok(image::open(dir.join(format!("{s}.jpg")))?.to_rgb8())).collect::<Result<_>>()?;
        readings.extend(reader.read_pages(&images)?);
    }
    let read_s = t.elapsed().as_secs_f64();
    for (stem, r) in stems.iter().zip(&readings) {
        let gt: serde_json::Value = serde_json::from_str(&fs::read_to_string(dir.join(format!("{stem}.json")))?)?;
        for c in &r.crops {
            phases[0] += c.output.vision_ms;
            phases[1] += c.output.prefill_ms;
            phases[2] += c.output.decode_ms;
            read_tokens += c.output.tokens.len();
        }
        let text: String = r.crops.iter().map(|c| squeeze(&c.output.text)).collect();
        let longest = r.crops.iter().max_by_key(|c| c.output.tokens.len());
        println!(
            "{stem}: 조각 {} · 패치 {} · OCR {:.1} s · 토큰 {} · 가장 긴 조각 {} 토큰{}",
            r.crops.len(),
            r.plan.crops.iter().map(|c| c.patches()).sum::<u32>(),
            r.timings.ocr_ms / 1e3,
            r.crops.iter().map(|c| c.output.tokens.len()).sum::<usize>(),
            longest.map_or(0, |c| c.output.tokens.len()),
            if r.crops.iter().any(|c| c.output.truncated) { " · 잘림 있음" } else { "" },
        );
        let doc = gt["doc"].as_str().unwrap().to_string();
        let row = by_doc.entry(doc).or_default();
        row.2 += 1;
        row.3 += r.timings.ocr_ms;
        row.4 += r.timings.siglip2_ms;
        row.5 += r.plan.crops.iter().map(|c| c.patches() as u64).sum::<u64>();
        for g in gt["gt"].as_array().unwrap() {
            let (kind, want) = (g["kind"].as_str().unwrap(), squeeze(g["text"].as_str().unwrap_or("")));
            if want.is_empty() {
                continue;
            }
            let found = text.contains(&want);
            let chars: Vec<char> = text.chars().collect();
            let wanted: Vec<char> = want.chars().collect();
            let distance = best_distance(&wanted, &chars);
            match kind {
                "value" => {
                    total += 1;
                    row.1 += 1;
                    if found {
                        hit += 1;
                        row.0 += 1;
                    } else {
                        let close = distance * 5 <= wanted.len();
                        if close {
                            misread += 1;
                        } else {
                            skipped += 1;
                        }
                        if misses.len() < 40 {
                            misses.push(format!("{stem}: {} {want}", if close { "오독" } else { "건너뜀" }));
                        }
                    }
                }
                "title" => {
                    title_total += 1;
                    title_hit += found as usize;
                }
                _ => {}
            }
        }
    }
    println!("\n{:<18}{:>9}{:>8}{:>10}{:>8}", "서식", "값", "장", "OCR s", "패치");
    for (doc, r) in &by_doc {
        println!("{doc:<18}{:>4}/{:<4}{:>8}{:>10.1}{:>8}", r.0, r.1, r.2, r.3 / r.2 as f64 / 1e3, r.5 / r.2 as u64);
    }
    println!(
        "\n값 그대로 읽힘 {hit}/{total} ({:.1}%) · 오독 {misread} · 건너뜀 {skipped} · 제목 {title_hit}/{title_total} · 장당 {:.2} s",
        100.0 * hit as f32 / total.max(1) as f32,
        read_s / stems.len().max(1) as f64
    );
    let pages = stems.len().max(1) as f64;
    println!(
        "장당 비전 {:.2} s · 프리필 {:.2} s · 디코딩 {:.2} s · 나머지(적재 등) {:.2} s · 토큰 {:.0}",
        phases[0] / pages / 1e3,
        phases[1] / pages / 1e3,
        phases[2] / pages / 1e3,
        read_s / pages - phases.iter().sum::<f64>() / pages / 1e3,
        read_tokens as f64 / pages
    );
    for m in misses {
        println!("  {m}");
    }
    accel::finish_process(&device, 0);
    Ok(())
}
