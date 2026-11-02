//! 시나리오 서류를 끝까지 읽어 조각 글자를 저장합니다. 항목 추출은 이 파일로 GPU 없이 되풀이합니다.
//!
//!   case_read <서류 폴더 (gen_case_docs 출력)> <출력 폴더> [시나리오 이름에 들어갈 글자]
//!
//! 시나리오 한 건을 민원 한 건의 서류 묶음으로 읽습니다 (`read_pages`). 이미 읽은 페이지는 건너뜁니다.

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};
use serde_json::json;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::document::DocumentReader;
use zorogravity_core::ai::model_store::models_dir;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::siglip2::plan::CropKind;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = Path::new(&args[0]);
    let out = Path::new(&args[1]);
    let filter = args.get(2).cloned().unwrap_or_default();
    fs::create_dir_all(out)?;

    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let reader = DocumentReader::open(&models_dir(), residency)?;

    let mut cases: Vec<String> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .filter(|n| n.contains(&filter))
        .collect();
    cases.sort();

    let started = Instant::now();
    let mut pages_done = 0usize;
    for case in cases {
        let target = out.join(&case);
        fs::create_dir_all(&target)?;
        let mut names: Vec<String> = fs::read_dir(dir.join(&case))?
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".jpg") && !target.join(n.replace(".jpg", ".json")).exists())
            .collect();
        names.sort();
        if names.is_empty() {
            continue;
        }
        let images: Vec<_> = names.iter().map(|n| Ok(image::open(dir.join(&case).join(n)).with_context(|| n.clone())?.to_rgb8())).collect::<Result<_>>()?;
        let t = Instant::now();
        let readings = reader.read_pages(&images)?;
        for (name, r) in names.iter().zip(&readings) {
            let crops: Vec<_> = r
                .crops
                .iter()
                .map(|c| {
                    json!({
                        "kind": c.kind.name(),
                        "bbox": [c.bbox.0, c.bbox.1, c.bbox.2, c.bbox.3],
                        "text": c.output.text,
                        "mean": c.output.mean_confidence,
                        "min": c.output.min_confidence,
                        "confs": c.output.char_confidence,
                        "truncated": c.output.truncated,
                    })
                })
                .collect();
            let signs: Vec<_> = r.plan.crops.iter().filter(|c| c.kind == CropKind::Sign).map(|c| [c.bbox.0, c.bbox.1, c.bbox.2, c.bbox.3]).collect();
            let page = json!({ "case": case, "file": name, "skew": r.skew, "signs": signs, "crops": crops, "ms": r.timings.layout_ms + r.timings.siglip2_ms + r.timings.ocr_ms });
            fs::write(target.join(name.replace(".jpg", ".json")), serde_json::to_string_pretty(&page)?)?;
        }
        pages_done += names.len();
        println!("{case}: {}장 {:.1}초 (누적 {pages_done}장, {:.0}초)", names.len(), t.elapsed().as_secs_f64(), started.elapsed().as_secs_f64());
    }
    accel::finish_process(&device, 0);
    Ok(())
}
