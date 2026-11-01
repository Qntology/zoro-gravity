//! 가중치 올리기 시간을 묶음별로 잽니다. 판단 · 문서 읽기에서 쓰는 단계 순서대로 모델을 바꿔 가며 올립니다.
//!
//!   load_bench [횟수] [모델 폴더]
//!
//! 첫 회는 파일이 OS 캐시에 있는지에 따라 달라지고, 그다음부터는 캐시에 있는 값입니다.

use std::path::Path;

use anyhow::Result;
use zorogravity_core::ai::accel;
use zorogravity_core::ai::gocr::{self, Strategy};
use zorogravity_core::ai::model_store::models_dir;
use zorogravity_core::ai::ocr::{language_phase, vision_phase};
use zorogravity_core::ai::residency::{Action, Budget, Phase, Residency};
use zorogravity_core::ai::siglip2::Siglip2;
use zorogravity_core::ai::zoro::ZoroBackbone;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let rounds: usize = args.first().map(|s| s.parse()).transpose()?.unwrap_or(4);
    let models = args.get(1).cloned().unwrap_or_else(|| models_dir().display().to_string());
    let root = Path::new(&models);

    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let zoro = ZoroBackbone::open(&root.join("zoro"), residency.clone())?;
    let ocr = gocr::open(&root.join("gocr"), residency.clone(), 16, Strategy::Autoregressive)?;
    let crop = Siglip2::open(&root.join("siglip2"), residency.clone())?;
    let blocks = ocr.spec().file.meta_usize("gocr.block_count")?;

    let plan = zoro.model().plan().to_vec();
    let steps: Vec<(&str, _, Vec<Phase>)> = vec![
        ("zoro 판단", zoro.model().spec().clone(), plan),
        ("siglip2 비전", crop.spec().clone(), vec![zorogravity_core::ai::siglip2::vision_phase()]),
        ("gocr 비전", ocr.spec().clone(), vec![vision_phase()]),
        ("gocr 언어", ocr.spec().clone(), vec![language_phase(blocks)]),
    ];

    let mut res = residency.lock().unwrap();
    for round in 0..rounds {
        let start = res.history().len();
        let mut line = Vec::new();
        for (name, spec, phases) in &steps {
            let before = res.history().len();
            for phase in phases {
                res.enter(spec, phase)?;
            }
            let (mut ms, mut bytes) = (0u64, 0u64);
            for t in &res.history()[before..] {
                if t.action == Action::Load {
                    ms += t.millis;
                    bytes += t.bytes;
                }
            }
            line.push(format!("{name} {ms} ms ({:.0} MB, {:.1} GB/s)", bytes as f64 / 1e6, bytes as f64 / 1e9 / (ms.max(1) as f64 / 1e3)));
        }
        let released: u64 = res.history()[start..].iter().filter(|t| t.action == Action::Release).map(|t| t.millis).sum();
        let loaded: u64 = res.history()[start..].iter().filter(|t| t.action == Action::Load).map(|t| t.millis).sum();
        println!("[{round}] {} · 올리기 합 {loaded} ms · 내리기 합 {released} ms", line.join(" · "));
        res.release_all()?;
    }
    drop(res);
    accel::finish_process(&device, 0);
    Ok(())
}
