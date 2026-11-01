//! siglip2 특징을 파일로 뽑습니다. 앵커 문구를 고를 때 GPU 없이 반복해서 분석하려고 씁니다.
//!
//!   siglip2_dump grids <이미지 폴더> <최대 패치> <출력 폴더>    페이지마다 <이름>.p<최대 패치>.bin
//!   siglip2_dump phrases <문구 파일> <출력.bin>                 한 줄에 문구 하나
//!
//! 파일 형식 (little endian): grids 는 u32 rows, cols, dim 뒤에 f32 patches[rows*cols*dim], pooled[dim].
//! phrases 는 u32 n, dim 뒤에 f32 vectors[n*dim].

use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Result};
use zorogravity_core::ai::accel;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::siglip2::image::prepare;
use zorogravity_core::ai::siglip2::Siglip2;

fn write_f32(out: &mut impl Write, v: &[f32]) -> Result<()> {
    for x in v {
        out.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let device = accel::select_device()?;
    println!("장치 {}", accel::backend_name(&device));
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let model = Siglip2::open(Path::new("models/siglip2"), residency)?;
    match args.first().map(String::as_str) {
        Some("grids") if args.len() == 4 => {
            let max: usize = args[2].parse()?;
            fs::create_dir_all(&args[3])?;
            let mut paths: Vec<_> = fs::read_dir(&args[1])?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|x| x == "jpg" || x == "png"))
                .collect();
            paths.sort();
            let t = Instant::now();
            for p in &paths {
                let img = image::open(p)?.to_rgb8();
                let input = prepare(&img, model.config().patch, max)?;
                let grid = model.encode_image(&input)?;
                let name = format!("{}.p{max}.bin", p.file_stem().unwrap().to_string_lossy());
                let mut out = std::io::BufWriter::new(fs::File::create(Path::new(&args[3]).join(name))?);
                for v in [grid.rows, grid.cols, grid.dim] {
                    out.write_all(&(v as u32).to_le_bytes())?;
                }
                write_f32(&mut out, &grid.patches)?;
                write_f32(&mut out, &grid.pooled)?;
            }
            println!("{}장 · 장당 {:.2} s", paths.len(), t.elapsed().as_secs_f64() / paths.len().max(1) as f64);
        }
        Some("phrases") if args.len() == 3 => {
            let text = fs::read_to_string(&args[1])?;
            let phrases: Vec<String> = text.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string).collect();
            let t = Instant::now();
            let vectors = model.encode_phrases(&phrases)?;
            let mut out = std::io::BufWriter::new(fs::File::create(&args[2])?);
            out.write_all(&(phrases.len() as u32).to_le_bytes())?;
            out.write_all(&(model.config().dim() as u32).to_le_bytes())?;
            for v in &vectors {
                write_f32(&mut out, v)?;
            }
            println!("{}개 · {:.2} s", phrases.len(), t.elapsed().as_secs_f64());
        }
        _ => bail!("사용법: siglip2_dump grids <폴더> <최대 패치> <출력 폴더> | phrases <파일> <출력.bin>"),
    }
    accel::finish_process(&device, 0);
    Ok(())
}
