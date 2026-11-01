//! 양자화 행렬곱 점검: 선택된 GGUF(기본 q8)의 실제 가중치로 GPU 와 CPU 의 결과를 비교하고 시간을 잽니다.
//!
//! 백본의 시간은 이런 큰 행렬곱이 거의 전부입니다. 판단 헤드는 너무 작아서 GPU 가 이득이 없지만, 여기서는 이득이 나옵니다.
//! 결과는 `ai::verify::trusted` 로 CPU 기준과 비교하고, 틀리면 그 연산은 GPU 에서 쓰지 않는다는 방식(logis-center 와 같음)을 그대로 씁니다.
//!
//! 실행 예:
//!   cargo run -p zorogravity-core --example qmatmul_check -- models/zoro            (active.json 의 파일, 장치는 auto)
//!   ZOROGRAVITY_DEVICE=vulkan cargo run -p zorogravity-core --example qmatmul_check -- models/zoro q4
//!   ZOROGRAVITY_DEVICE=rocm   cargo run -p zorogravity-core --no-default-features --features rocm --example qmatmul_check -- models/zoro

use std::fs::File;
use std::hint::black_box;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

use anyhow::Result;
use candle_core::quantized::gguf_file::Content;
use candle_core::quantized::{GgmlDType, QMatMul};
use candle_core::{Device, Module, Tensor};
use zorogravity_core::ai::gguf_model::load_active;
use zorogravity_core::ai::{accel, verify};

/// 양자화 행렬곱 결과의 허용 상대오차 (GPU 와 CPU 가 다른 순서로 더하는 차이 정도)
const TOL: f32 = 1e-2;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| "models/zoro".into());
    let dir = Path::new(&dir);
    let path = match args.next() {
        Some(tag) => {
            let name = dir.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            dir.join(format!("{name}-{tag}.gguf"))
        }
        None => load_active(dir)?.path,
    };
    let device = accel::select_device()?;
    println!("모델 파일: {} · 장치: {}", path.display(), accel::backend_name(&device));

    let mut reader = BufReader::new(File::open(&path)?);
    let content = Content::read(&mut reader)?;

    // 가장 큰 양자화 선형 가중치 두 개
    let mut picks: Vec<(String, usize, GgmlDType)> = content
        .tensor_infos
        .iter()
        .filter(|(name, info)| {
            name.starts_with("blk.")
                && info.shape.rank() == 2
                && matches!(info.ggml_dtype, GgmlDType::Q8_0 | GgmlDType::Q4K)
        })
        .map(|(name, info)| (name.clone(), info.shape.elem_count(), info.ggml_dtype))
        .collect();
    picks.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    picks.truncate(2);

    for (name, _, dtype) in picks {
        let q_cpu = content.tensor(&mut reader, &name, &Device::Cpu)?;
        let q_dev = content.tensor(&mut reader, &name, &device)?;
        let dims = q_cpu.shape().dims().to_vec();
        let (out_dim, in_dim) = (dims[0], dims[1]);
        // 정확한 기준: 가중치를 F32 로 풀어서 곱합니다 (활성값 양자화 없음)
        let w_exact = content.tensor(&mut reader, &name, &Device::Cpu)?.dequantize(&Device::Cpu)?;
        let m_cpu = QMatMul::from_qtensor(q_cpu)?;
        let m_dev = QMatMul::from_qtensor(q_dev)?;

        for rows in [1usize, 64] {
            let x = Tensor::from_vec(verify::det_normal(rows * in_dim, 0x51_0000 + rows as u64), (rows, in_dim), &Device::Cpu)?;
            let y_ref = m_cpu.forward(&x)?;
            let x_dev = x.to_device(&device)?;

            let op = format!("qmatmul/{name}/m{rows}");
            let ok = verify::trusted(&device, &op, TOL, || {
                let y = m_dev.forward(&x_dev)?.to_device(&Device::Cpu)?;
                verify::rel_err(&y, &y_ref)
            });
            let y_dev = m_dev.forward(&x_dev)?.to_device(&Device::Cpu)?;
            let err = verify::rel_err(&y_dev, &y_ref)?;
            let y_exact = x.matmul(&w_exact.t()?)?;
            let dev_vs_exact = verify::rel_err(&y_dev, &y_exact)?;
            let cpu_vs_exact = verify::rel_err(&y_ref, &y_exact)?;

            let iters = if rows == 1 { 50 } else { 10 };
            let t = Instant::now();
            for _ in 0..iters {
                black_box(m_dev.forward(&x_dev)?);
            }
            black_box(m_dev.forward(&x_dev)?.to_device(&Device::Cpu)?); // 마지막 결과를 내려받아 GPU 작업이 끝났음을 보장
            let dev_ms = t.elapsed().as_secs_f64() * 1e3 / (iters + 1) as f64;

            let t = Instant::now();
            for _ in 0..iters {
                black_box(m_cpu.forward(&x)?);
            }
            let cpu_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;

            println!(
                "{name:<26} [{out_dim}x{in_dim}] {dtype:?} 행 {rows:>2}: 장치↔CPU {err:.1e} ({}) · 장치↔정확 {dev_vs_exact:.1e} · CPU↔정확 {cpu_vs_exact:.1e} · 장치 {dev_ms:.3} ms · CPU {cpu_ms:.3} ms",
                if ok { "신뢰" } else { "CPU 로 대체" },
            );
        }
    }
    accel::finish_process(&device, 0);
    Ok(())
}
