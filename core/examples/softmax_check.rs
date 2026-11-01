//! softmax 점검: GPU softmax 가 실제로 GPU 에서 도는지, 결과가 CPU 와 맞는지, 호출당 시간이 얼마인지 봅니다.
//!
//! 실행 예:
//!   ZOROGRAVITY_DEVICE=vulkan CANDLE_VULKAN_DEBUG=1 cargo run -p zorogravity-core --example softmax_check
//!   ZOROGRAVITY_DEVICE=rocm cargo run -p zorogravity-core --no-default-features --features rocm --example softmax_check

use std::time::Instant;

use anyhow::Result;
use candle_core::{Device, Tensor};
use zorogravity_core::ai::{accel, verify};

/// 2048 × 1024 = 2M 원소. 포크의 GPU 임계값(32768 원소)보다 훨씬 커서 GPU 경로를 탑니다.
const ROWS: usize = 2048;
const COLS: usize = 1024;
const ITERS: usize = 50;

fn main() -> Result<()> {
    let dev = accel::select_device()?;
    println!("장치: {} · 크기 {ROWS}x{COLS}", accel::backend_name(&dev));

    let x = Tensor::from_vec(verify::det_normal(ROWS * COLS, 11), (ROWS, COLS), &Device::Cpu)?;
    let reference = candle_nn::ops::softmax_last_dim(&x)?;

    let xd = x.to_device(&dev)?;
    let got = candle_nn::ops::softmax_last_dim(&xd)?;
    println!("GPU 결과 종류: {:?}", got.device().location());
    let got_cpu = got.to_device(&Device::Cpu)?;
    println!("CPU 기준과 상대오차: {:.3e}", verify::rel_err(&got_cpu, &reference)?);

    // 시간 측정. 마지막 결과를 CPU 로 내려받아 GPU 작업이 끝났음을 보장합니다.
    let t = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(candle_nn::ops::softmax_last_dim(&xd)?);
    }
    std::hint::black_box(candle_nn::ops::softmax_last_dim(&xd)?.to_device(&Device::Cpu)?);
    let dev_us = t.elapsed().as_secs_f64() * 1e6 / (ITERS + 1) as f64;

    let t = Instant::now();
    for _ in 0..ITERS {
        std::hint::black_box(candle_nn::ops::softmax_last_dim(&x)?);
    }
    let cpu_us = t.elapsed().as_secs_f64() * 1e6 / ITERS as f64;
    println!("호출당 시간: 장치 {dev_us:.1} µs · CPU {cpu_us:.1} µs ({ITERS}회 평균)");

    // 같은 시험 셰이더 계열의 reduction (max · min). 이전 셰이더도 같은 무한대 리터럴을 썼습니다.
    let max_dev = xd.max(1)?.to_device(&Device::Cpu)?;
    let max_ref = x.max(1)?;
    let min_dev = xd.min(1)?.to_device(&Device::Cpu)?;
    let min_ref = x.min(1)?;
    println!(
        "reduce max 상대오차 {:.3e} · reduce min 상대오차 {:.3e}",
        verify::rel_err(&max_dev, &max_ref)?,
        verify::rel_err(&min_dev, &min_ref)?
    );
    accel::finish_process(&dev, 0);
    Ok(())
}
