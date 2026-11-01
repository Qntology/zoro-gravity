//! 점검: 같은 GPU 를 다른 프로그램이 쓰고 있어 VRAM 이 모자랄 때 무슨 일이 생기는지 보여 줍니다.
//!
//! 기본 설정에서는 `Residency` 가 자리가 모자란 적재를 거부합니다. 이 예제는 그 방어를 끄고(`ZOROGRAVITY_ALLOW_HOST_WEIGHTS=1`),
//! 양자화 가중치로 VRAM 을 채운 채 판단을 돌려, 일부 가중치만 GPU 에 올라가고 확률이 달라지는 것을 측정합니다.
//! (방어 없이 실제로 겪은 일: 같은 GPU 에서 zoro 프로세스 둘을 동시에 돌렸더니 확률이 실행마다 달랐습니다.)
//!
//!   set ZOROGRAVITY_ALLOW_HOST_WEIGHTS=1 && cargo run -p zorogravity-core --example pressure_probe
use std::path::Path;

use anyhow::Result;
use candle_core::quantized::ggml_file::qtensor_from_ggml;
use candle_core::quantized::{GgmlDType, QTensor};
use zorogravity_core::ai::accel;
use zorogravity_core::ai::residency::{Budget, Residency};
use zorogravity_core::ai::zoro::open_decision_model;
use zorogravity_core::decision::{DecisionModel, DecisionTask, Fact};
use zorogravity_core::state_machine::gate3_substantive::OPTIONS;

fn main() -> Result<()> {
    let device = accel::select_device()?;
    let res = Residency::shared(device.clone(), Budget::unlimited());
    let model = open_decision_model(Path::new("models/zoro"), res.clone())?;
    let task = DecisionTask {
        title: "임차 요건 심사".into(),
        facts: vec![
            Fact::new("적용 규칙", "임차인과 임대인이 2촌 이내 친족이거나 같은 세대·같은 계좌이면 지급 제외"),
            Fact::new("가까운 연결 경로", "없음"),
            Fact::new("동명이인 후보", "아니오"),
        ],
        options: OPTIONS.to_vec(),
    };
    let show = |label: &str, p: &[f32]| println!("{label:<28} {:.7}/{:.7}/{:.7}", p[0], p[1], p[2]);
    let base = model.probabilities(&task)?;
    show("여유 충분", &base);
    res.lock().unwrap().release_all()?;

    let (rows, cols) = (8192usize, 8192usize);
    let bytes = vec![0u8; rows * cols / 32 * 34];
    let each = (rows * cols / 32 * 36) as u64;
    for leave in [2048u64, 3000, 3400] {
        let free = device.mem_info()?.unwrap().0 as u64;
        let n = free.saturating_sub(leave << 20) / each;
        let hog: Vec<QTensor> = (0..n).map(|_| qtensor_from_ggml(GgmlDType::Q8_0, &bytes, vec![rows, cols], &device).unwrap()).collect();
        let before = device.mem_info()?.unwrap().0 >> 20;
        let p = model.probabilities(&task)?;
        let after = device.mem_info()?.unwrap().0 >> 20;
        show(&format!("여유 {leave} MB 로 줄임"), &p);
        println!("{:<28} 적재로 줄어든 장치 여유 {} MB (전부 올라가면 약 3846 MB)", "", before - after);
        let diff = base.iter().zip(&p).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("{:<28} 기준과의 최대 차이 {diff:.2e}", "");
        res.lock().unwrap().release_all()?;
        drop(hog);
        device.trim_memory_pool(0)?;
    }
    accel::finish_process(&device, 0);
    Ok(())
}
