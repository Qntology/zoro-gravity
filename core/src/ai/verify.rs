//! GPU 커널 자가 검증 (logis-center `models/common/accel_check.rs` 와 같은 방식).
//! 장치와 연산 조합마다 처음 쓰기 전에 한 번만 CPU 기준 계산과 비교합니다.
//! 틀리면 그 장치의 그 연산만 CPU 로 돌리고, 판정은 프로세스가 끝날 때까지 캐시합니다.

use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Mutex, MutexGuard, OnceLock};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use super::accel;

/// 자가 검증을 끄는 환경 변수 (`off` / `0` / `false` …). logis-center 의 `LOGIS_ACCEL_CHECK` 와 같은 역할.
pub const CHECK_ENV: &str = "ZOROGRAVITY_ACCEL_CHECK";

pub fn checks_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var(CHECK_ENV) {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no" | "disable" | "disabled"
        ),
        Err(_) => true,
    })
}

fn verdicts() -> MutexGuard<'static, HashMap<String, bool>> {
    static V: OnceLock<Mutex<HashMap<String, bool>>> = OnceLock::new();
    V.get_or_init(|| Mutex::new(HashMap::new())).lock().unwrap_or_else(|p| p.into_inner())
}

/// `op` 연산을 `device` 에서 써도 되는지 정합니다.
///
/// `probe` 는 장치에서 계산한 결과와 CPU 기준값의 상대오차를 돌려줍니다. 같은 (장치, 연산) 에 대해 한 번만 호출됩니다.
/// CPU 이거나 `ZOROGRAVITY_ACCEL_CHECK=off` 이면 검증 없이 true 입니다.
pub fn trusted(device: &Device, op: &str, tol: f32, probe: impl FnOnce() -> Result<f32>) -> bool {
    if !accel::is_gpu(device) || !checks_enabled() {
        return true;
    }
    let key = format!("{}#{:?}/{op}", accel::backend_name(device), device.location());
    if let Some(ok) = verdicts().get(&key) {
        return *ok;
    }
    let ok = match catch_unwind(AssertUnwindSafe(probe)) {
        Ok(Ok(err)) if err <= tol => {
            eprintln!("[ZOROGRAVITY-ACCEL] 통과 · {key} (상대오차 {err:.1e} ≤ {tol:.0e})");
            true
        }
        Ok(Ok(err)) => {
            eprintln!("[ZOROGRAVITY-ACCEL] ❌ {key}: 상대오차 {err:.2e} > {tol:.0e} → 이 연산은 CPU 로 실행합니다");
            false
        }
        Ok(Err(e)) => {
            eprintln!("[ZOROGRAVITY-ACCEL] ❌ {key}: 검증 실행 실패 ({e}) → CPU 로 실행합니다");
            false
        }
        Err(_) => {
            eprintln!("[ZOROGRAVITY-ACCEL] ❌ {key}: 검증 중 패닉 → CPU 로 실행합니다");
            false
        }
    };
    verdicts().insert(key, ok);
    ok
}

/// 결정적 정규분포 난수 (xorshift64 + Box–Muller). 장치와 무관하게 같은 검증 입력을 만듭니다.
pub fn det_normal(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03;
    if s == 0 {
        s = 1;
    }
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            let (u1, u2) = (next(), next());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect()
}

/// 상대오차 ‖a−b‖/‖b‖ (f32). `b` 가 CPU 기준값입니다. NaN/Inf 가 섞이면 Inf.
pub fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?;
    let b = b.to_dtype(DType::F32)?.to_device(a.device())?;
    let d = (&a - &b)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let n = b.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let r = (d / n.max(1e-30)).sqrt();
    Ok(if r.is_finite() { r } else { f32::INFINITY })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det_normal_is_deterministic_and_centered() {
        let a = det_normal(4096, 7);
        let b = det_normal(4096, 7);
        assert_eq!(a, b);
        let mean = a.iter().sum::<f32>() / a.len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert_ne!(det_normal(16, 1), det_normal(16, 2));
    }

    #[test]
    fn rel_err_is_zero_for_equal_and_inf_for_nan() -> Result<()> {
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0], 3, &Device::Cpu)?;
        assert_eq!(rel_err(&x, &x)?, 0.0);
        let nan = Tensor::from_vec(vec![f32::NAN, 2.0, 3.0], 3, &Device::Cpu)?;
        assert_eq!(rel_err(&nan, &x)?, f32::INFINITY);
        Ok(())
    }

    #[test]
    fn cpu_is_always_trusted_without_probing() {
        let ok = trusted(&Device::Cpu, "never-probed", 0.0, || panic!("CPU 는 검증하지 않습니다"));
        assert!(ok);
    }
}
