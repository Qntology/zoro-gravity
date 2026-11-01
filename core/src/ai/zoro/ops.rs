//! zoro 백본의 게이트 델타 규칙 선형 주의. 모두 F32 입니다. RMSNorm · RoPE · 인과 주의는 `ai::attention` 에 있습니다.
//!
//! 구조는 config 와 가중치 모양에서 읽었습니다 (docs/zoro-backbone.md). 장치 전용 커널은 처음 쓰기 전에 CPU 기준과 비교해
//! 맞을 때만 쓰고, 틀리거나 없으면 CPU 커널로 계산합니다 (`ai::verify`).

use anyhow::{bail, Result};
use candle_core::delta_rule::{fused_gated_delta_rule, fused_gated_rms_norm, fused_gdn_supported_for, GdnConfig};
use candle_core::{Device, Tensor};

use crate::ai::verify;

// ---------------------------------------------------------------------------------------------
// 게이트 델타 규칙 선형 주의의 핵심 (컨볼루션 → 순환 → 게이트 정규화)
// ---------------------------------------------------------------------------------------------

/// 선형 주의 층의 작은 텐서들. 장치 커널을 쓰면 장치에, 아니면 CPU 에 둔 사본을 씁니다.
pub struct GdnWeights {
    /// `[C, 1, K]` 깊이별 컨볼루션
    pub conv: Tensor,
    /// `-exp(A_log)` `[Hv]`
    pub neg_a: Tensor,
    /// `[Hv]`
    pub dt_bias: Tensor,
    /// 게이트 정규화 가중치 `[Dv]` (직접 곱함)
    pub norm: Tensor,
}

impl GdnWeights {
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        Ok(Self {
            conv: self.conv.to_device(device)?,
            neg_a: self.neg_a.to_device(device)?,
            dt_bias: self.dt_bias.to_device(device)?,
            norm: self.norm.to_device(device)?,
        })
    }
}

pub fn gdn_config(cfg: &super::ZoroConfig) -> GdnConfig {
    let mut g = GdnConfig::new(cfg.gdn_k_heads, cfg.gdn_v_heads, cfg.gdn_k_dim, cfg.gdn_v_dim);
    g.conv_kernel = cfg.conv_kernel;
    g
}

/// 이 장치에서 융합 커널로 계산해도 되는가. 커널이 없으면 false, 있어도 CPU 기준과 다르면 false 입니다.
pub fn gdn_on_device(device: &Device, cfg: &GdnConfig) -> bool {
    if device.is_cpu() || !fused_gdn_supported_for(device, cfg) {
        return device.is_cpu();
    }
    verify::trusted(device, "gated_delta_rule", 1e-3, || {
        // 작은 문제로 장치 커널과 CPU 커널을 비교합니다 (기하는 실제 층과 같고 토큰만 적습니다).
        let t = 6;
        let c = cfg.conv_dim();
        let hv = cfg.num_v_heads;
        let det = |n: usize, seed: u64| verify::det_normal(n, seed);
        let mk = |data: Vec<f32>, shape: &[usize], d: &Device| Tensor::from_vec(data, shape.to_vec(), d);
        let build = |d: &Device| -> candle_core::Result<GdnWeights> {
            Ok(GdnWeights {
                conv: mk(det(c * cfg.conv_kernel, 21).iter().map(|x| x * 0.3).collect(), &[c, 1, cfg.conv_kernel], d)?,
                neg_a: mk(det(hv, 22).iter().map(|x| -(x.abs() + 0.1)).collect(), &[hv], d)?,
                dt_bias: mk(det(hv, 23), &[hv], d)?,
                norm: mk(det(cfg.head_v_dim, 24).iter().map(|x| 1.0 + 0.1 * x).collect(), &[cfg.head_v_dim], d)?,
            })
        };
        let run = |d: &Device| -> Result<Tensor> {
            let w = build(d)?;
            let mixed = mk(det(t * c, 25), &[t, c], d)?;
            let z = mk(det(t * cfg.value_dim(), 26), &[t, cfg.value_dim()], d)?;
            let a = mk(det(t * hv, 27), &[t, hv], d)?;
            let b = mk(det(t * hv, 28), &[t, hv], d)?;
            let y = gated_delta_net(&mixed, &z, &a, &b, &w, cfg, 1e-6)?;
            Ok(y.to_device(&Device::Cpu)?)
        };
        verify::rel_err(&run(device)?, &run(&Device::Cpu)?)
    })
}

/// 이어서 읽을 때 넘기는 선형 주의 상태: 컨볼루션의 지난 입력 `[1, C, K-1]` 과 순환 상태 `[1, Hv, Dk, Dv]`
#[derive(Clone)]
pub struct GdnState {
    pub conv: Tensor,
    pub recurrent: Tensor,
}

/// 선형 주의 한 층의 핵심.
/// - `mixed`: 입력 투영 `[T, 2·Hk·Dk + Hv·Dv]` (q, k, v 순서)
/// - `z`: 출력 게이트 `[T, Hv·Dv]`, `a`·`b`: `[T, Hv]`
///
/// 깊이별 인과 컨볼루션 + SiLU → q·k 를 L2 정규화 → 게이트 `g = −exp(A_log)·softplus(a + dt_bias)`, `β = σ(b)` 로
/// 델타 규칙 순환 → 게이트 RMSNorm(`z`) 을 거쳐 `[T, Hv·Dv]` 를 돌려줍니다. 모든 텐서는 같은 장치에 있어야 합니다.
pub fn gated_delta_net(
    mixed: &Tensor,
    z: &Tensor,
    a: &Tensor,
    b: &Tensor,
    w: &GdnWeights,
    cfg: &GdnConfig,
    eps: f64,
) -> Result<Tensor> {
    Ok(gated_delta_step(mixed, z, a, b, w, cfg, eps, None)?.0)
}

/// [`gated_delta_net`] 을 이어서 읽습니다. `state` 가 없으면 처음부터 읽고, 다음 호출에 넘길 상태를 함께 돌려줍니다.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_step(
    mixed: &Tensor,
    z: &Tensor,
    a: &Tensor,
    b: &Tensor,
    w: &GdnWeights,
    cfg: &GdnConfig,
    eps: f64,
    state: Option<&GdnState>,
) -> Result<(Tensor, GdnState)> {
    let (t, c) = mixed.dims2()?;
    if c != cfg.conv_dim() {
        bail!("선형 주의 입력 채널 {c} 가 기대값 {} 와 다릅니다", cfg.conv_dim());
    }
    let out = fused_gated_delta_rule(
        &mixed.unsqueeze(0)?,
        &w.conv,
        None,
        state.map(|s| &s.conv),
        &a.unsqueeze(0)?,
        &b.unsqueeze(0)?,
        &w.neg_a,
        &w.dt_bias,
        state.map(|s| &s.recurrent),
        cfg,
    )?;
    let z = z.reshape((1, t, cfg.num_v_heads, cfg.head_v_dim))?;
    let y = fused_gated_rms_norm(&out.out, &z, &w.norm, eps)?;
    Ok((y.reshape((t, cfg.value_dim()))?, GdnState { conv: out.conv_state, recurrent: out.recurrent_state }))
}

/// 출력 없이 순환 상태만 구합니다. 읽은 토큰 앞부분만 문맥에 넣을 때 그 자리의 상태를 얻는 데 씁니다.
pub fn gated_delta_state(mixed: &Tensor, a: &Tensor, b: &Tensor, w: &GdnWeights, cfg: &GdnConfig, state: Option<&GdnState>) -> Result<GdnState> {
    let out = fused_gated_delta_rule(
        &mixed.unsqueeze(0)?,
        &w.conv,
        None,
        state.map(|s| &s.conv),
        &a.unsqueeze(0)?,
        &b.unsqueeze(0)?,
        &w.neg_a,
        &w.dt_bias,
        state.map(|s| &s.recurrent),
        cfg,
    )?;
    Ok(GdnState { conv: out.conv_state, recurrent: out.recurrent_state })
}

/// [`gated_delta_state`] 의 CPU 경로. 상태는 CPU 에 둡니다.
pub fn gated_delta_state_on_host(mixed: &Tensor, a: &Tensor, b: &Tensor, host: &GdnWeights, cfg: &GdnConfig, state: Option<&GdnState>) -> Result<GdnState> {
    let cpu = Device::Cpu;
    gated_delta_state(&mixed.to_device(&cpu)?, &a.to_device(&cpu)?, &b.to_device(&cpu)?, host, cfg, state)
}

/// 장치 커널이 없을 때: 입력을 CPU 로 내려 CPU 커널로 계산하고 결과만 장치로 올립니다.
pub fn gated_delta_net_on_host(
    mixed: &Tensor,
    z: &Tensor,
    a: &Tensor,
    b: &Tensor,
    host: &GdnWeights,
    cfg: &GdnConfig,
    eps: f64,
) -> Result<Tensor> {
    Ok(gated_delta_step_on_host(mixed, z, a, b, host, cfg, eps, None)?.0)
}

/// [`gated_delta_net_on_host`] 를 이어서 읽습니다. 상태는 CPU 에 둡니다.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_step_on_host(
    mixed: &Tensor,
    z: &Tensor,
    a: &Tensor,
    b: &Tensor,
    host: &GdnWeights,
    cfg: &GdnConfig,
    eps: f64,
    state: Option<&GdnState>,
) -> Result<(Tensor, GdnState)> {
    let device = mixed.device().clone();
    let cpu = Device::Cpu;
    let (y, next) = gated_delta_step(
        &mixed.to_device(&cpu)?,
        &z.to_device(&cpu)?,
        &a.to_device(&cpu)?,
        &b.to_device(&cpu)?,
        host,
        cfg,
        eps,
        state,
    )?;
    Ok((y.to_device(&device)?, next))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::verify::{det_normal, rel_err};
    use crate::ai::zoro::reference::naive_gdn;
    use candle_core::DType;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape.to_vec(), &Device::Cpu).unwrap()
    }

    pub(crate) fn small_gdn() -> GdnConfig {
        let mut g = GdnConfig::new(2, 4, 8, 8);
        g.conv_kernel = 4;
        g
    }

    #[test]
    fn the_gated_delta_net_matches_the_scalar_reference() -> Result<()> {
        let cfg = small_gdn();
        let (tt, c, hv, vd) = (9usize, cfg.conv_dim(), cfg.num_v_heads, cfg.value_dim());
        let mixed = det_normal(tt * c, 31);
        let z = det_normal(tt * vd, 32);
        let a = det_normal(tt * hv, 33);
        let b = det_normal(tt * hv, 34);
        let conv: Vec<f32> = det_normal(c * cfg.conv_kernel, 35).iter().map(|x| x * 0.5).collect();
        let neg_a: Vec<f32> = det_normal(hv, 36).iter().map(|x| -(x.abs() + 0.05)).collect();
        let dt_bias = det_normal(hv, 37);
        let norm: Vec<f32> = det_normal(cfg.head_v_dim, 38).iter().map(|x| 1.0 + 0.2 * x).collect();

        let w = GdnWeights {
            conv: t(conv.clone(), &[c, 1, cfg.conv_kernel]),
            neg_a: t(neg_a.clone(), &[hv]),
            dt_bias: t(dt_bias.clone(), &[hv]),
            norm: t(norm.clone(), &[cfg.head_v_dim]),
        };
        let run = |mixed: &[f32]| {
            gated_delta_net(
                &t(mixed.to_vec(), &[tt, c]),
                &t(z.clone(), &[tt, vd]),
                &t(a.clone(), &[tt, hv]),
                &t(b.clone(), &[tt, hv]),
                &w,
                &cfg,
                1e-6,
            )
        };
        let got = run(&mixed)?;
        assert_eq!(got.dims(), &[tt, vd]);
        assert_eq!(got.dtype(), DType::F32);
        let want = naive_gdn(&mixed, &z, &a, &b, &conv, &neg_a, &dt_bias, &norm, &cfg, tt, 1e-6);
        let err = rel_err(&got, &t(want, &[tt, vd]))?;
        assert!(err < 1e-4, "스칼라 기준과의 상대오차 {err}");

        // 인과성: 마지막 토큰 입력만 바꿔도 앞 토큰 출력은 같습니다.
        let mut mixed2 = mixed.clone();
        mixed2[(tt - 1) * c..].iter_mut().for_each(|e| *e += 3.0);
        let got2 = run(&mixed2)?;
        let head_diff = (got.narrow(0, 0, tt - 1)? - got2.narrow(0, 0, tt - 1)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(head_diff, 0.0);
        Ok(())
    }

    #[test]
    fn stepping_with_state_equals_reading_everything_at_once() -> Result<()> {
        let cfg = small_gdn();
        let (tt, c, hv, vd) = (9usize, cfg.conv_dim(), cfg.num_v_heads, cfg.value_dim());
        let w = GdnWeights {
            conv: t(det_normal(c * 4, 51).iter().map(|x| x * 0.5).collect(), &[c, 1, 4]),
            neg_a: t(det_normal(hv, 52).iter().map(|x| -(x.abs() + 0.05)).collect(), &[hv]),
            dt_bias: t(det_normal(hv, 53), &[hv]),
            norm: t(vec![1.0; 8], &[8]),
        };
        let (mixed, z) = (t(det_normal(tt * c, 54), &[tt, c]), t(det_normal(tt * vd, 55), &[tt, vd]));
        let (a, b) = (t(det_normal(tt * hv, 56), &[tt, hv]), t(det_normal(tt * hv, 57), &[tt, hv]));
        let whole = gated_delta_net(&mixed, &z, &a, &b, &w, &cfg, 1e-6)?;
        let mut state = None;
        let mut parts = Vec::new();
        let mut at = 0;
        for n in [4usize, 1, 3, 1] {
            let cut = |x: &Tensor| x.narrow(0, at, n).and_then(|v| v.contiguous());
            let (y, next) = gated_delta_step(&cut(&mixed)?, &cut(&z)?, &cut(&a)?, &cut(&b)?, &w, &cfg, 1e-6, state.as_ref())?;
            parts.push(y);
            state = Some(next);
            at += n;
        }
        let err = rel_err(&Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0)?, &whole)?;
        assert!(err < 1e-5, "{err}");
        Ok(())
    }

    #[test]
    fn the_state_alone_equals_the_state_of_a_full_step() -> Result<()> {
        let cfg = small_gdn();
        let (tt, c, hv, vd) = (6usize, cfg.conv_dim(), cfg.num_v_heads, cfg.value_dim());
        let w = GdnWeights {
            conv: t(det_normal(c * 4, 61).iter().map(|x| x * 0.5).collect(), &[c, 1, 4]),
            neg_a: t(det_normal(hv, 62).iter().map(|x| -(x.abs() + 0.05)).collect(), &[hv]),
            dt_bias: t(det_normal(hv, 63), &[hv]),
            norm: t(vec![1.0; 8], &[8]),
        };
        let (mixed, z) = (t(det_normal(tt * c, 64), &[tt, c]), t(det_normal(tt * vd, 65), &[tt, vd]));
        let (a, b) = (t(det_normal(tt * hv, 66), &[tt, hv]), t(det_normal(tt * hv, 67), &[tt, hv]));
        let (_, full) = gated_delta_step(&mixed, &z, &a, &b, &w, &cfg, 1e-6, None)?;
        for s in [gated_delta_state(&mixed, &a, &b, &w, &cfg, None)?, gated_delta_state_on_host(&mixed, &a, &b, &w, &cfg, None)?] {
            assert_eq!(rel_err(&s.conv, &full.conv)?, 0.0);
            assert_eq!(rel_err(&s.recurrent, &full.recurrent)?, 0.0);
        }
        Ok(())
    }

    #[test]
    fn the_host_path_equals_the_direct_path_on_cpu() -> Result<()> {
        let cfg = small_gdn();
        let (tt, c, hv, vd) = (5usize, cfg.conv_dim(), cfg.num_v_heads, cfg.value_dim());
        let w = GdnWeights {
            conv: t(det_normal(c * 4, 41), &[c, 1, 4]),
            neg_a: t(det_normal(hv, 42).iter().map(|x| -x.abs()).collect(), &[hv]),
            dt_bias: t(det_normal(hv, 43), &[hv]),
            norm: t(vec![1.0; 8], &[8]),
        };
        let args = (
            t(det_normal(tt * c, 44), &[tt, c]),
            t(det_normal(tt * vd, 45), &[tt, vd]),
            t(det_normal(tt * hv, 46), &[tt, hv]),
            t(det_normal(tt * hv, 47), &[tt, hv]),
        );
        let direct = gated_delta_net(&args.0, &args.1, &args.2, &args.3, &w, &cfg, 1e-6)?;
        let host = gated_delta_net_on_host(&args.0, &args.1, &args.2, &args.3, &w, &cfg, 1e-6)?;
        assert_eq!(rel_err(&host, &direct)?, 0.0);
        assert!(gdn_on_device(&Device::Cpu, &cfg));
        Ok(())
    }
}
