//! Zoro-1 판단 헤드 `jev_head` (명세 Part 3 §3.2 의 Switch A).
//!
//! 실제 가중치 구조 (`models/zoro/jev_head.pt` 를 pickletools 로 확인했고, F32 safetensors 로 바꿔 둠):
//! - `q.weight` [256, 2560], `q.bias` [256]: 질의(문맥) 은닉 상태 h → 256차원 질의 q
//! - `k.weight` [256, 2560], `k.bias` [256]: 선택지 은닉 상태 e_i → 256차원 키 k_i
//!
//! 선택지 i 의 점수는 `s_i = q · k_i × 스케일` 이고, softmax 로 확률을 냅니다.
//! 어휘 사전(lm_head) 행렬곱은 없습니다. 명세의 "0-Generation 포인터 판단"입니다.
//!
//! 스케일 1/32: 정답 사례로 보정한 값입니다 (docs/jev-head.md 3절). 1/16 은 "0.95 이상"이라고 답한 질문의 76% 만 맞아 확률이 과신이고,
//! 1/32 는 평균 최고 확률이 실제 정답률에 가깝습니다. 학습 코드로 확인한 값은 아닙니다.
//! 질의와 선택지를 프롬프트의 어느 위치에서 뽑는지는 `ai::pointer` 가 정한 약속입니다.

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::softmax_last_dim;

/// 백본 은닉 차원 (zoro 텍스트 모델의 hidden_size)
pub const HIDDEN: usize = 2560;
/// 질의·키 차원
pub const QK_DIM: usize = 256;
/// 점수 스케일의 기본값. 정답 사례로 보정했습니다.
pub const DEFAULT_SCALE: f64 = 1.0 / 32.0;

pub struct JevHead {
    /// [HIDDEN, QK_DIM] 로 미리 전치해 둔 질의 가중치
    wq_t: Tensor,
    bq: Tensor,
    /// [HIDDEN, QK_DIM] 로 미리 전치해 둔 키 가중치
    wk_t: Tensor,
    bk: Tensor,
    /// 점수에 곱하는 스케일
    scale: f64,
}

fn check_shape(t: &Tensor, name: &str, expected: &[usize]) -> Result<()> {
    if t.dims() != expected {
        bail!("{name} 의 모양이 {:?} 이어야 하는데 {:?} 입니다", expected, t.dims());
    }
    Ok(())
}

impl JevHead {
    /// safetensors 파일 (`q.weight`, `q.bias`, `k.weight`, `k.bias`) 에서 헤드를 만듭니다.
    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let t = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("jev_head 를 읽을 수 없습니다: {}", path.display()))?;
        let get = |name: &str| -> Result<Tensor> {
            Ok(t.get(name)
                .with_context(|| format!("{name} 텐서가 없습니다"))?
                .to_dtype(DType::F32)?)
        };
        Self::from_tensors(get("q.weight")?, get("q.bias")?, get("k.weight")?, get("k.bias")?)
    }

    /// 가중치 네 개로 헤드를 만듭니다. 가중치는 PyTorch 와 같은 [out, in] 모양입니다.
    pub fn from_tensors(q_w: Tensor, q_b: Tensor, k_w: Tensor, k_b: Tensor) -> Result<Self> {
        check_shape(&q_w, "q.weight", &[QK_DIM, HIDDEN])?;
        check_shape(&q_b, "q.bias", &[QK_DIM])?;
        check_shape(&k_w, "k.weight", &[QK_DIM, HIDDEN])?;
        check_shape(&k_b, "k.bias", &[QK_DIM])?;
        Ok(Self {
            wq_t: q_w.to_dtype(DType::F32)?.t()?.contiguous()?,
            bq: q_b.to_dtype(DType::F32)?,
            wk_t: k_w.to_dtype(DType::F32)?.t()?.contiguous()?,
            bk: k_b.to_dtype(DType::F32)?,
            scale: DEFAULT_SCALE,
        })
    }

    /// 점수 스케일을 바꿉니다 (기본 1/32).
    pub fn with_scale(mut self, scale: f64) -> Result<Self> {
        if !(scale > 0.0 && scale.is_finite()) {
            bail!("스케일은 0 보다 큰 유한한 값이어야 합니다: {scale}");
        }
        self.scale = scale;
        Ok(self)
    }

    /// 질의 은닉 상태 `query` (길이 2560) 와 선택지 은닉 상태 `options` (행 우선, 길이 n×2560) 로
    /// 선택지별 확률을 냅니다. 합은 1 입니다.
    pub fn probabilities(&self, query: &[f32], options: &[f32]) -> Result<Vec<f32>> {
        let scores = self.dot(query, options)?.affine(self.scale, 0.0)?;
        let rows = softmax_last_dim(&scores)?.to_vec2::<f32>()?;
        rows.into_iter().next().context("헤드 출력이 비었습니다")
    }

    /// 스케일을 곱하기 전의 점수 `q · k_i`
    pub fn scores(&self, query: &[f32], options: &[f32]) -> Result<Vec<f32>> {
        self.dot(query, options)?.to_vec2::<f32>()?.into_iter().next().context("헤드 출력이 비었습니다")
    }

    fn dot(&self, query: &[f32], options: &[f32]) -> Result<Tensor> {
        if query.len() != HIDDEN {
            bail!("질의 길이 {} != {HIDDEN}", query.len());
        }
        if options.is_empty() || !options.len().is_multiple_of(HIDDEN) {
            bail!("선택지 은닉 상태 길이 {}는 {HIDDEN} 의 배수여야 합니다", options.len());
        }
        let n = options.len() / HIDDEN;
        let h = Tensor::from_slice(query, (1, HIDDEN), &Device::Cpu)?;
        let e = Tensor::from_slice(options, (n, HIDDEN), &Device::Cpu)?;
        let q = h.matmul(&self.wq_t)?.broadcast_add(&self.bq)?;
        let k = e.matmul(&self.wk_t)?.broadcast_add(&self.bk)?;
        Ok(q.matmul(&k.t()?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: Vec<f32>, shape: (usize, usize)) -> Tensor {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    /// 질의와 키가 같은 방향을 보는 가중치: 선택지 0 의 키가 질의와 가장 잘 맞습니다.
    fn aligned_head() -> Result<JevHead> {
        let mut w = vec![0f32; QK_DIM * HIDDEN];
        for i in 0..QK_DIM {
            w[i * HIDDEN + i] = 1.0; // 첫 256 개 입력 차원을 그대로 투영
        }
        let q_w = tensor(w.clone(), (QK_DIM, HIDDEN));
        let k_w = tensor(w, (QK_DIM, HIDDEN));
        let zeros = Tensor::zeros(QK_DIM, DType::F32, &Device::Cpu)?;
        JevHead::from_tensors(q_w, zeros.clone(), k_w, zeros)
    }

    #[test]
    fn scores_follow_the_query_key_dot_product() -> Result<()> {
        let head = aligned_head()?;
        let mut query = vec![0f32; HIDDEN];
        query[3] = 8.0;
        let mut options = vec![0f32; 3 * HIDDEN];
        options[HIDDEN + 3] = 8.0; // 선택지 1 이 질의와 같은 방향
        options[2 * HIDDEN + 10] = 8.0; // 선택지 2 는 직교
        let p = head.probabilities(&query, &options)?;
        assert_eq!(p.len(), 3);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[1] > p[0] && p[1] > p[2], "{p:?}");
        Ok(())
    }

    #[test]
    fn a_larger_scale_makes_the_distribution_peakier() -> Result<()> {
        let mut query = vec![0f32; HIDDEN];
        query[3] = 8.0;
        let mut options = vec![0f32; 2 * HIDDEN];
        options[3] = 8.0;
        let flat = aligned_head()?.with_scale(DEFAULT_SCALE / 4.0)?.probabilities(&query, &options)?;
        let base = aligned_head()?.probabilities(&query, &options)?;
        let sharp = aligned_head()?.with_scale(DEFAULT_SCALE * 4.0)?.probabilities(&query, &options)?;
        assert!(flat[0] < base[0] && base[0] < sharp[0], "{flat:?} {base:?} {sharp:?}");
        assert!(aligned_head()?.with_scale(0.0).is_err());
        assert!(aligned_head()?.with_scale(f64::NAN).is_err());
        Ok(())
    }

    #[test]
    fn rejects_wrong_shapes_and_lengths() -> Result<()> {
        let bad = Tensor::zeros((QK_DIM, 2), DType::F32, &Device::Cpu)?;
        let zeros = Tensor::zeros(QK_DIM, DType::F32, &Device::Cpu)?;
        assert!(JevHead::from_tensors(bad, zeros.clone(), zeros.clone().reshape((QK_DIM, 1))?, zeros.clone()).is_err());
        let head = aligned_head()?;
        assert!(head.probabilities(&[0f32; 10], &[0f32; HIDDEN]).is_err());
        assert!(head.probabilities(&[0f32; HIDDEN], &[0f32; HIDDEN + 1]).is_err());
        Ok(())
    }

    /// 실제 가중치 파일이 있을 때 모양을 확인합니다. `cargo test -- --ignored` 로 실행하세요.
    #[test]
    #[ignore]
    fn real_weights_have_the_expected_shapes() -> Result<()> {
        let head = JevHead::from_safetensors(Path::new("../models/zoro/jev_head.safetensors"))?;
        assert_eq!(head.wq_t.dims(), &[HIDDEN, QK_DIM]);
        assert_eq!(head.wk_t.dims(), &[HIDDEN, QK_DIM]);
        Ok(())
    }
}
