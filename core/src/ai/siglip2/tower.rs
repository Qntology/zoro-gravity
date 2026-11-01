//! 두 타워가 함께 쓰는 층: Pre-LayerNorm · 전체 주의 · GELU MLP. 모두 F32 활성값으로 계산합니다.

use anyhow::Result;
use candle_core::{DType, Tensor};

use super::config::{Tower, NORM_EPS};
use crate::ai::attention::{full_attention, layer_norm, ATTN_QUERY_CHUNK};
use crate::ai::residency::Loaded;

pub(super) struct Stack<'a> {
    pub w: &'a Loaded,
    /// 텐서 이름 앞부분 (`vision_model`, `text_model`)
    pub prefix: &'static str,
    pub cfg: &'a Tower,
}

impl Stack<'_> {
    pub fn tensor(&self, name: &str) -> Result<Tensor> {
        Ok(self.w.dense(&format!("{}.{name}", self.prefix))?.to_dtype(DType::F32)?)
    }

    pub fn linear(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        let y = self.w.linear(&format!("{}.{name}.weight", self.prefix), x)?;
        Ok(y.broadcast_add(&self.tensor(&format!("{name}.bias"))?)?)
    }

    pub fn norm(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        layer_norm(x, &self.tensor(&format!("{name}.weight"))?, &self.tensor(&format!("{name}.bias"))?, NORM_EPS)
    }

    pub fn mlp(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        let h = self.linear(&format!("{name}.fc1"), x)?.gelu()?;
        self.linear(&format!("{name}.fc2"), &h)
    }

    /// `[B, T, H·d] → [B·H, T, d]`
    pub fn split(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let (h, d) = (self.cfg.heads, self.cfg.hidden / self.cfg.heads);
        Ok(x.reshape((b, t, h, d))?.transpose(1, 2)?.contiguous()?.reshape((b * h, t, d))?)
    }

    /// `[B·H, T, d] → [B, T, H·d]`
    pub fn join(&self, x: &Tensor) -> Result<Tensor> {
        let (bh, t, d) = x.dims3()?;
        let h = self.cfg.heads;
        Ok(x.reshape((bh / h, h, t, d))?.transpose(1, 2)?.contiguous()?.reshape((bh / h, t, h * d))?)
    }

    pub fn scale(&self) -> f64 {
        1.0 / ((self.cfg.hidden / self.cfg.heads) as f64).sqrt()
    }

    fn attention(&self, b: &str, x: &Tensor) -> Result<Tensor> {
        let q = self.split(&self.linear(&format!("{b}.q_proj"), x)?)?;
        let k = self.split(&self.linear(&format!("{b}.k_proj"), x)?)?;
        let v = self.split(&self.linear(&format!("{b}.v_proj"), x)?)?;
        let o = full_attention(&q, &k, &v, self.scale(), ATTN_QUERY_CHUNK)?;
        self.linear(&format!("{b}.out_proj"), &self.join(&o)?)
    }

    fn layer(&self, i: usize, x: Tensor) -> Result<Tensor> {
        let b = format!("encoder.layers.{i}");
        let h = self.attention(&format!("{b}.self_attn"), &self.norm(&format!("{b}.layer_norm1"), &x)?)?;
        let x = (x + h)?;
        let h = self.mlp(&format!("{b}.mlp"), &self.norm(&format!("{b}.layer_norm2"), &x)?)?;
        Ok((x + h)?)
    }

    /// 인코더 전체. `x`: `[B, T, D]`
    pub fn encoder(&self, mut x: Tensor) -> Result<Tensor> {
        for i in 0..self.cfg.layers {
            x = self.layer(i, x)?;
        }
        Ok(x)
    }
}
