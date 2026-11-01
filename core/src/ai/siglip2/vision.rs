//! siglip2 비전 타워: 패치 → 층 27개 → 패치별 공유 공간 특징 + 이미지 전체 풀링 벡터.
//!
//! 풀링 헤드의 어텐션 출력은 `out_proj(Σ αᵢ · v(xᵢ))` 이므로, 패치 i 의 몫은 `out_proj(v(xᵢ))` 입니다.
//! 여기에 헤드의 잔차 블록을 똑같이 적용하면 텍스트 헤드 출력과 같은 좌표계의 패치 특징이 됩니다 (logis-center 의 방식).
//! 이 값과 앵커 문구의 코사인이 히트맵의 원본입니다.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};

use super::config::{Siglip2Config, PATCH_EMBED, VISION_POS};
use super::image::{interpolate_grid, PatchInput};
use super::tower::Stack;
use crate::ai::attention::{full_attention, ATTN_QUERY_CHUNK};
use crate::ai::residency::{dense_linear, Loaded};

const HEAD: &str = "head.attention";

/// 비전 단계의 결과 (`w` 의 장치). `patches`: `[B, N, D]`, `pooled`: `[B, D]`. 둘 다 L2 정규화 전입니다.
pub struct VisionOutput {
    pub patches: Tensor,
    pub pooled: Tensor,
}

fn stack<'a>(cfg: &'a Siglip2Config, w: &'a Loaded) -> Stack<'a> {
    Stack { w, prefix: "vision_model", cfg: &cfg.vision }
}

/// 패치 임베딩 + 격자에 맞춰 다시 뽑은 위치 임베딩 `[B, N, D]`
fn embed(cfg: &Siglip2Config, w: &Loaded, imgs: &[&PatchInput], device: &Device) -> Result<Tensor> {
    let d = cfg.vision.hidden;
    let img = imgs.first().context("이미지가 없습니다")?;
    if imgs.iter().any(|i| (i.rows, i.cols) != (img.rows, img.cols)) {
        bail!("묶음 안의 격자가 다릅니다");
    }
    let n = img.n_patches();
    let pixels: Vec<f32> = imgs.iter().flat_map(|i| i.pixels.iter().copied()).collect();
    let px = Tensor::from_vec(pixels, (imgs.len(), n, cfg.patch_len()), device)?;
    let y = w.linear(PATCH_EMBED, &px)?;
    let bias = w.dense("vision_model.embeddings.patch_embedding.bias")?.to_dtype(DType::F32)?;
    let table = w.dense(VISION_POS)?.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let pos = interpolate_grid(&table, cfg.pos_side, d, img.rows, img.cols);
    let pos = Tensor::from_vec(pos, (1, n, d), device)?;
    Ok(y.broadcast_add(&bias)?.broadcast_add(&pos)?)
}

/// 풀링 헤드의 입력 투영. `in_proj_weight` 는 q · k · v 가중치를 세로로 이어 붙인 `[3D, D]` 입니다.
fn head_proj(s: &Stack<'_>, slot: usize, x: &Tensor) -> Result<Tensor> {
    let d = s.cfg.hidden;
    let weight = s.w.dense(&format!("{}.{HEAD}.in_proj_weight", s.prefix))?.narrow(0, slot * d, d)?.contiguous()?;
    let bias = s.tensor(&format!("{HEAD}.in_proj_bias"))?.narrow(0, slot * d, d)?.contiguous()?;
    Ok(dense_linear(x, &weight)?.broadcast_add(&bias)?)
}

/// 헤드의 잔차 블록 `h + mlp(norm(h))`
fn head_residual(s: &Stack<'_>, h: &Tensor) -> Result<Tensor> {
    Ok((h + s.mlp("head.mlp", &s.norm("head.layernorm", h)?)?)?)
}

fn pool(s: &Stack<'_>, hidden: &Tensor) -> Result<Tensor> {
    let (batch, _, d) = hidden.dims3()?;
    let probe = s.tensor("head.probe")?.broadcast_as((batch, 1, d))?.contiguous()?;
    let q = s.split(&head_proj(s, 0, &probe)?)?;
    let k = s.split(&head_proj(s, 1, hidden)?)?;
    let v = s.split(&head_proj(s, 2, hidden)?)?;
    let o = full_attention(&q, &k, &v, s.scale(), ATTN_QUERY_CHUNK)?;
    let o = s.linear(&format!("{HEAD}.out_proj"), &s.join(&o)?)?;
    Ok(head_residual(s, &o)?.squeeze(1)?)
}

fn project_patches(s: &Stack<'_>, hidden: &Tensor) -> Result<Tensor> {
    let o = s.linear(&format!("{HEAD}.out_proj"), &head_proj(s, 2, hidden)?)?;
    head_residual(s, &o)
}

/// 격자가 같은 이미지들을 한 번에 읽습니다.
pub fn encode(cfg: &Siglip2Config, w: &Loaded, imgs: &[&PatchInput], device: &Device) -> Result<VisionOutput> {
    let s = stack(cfg, w);
    let x = s.encoder(embed(cfg, w, imgs, device)?)?;
    let hidden = s.norm("post_layernorm", &x)?;
    Ok(VisionOutput { patches: project_patches(&s, &hidden)?, pooled: pool(&s, &hidden)? })
}
