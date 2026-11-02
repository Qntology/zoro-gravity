//! gocr 비전 인코더: 패치 → 언어 디코더 차원의 이미지 토큰 `[병합 토큰 수, 1536]`.
//!
//! 패치 임베딩 → 블록 24개 (RMSNorm · 전체 주의 · SwiGLU) → 정규화 → 2×2 다운샘플 → 병합기.
//! 가중치는 `model.visual.*` 입니다. 합성곱 두 개(패치 임베딩 · 다운샘플)는 입력이 겹치지 않아서 선형 변환과 같으므로,
//! 올릴 때 펼쳐 두고 행렬곱으로 계산합니다 (`prepare`).
//! 위치 정보는 위치 임베딩 표 없이 2차원 RoPE 로만 줍니다.
//! 격자가 같은 이미지는 한 번에 계산합니다. 주의만 이미지 안에서 일어나므로 나머지 층은 행을 이어 붙여도 같고, 주의는 묶음을 헤드처럼 다룹니다.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::silu;

use super::config::GocrConfig;
use super::image::PreparedImage;
use crate::ai::attention::{full_attention, layer_norm, rms_norm, Rope, ATTN_QUERY_CHUNK};
use crate::ai::residency::{Loaded, Prepare};

const P: &str = "model.visual";

/// 올릴 때 한 번 하는 변환: 합성곱 가중치를 선형 가중치 모양으로 펼칩니다.
///
/// - 패치 임베딩 `[D, 3×2, 14, 14]` (GGUF 가 시간축과 채널을 합친 모양): 이미지는 같은 프레임 두 장이므로 시간축을 더해 `[D, 3×14×14]`
/// - 다운샘플 `[1536, 1024, 2, 2]`: `[1536, 1024×4]`
pub const PREPARE: Prepare = Prepare { touches: reshapes, apply: prepare };

fn reshapes(name: &str) -> bool {
    matches!(name, "model.visual.patch_embed.proj.weight" | "model.visual.downsample.weight")
}

fn prepare(name: &str, t: Tensor) -> candle_core::Result<Tensor> {
    if name == "model.visual.patch_embed.proj.weight" {
        let (d, ct, h, w) = t.dims4()?;
        let dtype = t.dtype();
        let summed = t.to_dtype(DType::F32)?.reshape((d, 3, ct / 3, h, w))?.sum(2)?;
        summed.reshape((d, 3 * h * w))?.to_dtype(dtype)
    } else if name == "model.visual.downsample.weight" {
        let (o, c, h, w) = t.dims4()?;
        t.reshape((o, c * h * w))
    } else {
        Ok(t)
    }
}

/// 가중치가 BF16 이어도(ROCm) 활성값은 F32 로 다룹니다. 행렬곱만 가중치의 dtype 으로 합니다.
fn linear(w: &Loaded, name: &str, x: &Tensor) -> Result<Tensor> {
    let weight = w.dense(&format!("{P}.{name}.weight"))?;
    Ok(x.to_dtype(weight.dtype())?.matmul(&weight.t()?)?.to_dtype(DType::F32)?)
}

fn linear_bias(w: &Loaded, name: &str, x: &Tensor) -> Result<Tensor> {
    let bias = w.dense(&format!("{P}.{name}.bias"))?.to_dtype(DType::F32)?;
    Ok(linear(w, name, x)?.broadcast_add(&bias)?)
}

fn norm(w: &Loaded, name: &str, x: &Tensor, eps: f32) -> Result<Tensor> {
    let weight = w.dense(&format!("{P}.{name}.weight"))?.to_dtype(DType::F32)?;
    rms_norm(x, &weight, eps)
}

/// 패치의 2차원 RoPE 표 (cos, sin `[패치, head_dim/2]`). 앞 절반 주파수는 행 위치, 뒤 절반은 열 위치입니다.
fn vision_rope(cfg: &GocrConfig, grid_h: usize, grid_w: usize, device: &Device) -> Result<Rope> {
    let half = cfg.v_head_dim() / 2; // 축 하나당 half/2 주파수
    let inv: Vec<f64> = (0..half / 2).map(|i| 10000f64.powf(-(2.0 * i as f64) / half as f64)).collect();
    let n = grid_h * grid_w;
    let m = cfg.v_merge;
    let (mut cos, mut sin) = (Vec::with_capacity(n * half), Vec::with_capacity(n * half));
    for mr in 0..grid_h / m {
        for mc in 0..grid_w / m {
            for i in 0..m {
                for j in 0..m {
                    for pos in [mr * m + i, mc * m + j] {
                        for f in &inv {
                            let a = pos as f64 * f;
                            cos.push(a.cos() as f32);
                            sin.push(a.sin() as f32);
                        }
                    }
                }
            }
        }
    }
    Rope::from_tables(Tensor::from_vec(cos, (n, half), device)?, Tensor::from_vec(sin, (n, half), device)?)
}

/// `h`: `[B·N, C]`. 이미지가 섞이지 않도록 `[B·H, N, D]` 로 주의를 계산합니다.
fn attention(cfg: &GocrConfig, w: &Loaded, b: &str, h: &Tensor, rope: &Rope, batch: usize) -> Result<Tensor> {
    let (total, _) = h.dims2()?;
    let n = total / batch;
    let (nh, d) = (cfg.v_heads, cfg.v_head_dim());
    let qkv = linear_bias(w, &format!("{b}.attn.qkv"), h)?
        .reshape((batch, n, 3, nh, d))?
        .permute((2, 0, 3, 1, 4))?
        .contiguous()?
        .reshape((3, batch * nh, n, d))?;
    let (q, k, v) = (qkv.narrow(0, 0, 1)?.squeeze(0)?, qkv.narrow(0, 1, 1)?.squeeze(0)?, qkv.narrow(0, 2, 1)?.squeeze(0)?);
    let q = rope.apply(&norm(w, &format!("{b}.attn.q_norm"), &q, cfg.v_rms_eps)?)?;
    let k = rope.apply(&norm(w, &format!("{b}.attn.k_norm"), &k, cfg.v_rms_eps)?)?;
    let o = full_attention(&q, &k, &v.contiguous()?, 1.0 / (d as f64).sqrt(), ATTN_QUERY_CHUNK)?;
    let o = o.reshape((batch, nh, n, d))?.transpose(1, 2)?.contiguous()?.reshape((total, nh * d))?;
    linear_bias(w, &format!("{b}.attn.proj"), &o)
}

fn block(cfg: &GocrConfig, w: &Loaded, i: usize, x: &Tensor, rope: &Rope, batch: usize) -> Result<Tensor> {
    let b = format!("blocks.{i}");
    let h = norm(w, &format!("{b}.norm1"), x, cfg.v_rms_eps)?;
    let x = (x + attention(cfg, w, &b, &h, rope, batch)?)?;
    let h = norm(w, &format!("{b}.norm2"), &x, cfg.v_rms_eps)?;
    let gate = linear_bias(w, &format!("{b}.mlp.gate_proj"), &h)?;
    let up = linear_bias(w, &format!("{b}.mlp.up_proj"), &h)?;
    let down = linear_bias(w, &format!("{b}.mlp.down_proj"), &(silu(&gate)? * up)?)?;
    Ok((x + down)?)
}

fn merger_norm(w: &Loaded, x: &Tensor) -> Result<Tensor> {
    let weight = w.dense(&format!("{P}.merger.post_projection_norm.weight"))?.to_dtype(DType::F32)?;
    let bias = w.dense(&format!("{P}.merger.post_projection_norm.bias"))?.to_dtype(DType::F32)?;
    layer_norm(x, &weight, &bias, 1e-5)
}

/// 이미지 한 장을 이미지 토큰으로 바꿉니다. 결과는 `[병합 토큰 수, 1536]` F32 이고 `w` 의 장치에 있습니다.
pub fn encode(cfg: &GocrConfig, w: &Loaded, img: &PreparedImage, device: &Device) -> Result<Tensor> {
    encode_batch(cfg, w, &[img], device)
}

/// 격자가 같은 이미지들을 한 번에 이미지 토큰으로 바꿉니다. 결과는 이미지 순서대로 이어 붙인 `[B × 병합 토큰 수, 1536]` 입니다.
pub fn encode_batch(cfg: &GocrConfig, w: &Loaded, imgs: &[&PreparedImage], device: &Device) -> Result<Tensor> {
    let first = imgs.first().context("이미지가 없습니다")?;
    if imgs.iter().any(|i| (i.grid_h, i.grid_w) != (first.grid_h, first.grid_w)) {
        bail!("묶음 안의 격자가 다릅니다");
    }
    let batch = imgs.len();
    let n = batch * first.n_patches();
    let patches: Vec<f32> = imgs.iter().flat_map(|i| i.patches.iter().copied()).collect();
    let x = Tensor::from_vec(patches, (n, cfg.patch_len()), device)?;
    let mut x = linear_bias(w, "patch_embed.proj", &x)?;
    let rope = vision_rope(cfg, first.grid_h, first.grid_w, device)?;
    for i in 0..cfg.v_depth {
        x = block(cfg, w, i, &x, &rope, batch)?;
    }
    let x = norm(w, "post_layernorm", &x, cfg.v_rms_eps)?;
    // 2×2 패치 → 병합 토큰 하나: [M, 4, C] → [M, C, 4] 로 펴서 펼친 다운샘플 가중치와 맞춥니다.
    let mm = cfg.v_merge * cfg.v_merge;
    let m = n / mm;
    let x = x.reshape((m, mm, cfg.v_hidden))?.transpose(1, 2)?.contiguous()?.reshape((m, mm * cfg.v_hidden))?;
    let x = linear_bias(w, "downsample", &x)?;
    // 병합기
    let h = linear(w, "merger.proj", &x)?;
    let h = merger_norm(w, &h)?.gelu_erf()?;
    let gate = linear(w, "merger.gate_proj", &h)?;
    let up = linear(w, "merger.up_proj", &h)?;
    linear(w, "merger.down_proj", &(silu(&gate)? * up)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::accel;
    use crate::ai::gocr::{GocrInput, GocrVision, VisionFeatures};
    use crate::ai::ocr::{PhaseCtx, VisionEncoder};
    use crate::ai::residency::{Budget, Group, ModelId, ModelSpec, Phase, Residency};
    use crate::ai::verify::{det_normal, rel_err};
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use crate::ai::weights::WeightFile;
    use candle_core::quantized::gguf_file::Value;
    use candle_core::quantized::GgmlDType;
    use std::sync::Arc;

    fn cfg() -> GocrConfig {
        GocrConfig {
            n_layers: 1,
            hidden: 16,
            ffn: 32,
            vocab: 100,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            rms_eps: 1e-5,
            rope_theta: 10000.0,
            rope_sections: [2, 1, 1],
            v_depth: 1,
            v_hidden: 32,
            v_ffn: 64,
            v_heads: 2,
            v_patch: 2,
            v_merge: 2,
            v_out: 16,
            v_rms_eps: 1e-5,
            image_token: 90,
            image_start: 91,
            image_end: 92,
            eos: vec![99],
        }
    }

    /// `cfg()` 크기의 작은 비전 가중치 파일
    fn tiny_spec(dir: &TempDir) -> ModelSpec {
        let (d, out, ffn, m, head) = (32usize, 16usize, 64usize, 24usize, 16usize);
        let mut list: Vec<(String, Vec<usize>)> = vec![
            ("patch_embed.proj.weight".into(), vec![d, 6, 2, 2]),
            ("patch_embed.proj.bias".into(), vec![d]),
            ("post_layernorm.weight".into(), vec![d]),
            ("downsample.weight".into(), vec![out, d, 2, 2]),
            ("downsample.bias".into(), vec![out]),
            ("merger.proj.weight".into(), vec![out, out]),
            ("merger.post_projection_norm.weight".into(), vec![out]),
            ("merger.post_projection_norm.bias".into(), vec![out]),
            ("merger.gate_proj.weight".into(), vec![m, out]),
            ("merger.up_proj.weight".into(), vec![m, out]),
            ("merger.down_proj.weight".into(), vec![out, m]),
        ];
        let b = "blocks.0";
        for (name, shape) in [
            ("norm1.weight", vec![d]),
            ("norm2.weight", vec![d]),
            ("attn.qkv.weight", vec![3 * d, d]),
            ("attn.qkv.bias", vec![3 * d]),
            ("attn.q_norm.weight", vec![head]),
            ("attn.k_norm.weight", vec![head]),
            ("attn.proj.weight", vec![d, d]),
            ("attn.proj.bias", vec![d]),
            ("mlp.gate_proj.weight", vec![ffn, d]),
            ("mlp.gate_proj.bias", vec![ffn]),
            ("mlp.up_proj.weight", vec![ffn, d]),
            ("mlp.up_proj.bias", vec![ffn]),
            ("mlp.down_proj.weight", vec![d, ffn]),
            ("mlp.down_proj.bias", vec![d]),
        ] {
            list.push((format!("{b}.{name}"), shape));
        }
        let tensors: Vec<(String, Tensor, GgmlDType)> = list
            .into_iter()
            .enumerate()
            .map(|(i, (name, shape))| {
                let n = shape.iter().product();
                let v: Vec<f32> = det_normal(n, 200 + i as u64).into_iter().map(|x| x * 0.3).collect();
                (format!("model.visual.{name}"), Tensor::from_vec(v, shape, &Device::Cpu).unwrap(), GgmlDType::F32)
            })
            .collect();
        let path = dir.0.join("gocr-vision.gguf");
        write_gguf(&path, &[("general.architecture", Value::String("gocr".into())), ("gocr.block_count", Value::U32(0))], &tensors).unwrap();
        ModelSpec::new(ModelId::Gocr, Arc::new(WeightFile::open(&path).unwrap())).with_prepare(PREPARE)
    }

    fn image(seed: u64, grid_h: usize, grid_w: usize) -> PreparedImage {
        PreparedImage { patches: det_normal(grid_h * grid_w * 12, seed), grid_h, grid_w }
    }

    fn close(a: &Tensor, b: &Tensor) -> Result<()> {
        let (a, b) = (a.flatten_all()?.to_vec1::<f32>()?, b.flatten_all()?.to_vec1::<f32>()?);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!((x - y).abs() < 1e-4, "[{i}] {x} != {y}");
        }
        Ok(())
    }

    #[test]
    fn images_read_together_match_images_read_one_by_one() -> Result<()> {
        let dir = TempDir::new("gocr-batch");
        let spec = tiny_spec(&dir);
        let c = cfg();
        let shared = Residency::shared(Device::Cpu, Budget::unlimited());
        let mut res = shared.lock().unwrap();
        let h = res.enter(&spec, &Phase::new("비전", [Group::Vision]))?;
        let images: Vec<PreparedImage> = (0..3).map(|s| image(40 + s, 4, 6)).collect();
        let refs: Vec<&PreparedImage> = images.iter().collect();
        let together = encode_batch(&c, &h[0], &refs, &Device::Cpu)?;
        assert_eq!(together.dims(), &[18, 16], "이미지마다 병합 토큰 6개");
        for (i, img) in images.iter().enumerate() {
            close(&together.narrow(0, i * 6, 6)?, &encode(&c, &h[0], img, &Device::Cpu)?)?;
        }
        let a = encode(&c, &h[0], &images[0], &Device::Cpu)?;
        let b = encode(&c, &h[0], &images[1], &Device::Cpu)?;
        assert!((a - b)?.abs()?.max_all()?.to_scalar::<f32>()? > 1e-3, "서로 다른 이미지는 결과가 달라야 합니다");
        assert!(encode_batch(&c, &h[0], &[&images[0], &image(9, 4, 4)], &Device::Cpu).is_err(), "격자가 다르면 묶지 않습니다");
        assert!(encode_batch(&c, &h[0], &[], &Device::Cpu).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "GPU 가 필요합니다"]
    fn device_vision_matches_the_cpu() -> Result<()> {
        let device = accel::test_gpu();
        let dir = TempDir::new("gocr-gpu");
        let spec = tiny_spec(&dir);
        let c = cfg();
        let images: Vec<PreparedImage> = (0..3).map(|s| image(40 + s, 4, 6)).collect();
        let refs: Vec<&PreparedImage> = images.iter().collect();
        let on = |d: &Device| -> Result<Tensor> {
            let mut res = Residency::new(d.clone(), Budget::unlimited());
            let h = res.enter(&spec, &Phase::new("비전", [Group::Vision]))?;
            Ok(encode_batch(&c, &h[0], &refs, d)?.to_device(&Device::Cpu)?)
        };
        let err = rel_err(&on(&device)?, &on(&Device::Cpu)?)?;
        assert!(err < 1e-3, "{err}");
        println!("[DEVICE-CHECK] {} gocr 비전: CPU 대비 상대오차 {err:.1e}", accel::backend_name(&device));
        Ok(())
    }

    #[test]
    fn grouping_by_grid_keeps_the_input_order() -> Result<()> {
        let dir = TempDir::new("gocr-order");
        let spec = tiny_spec(&dir);
        let shared = Residency::shared(Device::Cpu, Budget::unlimited());
        let mut res = shared.lock().unwrap();
        let h = res.enter(&spec, &Phase::new("비전", [Group::Vision]))?;
        let grids = [(4, 6), (4, 4), (4, 6), (2, 2), (4, 4)];
        let inputs: Vec<GocrInput> = grids.iter().enumerate().map(|(i, &(gh, gw))| GocrInput { image: image(60 + i as u64, gh, gw), prompt: format!("{i}") }).collect();
        let ctx = PhaseCtx { residency: &res };
        let batched = GocrVision { cfg: Arc::new(cfg()), batching: true }.encode_many(&ctx, &h[0], &inputs)?;
        let single = GocrVision { cfg: Arc::new(cfg()), batching: false }.encode_many(&ctx, &h[0], &inputs)?;
        assert_eq!(batched.len(), inputs.len());
        for (i, (a, b)) in batched.iter().zip(&single).enumerate() {
            let VisionFeatures { tokens, rows, cols, prompt, .. } = a;
            assert_eq!((*rows, *cols), (grids[i].0 / 2, grids[i].1 / 2));
            assert_eq!(prompt, &format!("{i}"));
            close(tokens, &b.tokens)?;
        }
        Ok(())
    }

    #[test]
    fn conv_weights_are_flattened_like_a_linear_layer() -> Result<()> {
        // 시간축 합: 채널 c 의 두 프레임 가중치를 더한 값이어야 합니다.
        let (d, h, w) = (2usize, 2usize, 2usize);
        let raw = det_normal(d * 6 * h * w, 3);
        let t = Tensor::from_vec(raw.clone(), (d, 6, h, w), &Device::Cpu)?;
        let out = prepare("model.visual.patch_embed.proj.weight", t)?;
        assert_eq!(out.dims(), &[d, 3 * h * w]);
        let v = out.flatten_all()?.to_vec1::<f32>()?;
        for dd in 0..d {
            for c in 0..3 {
                for p in 0..h * w {
                    let a = raw[(dd * 6 + c * 2) * h * w + p];
                    let b = raw[(dd * 6 + c * 2 + 1) * h * w + p];
                    assert!((v[(dd * 3 + c) * h * w + p] - (a + b)).abs() < 1e-5);
                }
            }
        }
        let t = Tensor::zeros((4, 3, 2, 2), DType::F32, &Device::Cpu)?;
        assert_eq!(prepare("model.visual.downsample.weight", t)?.dims(), &[4, 12]);
        let t = Tensor::zeros((4, 3), DType::F32, &Device::Cpu)?;
        assert_eq!(prepare("model.visual.blocks.0.attn.proj.weight", t)?.dims(), &[4, 3], "다른 가중치는 그대로");
        Ok(())
    }

    #[test]
    fn rope_tables_use_row_then_column_positions() -> Result<()> {
        let c = cfg();
        // 패치 4×2, 병합 2: 병합 행 2 × 병합 열 1
        let rope = vision_rope(&c, 4, 2, &Device::Cpu)?;
        let x = Tensor::ones((1, 8, c.v_head_dim() * 4), DType::F32, &Device::Cpu);
        assert!(x.is_ok());
        // v_head_dim 16: half 8, 축당 주파수 4. 첫 패치(0,0)는 모든 각이 0 이라 항등
        let id = Tensor::from_vec(det_normal(8 * 16, 5), (1, 8, 16), &Device::Cpu)?;
        let out = rope.apply(&id)?;
        let (a, b) = (out.flatten_all()?.to_vec1::<f32>()?, id.flatten_all()?.to_vec1::<f32>()?);
        for i in 0..16 {
            assert!((a[i] - b[i]).abs() < 1e-6, "위치 (0,0) 은 회전하지 않습니다");
        }
        // 두 번째 패치 (0,1): 행 주파수(앞 4개)는 각 0, 열 주파수(뒤 4개)는 각이 있음
        assert!((a[16] - b[16]).abs() < 1e-6, "행 축 주파수는 행 0 에서 회전하지 않습니다");
        assert!(a[16..32].iter().zip(&b[16..32]).any(|(p, q)| (p - q).abs() > 1e-3), "열 축은 회전합니다");
        Ok(())
    }
}
