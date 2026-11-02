//! 주의 층에서 함께 쓰는 연산: RMSNorm · RoPE · 인과 주의 · KV 캐시. zoro 와 gocr 가 함께 씁니다. 모두 F32 입니다.
//!
//! 장치 전용 커널은 처음 쓰기 전에 CPU 기준과 비교해 맞을 때만 쓰고, 틀리거나 없으면 같은 식의 텐서 연산으로 계산합니다 (`ai::verify`).

use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor, D};
use candle_nn::ops::softmax_last_dim;

use crate::ai::kv_turbo::KvCodec;
use crate::ai::verify;

/// 질의를 이 크기로 잘라서 주의를 계산해 점수 행렬 `[헤드, 질의, 키]` 의 크기를 묶어 둡니다.
pub const ATTN_QUERY_CHUNK: usize = 512;
/// 마스크에 쓰는 매우 작은 값. softmax 에서 0 이 되고, 점수에 더해도 무한대가 되지 않는 값입니다.
pub(crate) const MASKED: f32 = -1e30;

// ---------------------------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------------------------

/// `x / sqrt(mean(x²) + eps) * w` (마지막 차원). `w` 는 이미 `1+w` 등으로 준비된 값입니다.
pub fn rms_norm(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let x = x.contiguous()?;
    if fused_rms_trusted(x.device()) {
        return Ok(candle_nn::ops::rms_norm(&x, w, eps)?);
    }
    rms_norm_composed(&x, w, eps)
}

fn rms_norm_composed(x: &Tensor, w: &Tensor, eps: f32) -> Result<Tensor> {
    let var = x.sqr()?.mean_keepdim(D::Minus1)?;
    let inv = (var + eps as f64)?.sqrt()?.recip()?;
    Ok(x.broadcast_mul(&inv)?.broadcast_mul(w)?)
}

fn fused_rms_trusted(device: &Device) -> bool {
    verify::trusted(device, "rms_norm", 1e-4, || {
        let x = Tensor::from_vec(verify::det_normal(4 * 256, 11), (4, 256), &Device::Cpu)?;
        let w = Tensor::from_vec(verify::det_normal(256, 12), 256, &Device::Cpu)?;
        let want = rms_norm_composed(&x, &w, 1e-6)?;
        let got = candle_nn::ops::rms_norm(&x.to_device(device)?, &w.to_device(device)?, 1e-6)?;
        verify::rel_err(&got.to_device(&Device::Cpu)?, &want)
    })
}

// ---------------------------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------------------------

/// `(x − 평균) / sqrt(분산 + eps) × w + b` (마지막 차원)
pub fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let c = x.broadcast_sub(&mean)?;
    let var = c.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(c.broadcast_div(&(var + eps)?.sqrt()?)?.broadcast_mul(w)?.broadcast_add(b)?)
}

// ---------------------------------------------------------------------------------------------
// RoPE (앞쪽 rope_dim 차원만 회전, 나머지는 그대로)
// ---------------------------------------------------------------------------------------------

/// 위치별 cos / sin 표. 텍스트만 다루므로 세 위치축(시간·높이·너비)이 같아서 일반 RoPE 와 같습니다.
pub struct Rope {
    cos: Tensor,
    sin: Tensor,
    rot: usize,
}

impl Rope {
    pub fn new(t: usize, rope_dim: usize, theta: f64, device: &Device) -> Result<Self> {
        Self::span(0, t, rope_dim, theta, device)
    }

    /// 위치 `start..start+t` 의 표. 이어서 읽을 때 씁니다.
    pub fn span(start: usize, t: usize, rope_dim: usize, theta: f64, device: &Device) -> Result<Self> {
        if rope_dim == 0 || !rope_dim.is_multiple_of(2) {
            bail!("회전 차원은 양의 짝수여야 합니다: {rope_dim}");
        }
        let half = rope_dim / 2;
        let mut cos = Vec::with_capacity(t * half);
        let mut sin = Vec::with_capacity(t * half);
        for pos in start..start + t {
            for i in 0..half {
                let inv_freq = theta.powf(-(2.0 * i as f64) / rope_dim as f64);
                let angle = pos as f64 * inv_freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (t, half), device)?,
            sin: Tensor::from_vec(sin, (t, half), device)?,
            rot: rope_dim,
        })
    }

    /// 이미 계산한 cos / sin 표 `[토큰, rot/2]` 로 만듭니다 (위치가 1차원이 아닌 비전 패치용).
    pub fn from_tables(cos: Tensor, sin: Tensor) -> Result<Self> {
        let (_, half) = cos.dims2()?;
        Ok(Self { cos, sin, rot: 2 * half })
    }

    /// `x`: `[헤드, 토큰, head_dim]`. 앞 `rot` 차원을 반으로 나눠 `(x1, x2) → (x1 cos − x2 sin, x2 cos + x1 sin)`.
    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, d) = x.dims3()?;
        if self.rot > d {
            bail!("회전 차원 {} 가 head_dim {d} 보다 큽니다", self.rot);
        }
        let half = self.rot / 2;
        let x1 = x.narrow(2, 0, half)?;
        let x2 = x.narrow(2, half, half)?;
        let (cos, sin) = (self.cos.unsqueeze(0)?, self.sin.unsqueeze(0)?);
        let o1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let o2 = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        if self.rot == d {
            Ok(Tensor::cat(&[&o1, &o2], 2)?)
        } else {
            let pass = x.narrow(2, self.rot, d - self.rot)?;
            Ok(Tensor::cat(&[&o1, &o2, &pass], 2)?)
        }
    }
}

// ---------------------------------------------------------------------------------------------
// 인과 주의 (GQA)
// ---------------------------------------------------------------------------------------------

/// 질의 구간별 인과 마스크. 구간 `[s, s+len)` 의 마스크는 `[len, s+len]` 이고 (i, j) 는 `j ≤ s+i` 일 때 0, 아니면 매우 작은 값입니다.
pub struct Causal {
    masks: Vec<Tensor>,
}

impl Causal {
    pub fn new(t: usize, chunk: usize, device: &Device) -> Result<Self> {
        if chunk == 0 {
            bail!("질의 구간 크기가 0 입니다");
        }
        let mut masks = Vec::new();
        let mut s = 0;
        while s < t {
            let len = chunk.min(t - s);
            let end = s + len;
            let mut m = vec![0f32; len * end];
            for i in 0..len {
                for j in (s + i + 1)..end {
                    m[i * end + j] = MASKED;
                }
            }
            masks.push(Tensor::from_vec(m, (len, end), device)?);
            s = end;
        }
        Ok(Self { masks })
    }
}

/// `[Hkv, T, D] → [Hkv·rep, T, D]` (질의 헤드 h 는 KV 헤드 h / rep 을 읽음)
fn repeat_kv(x: &Tensor, rep: usize) -> Result<Tensor> {
    if rep == 1 {
        return Ok(x.clone());
    }
    let (h, t, d) = x.dims3()?;
    Ok(x.unsqueeze(1)?.expand((h, rep, t, d))?.reshape((h * rep, t, d))?)
}

/// 마스크 없는 전체 주의 `softmax(q kᵀ · scale) v` (이미지 패치끼리). `q`·`k`·`v`: `[H, N, D]`. 질의를 `chunk` 개씩 나눠 계산합니다.
pub fn full_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, chunk: usize) -> Result<Tensor> {
    let (_, n, _) = q.dims3()?;
    if chunk == 0 {
        bail!("질의 구간 크기가 0 입니다");
    }
    let kt = k.transpose(1, 2)?.contiguous()?;
    let mut outs = Vec::with_capacity(n.div_ceil(chunk));
    let mut s = 0;
    while s < n {
        let len = chunk.min(n - s);
        let scores = (q.narrow(1, s, len)?.contiguous()?.matmul(&kt)? * scale)?;
        outs.push(softmax_last_dim(&scores)?.matmul(v)?);
        s += len;
    }
    Ok(if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 1)? })
}

/// `softmax(q kᵀ · scale + 마스크) v`. `q`: `[H, T, D]`, `k`·`v`: `[Hkv, T, D]`, 결과 `[H, T, D]`.
pub fn causal_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: &Causal) -> Result<Tensor> {
    let (h, t, _) = q.dims3()?;
    let hkv = k.dim(0)?;
    if hkv == 0 || !h.is_multiple_of(hkv) {
        bail!("질의 헤드 {h} 는 KV 헤드 {hkv} 의 배수여야 합니다");
    }
    let k = repeat_kv(k, h / hkv)?;
    let v = repeat_kv(v, h / hkv)?;
    let kt = k.transpose(1, 2)?.contiguous()?; // [H, D, T]
    let mut outs = Vec::with_capacity(causal.masks.len());
    let mut s = 0;
    for mask in &causal.masks {
        let len = mask.dim(0)?;
        let end = s + len;
        let qc = q.narrow(1, s, len)?.contiguous()?;
        let kc = kt.narrow(2, 0, end)?.contiguous()?;
        let vc = v.narrow(1, 0, end)?.contiguous()?;
        let scores = (qc.matmul(&kc)? * scale)?.broadcast_add(mask)?;
        outs.push(softmax_last_dim(&scores)?.matmul(&vc)?);
        s = end;
    }
    if s != t {
        bail!("마스크가 토큰 {t}개를 모두 덮지 못했습니다 ({s}개)");
    }
    Ok(if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 1)? })
}

// ---------------------------------------------------------------------------------------------
// KV 캐시 (이어서 읽기)
// ---------------------------------------------------------------------------------------------
//
// 프리필 구간(`base`)과 뒤에 붙는 구간(`tail`)으로 나눕니다. 토큰마다 전체 캐시를 복사하지 않으려는 것입니다.
// 키를 헤드 수만큼 반복하지 않고 질의를 KV 헤드별로 모아 곱해서(GQA) 메모리와 시간을 아낍니다.
// 코덱이 있으면 두 구간 모두 FP8 압축 행으로 보관합니다 (`ai::kv_turbo`).

/// `tail` 이 이만큼 쌓이면 `base` 로 합칩니다.
pub const TAIL_LIMIT: usize = 128;

/// 새 토큰끼리 서로를 어떻게 보는가
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnMode {
    /// 왼쪽 토큰만 봄 (자기회귀)
    Causal,
    /// 같은 묶음의 토큰을 모두 봄 (블록 디퓨전 초안)
    Block,
}

/// 새 토큰끼리 가리는 마스크. 층마다 값이 같아서 순전파마다 한 번만 만듭니다.
/// 장치로 올리는 복사는 앞선 계산이 끝나기를 기다리므로, 층마다 올리면 그때마다 GPU 가 멈춥니다.
pub struct StepMask {
    mode: AttnMode,
    /// 질의 구간마다 `[rep × 구간, 보이는 키]`. 토큰이 하나뿐인 구간은 가릴 것이 없습니다.
    chunks: Vec<Option<Tensor>>,
}

impl StepMask {
    /// 새 토큰 `n` 개. `rep` 는 KV 헤드 하나를 읽는 질의 헤드 수입니다.
    pub fn new(mode: AttnMode, n: usize, rep: usize, device: &Device) -> Result<Self> {
        let mut chunks = Vec::new();
        let mut s = 0;
        while mode == AttnMode::Causal && s < n {
            let len = ATTN_QUERY_CHUNK.min(n - s);
            let visible = s + len;
            chunks.push(if visible > 1 {
                let mut m = vec![0f32; rep * len * visible];
                for r in 0..rep {
                    for i in 0..len {
                        let row = (r * len + i) * visible;
                        for j in (s + i + 1)..visible {
                            m[row + j] = MASKED;
                        }
                    }
                }
                Some(Tensor::from_vec(m, (rep * len, visible), device)?)
            } else {
                None
            });
            s = visible;
        }
        Ok(Self { mode, chunks })
    }
}

#[derive(Default)]
pub struct LayerCache {
    /// `[Hkv, D, T]` (전치해 둔 키). 코덱이 있으면 `[Hkv, T, 워드]` 압축 행입니다.
    base_kt: Option<Tensor>,
    /// `[Hkv, T, D]`. 코덱이 있으면 `[Hkv, T, 워드]` 압축 행입니다.
    base_v: Option<Tensor>,
    tail_kt: Option<Tensor>,
    tail_v: Option<Tensor>,
    codec: Option<Arc<KvCodec>>,
}

impl LayerCache {
    pub fn new(codec: Option<Arc<KvCodec>>) -> Self {
        Self { codec, ..Self::default() }
    }

    pub fn push(&mut self, kt: &Tensor, v: &Tensor) -> Result<()> {
        // 실측 감시가 압축을 끈 층은 아직 비어 있으므로 원본으로 이어 갑니다.
        if self.codec.as_ref().is_some_and(|c| c.broken()) && self.segments().is_empty() {
            self.codec = None;
        }
        let (kt, v, key_axis) = match &self.codec {
            Some(c) => (c.pack_k(&kt.transpose(1, 2)?.contiguous()?)?, c.pack_v(v)?, 1),
            None => (kt.clone(), v.clone(), 2),
        };
        self.tail_kt = Some(match self.tail_kt.take() {
            Some(t) => Tensor::cat(&[&t, &kt], key_axis)?,
            None => kt,
        });
        self.tail_v = Some(match self.tail_v.take() {
            Some(t) => Tensor::cat(&[&t, &v], 1)?,
            None => v,
        });
        let tail_len = self.tail_v.as_ref().map_or(Ok(0), |t| t.dim(1))?;
        if tail_len >= TAIL_LIMIT || self.base_v.is_none() {
            let (tk, tv) = (self.tail_kt.take(), self.tail_v.take());
            if let (Some(tk), Some(tv)) = (tk, tv) {
                self.base_kt = Some(match self.base_kt.take() {
                    Some(b) => Tensor::cat(&[&b, &tk], key_axis)?,
                    None => tk,
                });
                self.base_v = Some(match self.base_v.take() {
                    Some(b) => Tensor::cat(&[&b, &tv], 1)?,
                    None => tv,
                });
            }
        }
        Ok(())
    }

    pub fn segments(&self) -> Vec<(&Tensor, &Tensor)> {
        let mut out = Vec::with_capacity(2);
        if let (Some(k), Some(v)) = (&self.base_kt, &self.base_v) {
            out.push((k, v));
        }
        if let (Some(k), Some(v)) = (&self.tail_kt, &self.tail_v) {
            out.push((k, v));
        }
        out
    }

    /// 캐시가 차지하는 바이트 수
    pub fn stored_bytes(&self) -> usize {
        let bytes = |t: &Tensor| t.elem_count() * t.dtype().size_in_bytes();
        self.segments().iter().map(|(k, v)| bytes(k) + bytes(v)).sum()
    }

    /// 새 토큰의 질의 `q` `[H, n, D]` 로 캐시와 새 키 · 값을 읽어 `[H, n, D]` 를 돌려줍니다.
    ///
    /// 압축 캐시는 logis-center `turbo_kv_step` 과 같이 나눕니다. 이력이 없으면 원본으로, 새 토큰이 하나면 압축 행 위에서 바로,
    /// 여러 토큰이면 이력만 풀어서 새 토큰의 원본과 함께 계산합니다.
    pub fn attend(&self, q: &Tensor, kt: &Tensor, v: &Tensor, mask: &StepMask, scale: f64) -> Result<Tensor> {
        let past = self.segments();
        let Some(codec) = &self.codec else {
            return attend(q, kt, v, &past, mask, scale);
        };
        if past.is_empty() {
            let out = attend(q, kt, v, &[], mask, scale)?;
            codec.watch(q, kt, v, &out, scale)?;
            return Ok(out);
        }
        if q.dim(1)? == 1 && mask.mode == AttnMode::Causal && codec.fused() {
            let (k_new, v_new) = (codec.pack_k(&kt.transpose(1, 2)?.contiguous()?)?, codec.pack_v(v)?);
            let mut rows = past;
            rows.push((&k_new, &v_new));
            return codec.attend_rows(q, &rows, scale);
        }
        let plain = past
            .iter()
            .map(|(k, v)| Ok((codec.unpack_k(k)?.transpose(1, 2)?.contiguous()?, codec.unpack_v(v)?)))
            .collect::<Result<Vec<_>>>()?;
        let refs: Vec<(&Tensor, &Tensor)> = plain.iter().map(|(k, v)| (k, v)).collect();
        attend(q, kt, v, &refs, mask, scale)
    }
}

/// 새 토큰의 질의 `q` `[H, n, D]` 로 캐시와 새 키 · 값을 읽어 `[H, n, D]` 를 돌려줍니다.
pub fn attend(q: &Tensor, kt: &Tensor, v: &Tensor, past: &[(&Tensor, &Tensor)], mask: &StepMask, scale: f64) -> Result<Tensor> {
    let (h, n, d) = q.dims3()?;
    let hkv = kt.dim(0)?;
    let rep = h / hkv;
    let mut outs = Vec::with_capacity(n.div_ceil(ATTN_QUERY_CHUNK));
    let mut s = 0;
    while s < n {
        let len = ATTN_QUERY_CHUNK.min(n - s);
        let visible = match mask.mode {
            AttnMode::Causal => s + len,
            AttnMode::Block => n,
        };
        // 같은 KV 헤드를 읽는 질의 헤드를 한 줄로 모읍니다: [Hkv, rep × len, D]
        let qc = q.narrow(1, s, len)?.contiguous()?.reshape((hkv, rep * len, d))?;
        let mut parts = Vec::with_capacity(past.len() + 1);
        for (pk, _) in past {
            parts.push(qc.matmul(pk)?);
        }
        let kc = if visible == n { kt.clone() } else { kt.narrow(2, 0, visible)?.contiguous()? };
        let mut new_scores = qc.matmul(&kc)?;
        if let Some(m) = mask.chunks.get(outs.len()).and_then(|m| m.as_ref()) {
            new_scores = new_scores.broadcast_add(m)?;
        }
        parts.push(new_scores);
        let scores = if parts.len() == 1 { parts.remove(0) } else { Tensor::cat(&parts, 2)? };
        let p = softmax_last_dim(&(scores * scale)?)?;
        // 값을 구간별로 곱해 더합니다 (캐시를 합치지 않습니다).
        let mut off = 0;
        let mut acc: Option<Tensor> = None;
        for (_, pv) in past {
            let l = pv.dim(1)?;
            let o = p.narrow(2, off, l)?.contiguous()?.matmul(pv)?;
            acc = Some(match acc {
                Some(a) => (a + o)?,
                None => o,
            });
            off += l;
        }
        let vc = if visible == n { v.clone() } else { v.narrow(1, 0, visible)?.contiguous()? };
        let o = p.narrow(2, off, visible)?.contiguous()?.matmul(&vc)?;
        let acc = match acc {
            Some(a) => (a + o)?,
            None => o,
        };
        outs.push(acc.reshape((h, len, d))?);
        s += len;
    }
    Ok(if outs.len() == 1 { outs.remove(0) } else { Tensor::cat(&outs, 1)? })
}

/// 로짓에서 가장 큰 토큰과 그 확률
pub fn top1(logits: &[f32]) -> (u32, f32) {
    let (mut best, mut max) = (0usize, f32::MIN);
    for (i, &l) in logits.iter().enumerate() {
        if l > max {
            (best, max) = (i, l);
        }
    }
    let z: f32 = logits.iter().map(|l| (l - max).exp()).sum();
    (best as u32, 1.0 / z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::accel;
    use crate::ai::verify::{det_normal, rel_err};
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use crate::ai::weights::WeightFile;
    use crate::ai::zoro::reference::naive_attention;
    use candle_core::quantized::gguf_file::Value;
    use candle_core::quantized::{GgmlDType, QMatMul};
    use candle_core::Module;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor {
        Tensor::from_vec(data, shape.to_vec(), &Device::Cpu).unwrap()
    }

    fn flat(x: &Tensor) -> Vec<f32> {
        x.flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    /// 주의 층이 쓰는 연산을 하나씩 GPU 에서 돌려 CPU 와 비교합니다. 모델 시험이 실패할 때 어느 연산이 틀렸는지 `[DEVICE-OP]` 줄이 알려 줍니다.
    #[test]
    #[ignore = "GPU 가 필요합니다"]
    fn device_basic_ops_match_the_cpu() -> Result<()> {
        let device = accel::test_gpu();
        let (h, n, d) = (2usize, 6usize, 8usize);
        let q = t(det_normal(h * n * d, 31), &[h, n, d]);
        let kt = t(det_normal(h * d * n, 32), &[h, d, n]);
        let a = t(det_normal(n * d, 33), &[n, d]);
        let w = t(det_normal(d * n, 34), &[d, n]);
        let on = |x: &Tensor| x.to_device(&device);
        let mut wrong = Vec::new();
        let mut within = |name: &str, tol: f32, got: Tensor, want: Tensor| -> Result<()> {
            let err = rel_err(&got.to_device(&Device::Cpu)?, &want)?;
            println!("[DEVICE-OP] {name}: {err:.1e}");
            if err > tol {
                wrong.push(format!("{name} {err:.1e}"));
            }
            Ok(())
        };
        let mut compare = |name: &str, got: Tensor, want: Tensor| within(name, 1e-3, got, want);
        compare("복사", on(&q)?, q.clone())?;
        compare("스칼라 곱", (on(&q)? * 2.0)?, (&q * 2.0)?)?;
        compare("전치", on(&q)?.transpose(1, 2)?.contiguous()?, q.transpose(1, 2)?.contiguous()?)?;
        compare("자르기", on(&q)?.narrow(1, 2, 3)?.contiguous()?, q.narrow(1, 2, 3)?.contiguous()?)?;
        compare("이어 붙이기", Tensor::cat(&[&on(&q)?, &on(&q)?], 2)?, Tensor::cat(&[&q, &q], 2)?)?;
        compare("평균", on(&q)?.mean_keepdim(D::Minus1)?, q.mean_keepdim(D::Minus1)?)?;
        compare("제곱근", on(&q)?.sqr()?.sqrt()?, q.sqr()?.sqrt()?)?;
        compare("행렬곱 2차원", on(&a)?.matmul(&on(&w)?)?, a.matmul(&w)?)?;
        compare("행렬곱 3차원", on(&q)?.matmul(&on(&kt)?)?, q.matmul(&kt)?)?;
        compare("softmax", softmax_last_dim(&on(&q)?)?, softmax_last_dim(&q)?)?;
        compare("silu", candle_nn::ops::silu(&on(&q)?)?, candle_nn::ops::silu(&q)?)?;
        let norm = t(det_normal(d, 35), &[d]);
        compare("rms_norm", candle_nn::ops::rms_norm(&on(&a)?, &on(&norm)?, 1e-6)?, rms_norm_composed(&a, &norm, 1e-6)?)?;

        // 기준은 가중치만 푼 정확한 곱입니다. 기본 곱은 장치에 따라 활성값도 8비트로 줄여서 그만큼의 오차를 허용합니다.
        let dir = TempDir::new("attn-gpu");
        let path = dir.0.join("q8.gguf");
        write_gguf(&path, &[("general.architecture", Value::String("zoro".into()))], &[("w".into(), t(det_normal(16 * 64, 36), &[16, 64]), GgmlDType::Q8_0)])?;
        let file = WeightFile::open(&path)?;
        let (on_gpu, on_cpu) = (QMatMul::from_qtensor(file.read_qtensor("w", &device)?)?, QMatMul::from_qtensor(file.read_qtensor("w", &Device::Cpu)?)?);
        let x = t(det_normal(n * 64, 37), &[n, 64]);
        let row = x.narrow(0, 0, 1)?.contiguous()?;
        compare("Q8_0 정확한 곱", on_gpu.forward_exact(&on(&x)?)?, on_cpu.forward_exact(&x)?)?;
        within("Q8_0 곱 여러 행", 1e-2, on_gpu.forward(&on(&x)?)?, on_cpu.forward_exact(&x)?)?;
        within("Q8_0 곱 한 행", 1e-2, on_gpu.forward(&on(&row)?)?, on_cpu.forward_exact(&row)?)?;

        assert!(wrong.is_empty(), "틀린 연산: {wrong:?}");
        println!("[DEVICE-CHECK] {} 기본 연산: 모두 허용 오차 안", accel::backend_name(&device));
        Ok(())
    }

    #[test]
    fn rms_norm_matches_the_formula_and_both_paths_agree() -> Result<()> {
        let x = t(det_normal(3 * 8, 1), &[3, 8]);
        let w = t(det_normal(8, 2), &[8]);
        let got = flat(&rms_norm(&x, &w, 1e-6)?);
        let xs = flat(&x);
        let ws = flat(&w);
        for r in 0..3 {
            let row = &xs[r * 8..(r + 1) * 8];
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f32>() / 8.0 + 1e-6).sqrt();
            for c in 0..8 {
                assert!((got[r * 8 + c] - row[c] * inv * ws[c]).abs() < 1e-5);
            }
        }
        let composed = rms_norm_composed(&x, &w, 1e-6)?;
        assert!(rel_err(&rms_norm(&x, &w, 1e-6)?, &composed)? < 1e-6);
        Ok(())
    }

    #[test]
    fn rope_rotates_the_front_and_keeps_the_tail() -> Result<()> {
        let (h, tt, d, rot) = (2usize, 5usize, 12usize, 8usize);
        let x = t(det_normal(h * tt * d, 3), &[h, tt, d]);
        let got = flat(&Rope::new(tt, rot, 10000.0, &Device::Cpu)?.apply(&x)?);
        let xs = flat(&x);
        for head in 0..h {
            for pos in 0..tt {
                let base = (head * tt + pos) * d;
                for i in 0..rot / 2 {
                    let ang = pos as f64 * 10000f64.powf(-(2.0 * i as f64) / rot as f64);
                    let (c, s) = (ang.cos() as f32, ang.sin() as f32);
                    let (x1, x2) = (xs[base + i], xs[base + rot / 2 + i]);
                    assert!((got[base + i] - (x1 * c - x2 * s)).abs() < 1e-5, "앞 절반 {head}/{pos}/{i}");
                    assert!((got[base + rot / 2 + i] - (x2 * c + x1 * s)).abs() < 1e-5, "뒤 절반 {head}/{pos}/{i}");
                }
                for i in rot..d {
                    assert_eq!(got[base + i], xs[base + i], "회전하지 않는 꼬리");
                }
            }
        }
        // 위치 0 은 항등
        for i in 0..d {
            assert!((got[i] - xs[i]).abs() < 1e-6);
        }
        assert!(Rope::new(4, 7, 1e4, &Device::Cpu).is_err());
        assert!(Rope::new(4, 16, 1e4, &Device::Cpu)?.apply(&x).is_err(), "회전 차원이 head_dim 보다 큼");
        Ok(())
    }

    #[test]
    fn attention_matches_the_naive_loops_for_every_chunking() -> Result<()> {
        let (h, hkv, tt, d) = (4usize, 2usize, 7usize, 6usize);
        let q = t(det_normal(h * tt * d, 4), &[h, tt, d]);
        let k = t(det_normal(hkv * tt * d, 5), &[hkv, tt, d]);
        let v = t(det_normal(hkv * tt * d, 6), &[hkv, tt, d]);
        let scale = 1.0 / (d as f32).sqrt();
        let want = naive_attention(&flat(&q), &flat(&k), &flat(&v), h, hkv, tt, d, scale);
        for chunk in [1usize, 3, 7, 512] {
            let causal = Causal::new(tt, chunk, &Device::Cpu)?;
            let got = causal_attention(&q, &k, &v, scale as f64, &causal)?;
            assert_eq!(got.dims(), &[h, tt, d]);
            let err = rel_err(&got, &t(want.clone(), &[h, tt, d]))?;
            assert!(err < 1e-5, "구간 {chunk}: {err}");
        }
        let bad_kv = t(det_normal(3 * tt * d, 7), &[3, tt, d]);
        assert!(causal_attention(&q, &bad_kv, &v, 1.0, &Causal::new(tt, 4, &Device::Cpu)?).is_err());
        assert!(Causal::new(4, 0, &Device::Cpu).is_err());
        Ok(())
    }

    #[test]
    fn a_token_never_sees_the_future() -> Result<()> {
        // 마지막 토큰의 K·V 를 바꿔도 앞쪽 토큰의 출력은 같아야 합니다.
        let (h, tt, d) = (2usize, 6usize, 4usize);
        let q = t(det_normal(h * tt * d, 8), &[h, tt, d]);
        let k = t(det_normal(h * tt * d, 9), &[h, tt, d]);
        let v = t(det_normal(h * tt * d, 10), &[h, tt, d]);
        let mut k2 = flat(&k);
        let mut v2 = flat(&v);
        for head in 0..h {
            for x in 0..d {
                k2[(head * tt + tt - 1) * d + x] += 5.0;
                v2[(head * tt + tt - 1) * d + x] -= 7.0;
            }
        }
        let causal = Causal::new(tt, 4, &Device::Cpu)?;
        let a = causal_attention(&q, &k, &v, 0.5, &causal)?;
        let b = causal_attention(&q, &t(k2, &[h, tt, d]), &t(v2, &[h, tt, d]), 0.5, &causal)?;
        let diff = (a.narrow(1, 0, tt - 1)? - b.narrow(1, 0, tt - 1)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0);
        let last = (a.narrow(1, tt - 1, 1)? - b.narrow(1, tt - 1, 1)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(last > 1e-3, "마지막 토큰은 자기 자신의 K·V 에 반응해야 합니다");
        Ok(())
    }

    /// 앞 `past` 토큰을 캐시에 넣고 뒤 `n` 토큰을 읽습니다. 코덱이 없으면 원본 기준값입니다.
    fn read_after(codec: Option<Arc<KvCodec>>, past: usize, n: usize, mode: AttnMode) -> Result<(Tensor, usize)> {
        let (h, hkv, d, tt) = (8usize, 4usize, 16usize, past + n);
        let q = t(det_normal(h * tt * d, 21), &[h, tt, d]);
        let k = t(det_normal(hkv * tt * d, 22), &[hkv, tt, d]);
        let v = t(det_normal(hkv * tt * d, 23), &[hkv, tt, d]);
        let kt = k.transpose(1, 2)?.contiguous()?;
        let cut = |x: &Tensor, dim: usize, s: usize, l: usize| x.narrow(dim, s, l).unwrap().contiguous().unwrap();
        let mut lc = LayerCache::new(codec);
        lc.push(&cut(&kt, 2, 0, past / 2), &cut(&v, 1, 0, past / 2))?;
        lc.push(&cut(&kt, 2, past / 2, past - past / 2), &cut(&v, 1, past / 2, past - past / 2))?;
        let mask = StepMask::new(mode, n, h / hkv, &Device::Cpu)?;
        let out = lc.attend(&cut(&q, 1, past, n), &cut(&kt, 2, past, n), &cut(&v, 1, past, n), &mask, 0.25)?;
        Ok((out, lc.stored_bytes()))
    }

    #[test]
    fn a_compressed_cache_stays_close_to_the_exact_one_and_is_smaller() -> Result<()> {
        // 새 토큰 하나 (압축 행 위에서 바로), 여러 토큰 (이력만 풀어서), 블록 초안을 모두 지납니다.
        for (n, mode) in [(1usize, AttnMode::Causal), (3, AttnMode::Causal), (3, AttnMode::Block)] {
            let (exact, plain_bytes) = read_after(None, 30, n, mode)?;
            let codec = Arc::new(KvCodec::new(16, &Device::Cpu)?);
            let (got, packed_bytes) = read_after(Some(codec), 30, n, mode)?;
            assert_eq!(got.dims(), exact.dims());
            let err = rel_err(&got, &exact)?;
            assert!(err < 0.1, "{n}토큰 {mode:?}: 원본 대비 상대오차 {err}");
            // 벡터 하나가 16 + 8 바이트입니다 (F32 는 64 바이트).
            assert_eq!(packed_bytes * 64, plain_bytes * 24);
        }
        Ok(())
    }

    #[test]
    fn a_compressed_cache_keeps_merging_past_the_tail_limit() -> Result<()> {
        let codec = Arc::new(KvCodec::new(16, &Device::Cpu)?);
        let (exact, _) = read_after(None, TAIL_LIMIT * 2 + 7, 1, AttnMode::Causal)?;
        let (got, _) = read_after(Some(codec), TAIL_LIMIT * 2 + 7, 1, AttnMode::Causal)?;
        assert!(rel_err(&got, &exact)? < 0.1);
        Ok(())
    }
}
