//! gocr 언어 디코더: 층 16개 (샌드위치 정규화 · GQA 주의 · SwiGLU) + KV 캐시.
//!
//! 층: `x + 정규화(주의(정규화(x)))` 다음 `x + 정규화(MLP(정규화(x)))`.
//! 위치는 (시간, 높이, 너비) 3축이고, head_dim/2 개의 주파수를 축별로 나눠 씁니다 (`mrope_section`). 이미지 토큰은 격자 좌표를, 글자 토큰은 세 축 모두 같은 값을 갖습니다.
//! 회전은 인접한 두 차원을 한 쌍으로 돌립니다 (zoro 의 앞뒤 절반 방식과 다릅니다).
//!
//! KV 캐시와 캐시를 읽는 주의는 `ai::attention` 에 있습니다 (zoro 도 함께 씁니다).

use std::sync::Arc;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::ops::silu;

use super::config::GocrConfig;
pub use crate::ai::attention::{top1, AttnMode};
use crate::ai::attention::{rms_norm, LayerCache, StepMask};
use crate::ai::kv_turbo::KvCodec;
use crate::ai::residency::Loaded;

/// 토큰의 (시간, 높이, 너비) 위치
pub type Pos = [u32; 3];

pub struct KvCache {
    layers: Vec<LayerCache>,
    len: usize,
}

/// 새 토큰들의 층별 (전치한 키, 값). 캐시에 넣을지, 앞 몇 개만 넣을지는 호출자가 정합니다 (초안은 버리고, 검증은 일치한 앞부분만 넣습니다).
pub struct PendingKv {
    layers: Vec<(Tensor, Tensor)>,
    n: usize,
}

impl KvCache {
    pub fn new(n_layers: usize) -> Self {
        Self::with_codec(n_layers, None)
    }

    /// 코덱이 있으면 키 · 값을 FP8 로 압축해 보관합니다.
    pub fn with_codec(n_layers: usize, codec: Option<Arc<KvCodec>>) -> Self {
        Self { layers: (0..n_layers).map(|_| LayerCache::new(codec.clone())).collect(), len: 0 }
    }

    /// 캐시에 들어간 토큰 수
    pub fn len(&self) -> usize {
        self.len
    }

    /// 새 토큰 앞 `keep` 개의 키 · 값을 캐시에 넣습니다.
    pub fn commit(&mut self, pending: &PendingKv, keep: usize) -> Result<()> {
        if keep > pending.n {
            return Err(anyhow!("넣을 토큰 {keep}개가 계산한 토큰 {}개보다 많습니다", pending.n));
        }
        if keep == 0 {
            return Ok(());
        }
        for (lc, (kt, v)) in self.layers.iter_mut().zip(&pending.layers) {
            if keep == pending.n {
                lc.push(kt, v)?;
            } else {
                lc.push(&kt.narrow(2, 0, keep)?.contiguous()?, &v.narrow(1, 0, keep)?.contiguous()?)?;
            }
        }
        self.len += keep;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// 위치 → cos, sin `[토큰, head_dim/2]`
pub fn rope_tables(cfg: &GocrConfig, pos: &[Pos], device: &Device) -> Result<(Tensor, Tensor)> {
    let half = cfg.head_dim / 2;
    let inv: Vec<f64> = (0..half).map(|i| cfg.rope_theta.powf(-(2.0 * i as f64) / cfg.head_dim as f64)).collect();
    let mut axis = Vec::with_capacity(half);
    for (a, &k) in cfg.rope_sections.iter().enumerate() {
        axis.extend(std::iter::repeat_n(a, k));
    }
    let n = pos.len();
    let (mut cos, mut sin) = (Vec::with_capacity(n * half), Vec::with_capacity(n * half));
    for p in pos {
        for i in 0..half {
            let angle = p[axis[i]] as f64 * inv[i];
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((Tensor::from_vec(cos, (n, half), device)?, Tensor::from_vec(sin, (n, half), device)?))
}

/// 인접한 두 차원을 한 쌍으로 회전: `(x₀, x₁) → (x₀ cos − x₁ sin, x₁ cos + x₀ sin)`. `x`: `[H, n, D]`
fn rotate_pairs(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (h, n, d) = x.dims3()?;
    let pairs = x.reshape((h, n, d / 2, 2))?;
    let a = pairs.narrow(3, 0, 1)?.squeeze(3)?;
    let b = pairs.narrow(3, 1, 1)?.squeeze(3)?;
    let (c, s) = (cos.unsqueeze(0)?, sin.unsqueeze(0)?);
    let ra = (a.broadcast_mul(&c)? - b.broadcast_mul(&s)?)?;
    let rb = (b.broadcast_mul(&c)? + a.broadcast_mul(&s)?)?;
    Ok(Tensor::stack(&[ra, rb], 3)?.reshape((h, n, d))?)
}

/// 언어 단계에서 받은 가중치로 돌아가는 디코더
pub struct Decoder<'a> {
    pub cfg: &'a GocrConfig,
    pub blocks: &'a Loaded,
    pub output: &'a Loaded,
}

impl Decoder<'_> {
    /// 새 토큰 임베딩 `x` `[n, hidden]` 를 층 전체에 통과시켜 은닉 상태 `[n, hidden]` (최종 정규화 전)와 새 토큰의 키 · 값을 돌려줍니다.
    /// 캐시는 바뀌지 않습니다. 넣으려면 [`KvCache::commit`] 을 부릅니다.
    pub fn forward(&self, cache: &KvCache, x: &Tensor, pos: &[Pos], mode: AttnMode) -> Result<(Tensor, PendingKv)> {
        let (cos, sin) = rope_tables(self.cfg, pos, x.device())?;
        let n = x.dim(0)?;
        let mask = StepMask::new(mode, n, self.cfg.n_heads / self.cfg.n_kv_heads, x.device())?;
        let mut h = x.clone();
        let mut layers = Vec::with_capacity(self.cfg.n_layers);
        for i in 0..self.cfg.n_layers {
            let (next, kv) = self.layer(&cache.layers[i], i, &h, &cos, &sin, &mask)?;
            h = next;
            layers.push(kv);
        }
        Ok((h, PendingKv { layers, n }))
    }

    fn layer(&self, lc: &LayerCache, i: usize, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: &StepMask) -> Result<(Tensor, (Tensor, Tensor))> {
        let w = self.blocks;
        let p = |s: &str| format!("blk.{i}.{s}");
        let eps = self.cfg.rms_eps;
        let h = rms_norm(x, w.dense(&p("attn_norm.weight"))?, eps)?;
        let (a, kv) = self.attention(lc, i, &h, cos, sin, mask)?;
        let a = rms_norm(&a, w.dense(&p("post_attn_norm.weight"))?, eps)?;
        let x = (x + a)?;
        let h = rms_norm(&x, w.dense(&p("ffn_norm.weight"))?, eps)?;
        let gu = w.linear(&p("ffn_gate_up.weight"), &h)?;
        let ffn = self.cfg.ffn;
        let gate = gu.narrow(1, 0, ffn)?;
        let up = gu.narrow(1, ffn, ffn)?;
        let m = w.linear(&p("ffn_down.weight"), &(silu(&gate)? * up)?)?;
        let m = rms_norm(&m, w.dense(&p("post_ffn_norm.weight"))?, eps)?;
        Ok(((x + m)?, kv))
    }

    fn attention(&self, lc: &LayerCache, i: usize, h: &Tensor, cos: &Tensor, sin: &Tensor, mask: &StepMask) -> Result<(Tensor, (Tensor, Tensor))> {
        let w = self.blocks;
        let c = self.cfg;
        let p = |s: &str| format!("blk.{i}.{s}");
        let n = h.dim(0)?;
        let (nh, nkv, d) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let q = w.linear(&p("attn_q.weight"), h)?.reshape((n, nh, d))?.transpose(0, 1)?.contiguous()?;
        let k = w.linear(&p("attn_k.weight"), h)?.reshape((n, nkv, d))?.transpose(0, 1)?.contiguous()?;
        let v = w.linear(&p("attn_v.weight"), h)?.reshape((n, nkv, d))?.transpose(0, 1)?.contiguous()?;
        let q = rotate_pairs(&q, cos, sin)?;
        let k = rotate_pairs(&k, cos, sin)?;
        let kt = k.transpose(1, 2)?.contiguous()?;
        let o = lc.attend(&q, &kt, &v, mask, 1.0 / (d as f64).sqrt())?;
        let o = o.transpose(0, 1)?.contiguous()?.reshape((n, nh * d))?;
        Ok((w.linear(&p("attn_output.weight"), &o)?, (kt, v)))
    }

    /// 은닉 상태 `[n, hidden]` → 어휘 로짓 `[n, vocab]` (CPU)
    pub fn logits(&self, h: &Tensor) -> Result<Vec<Vec<f32>>> {
        let n = rms_norm(h, self.output.dense("output_norm.weight")?, self.cfg.rms_eps)?;
        let l = self.output.linear("output.weight", &n)?;
        Ok(l.to_device(&Device::Cpu)?.to_vec2::<f32>()?)
    }

    pub fn last_logits(&self, h: &Tensor) -> Result<Vec<f32>> {
        let n = h.dim(0)?;
        self.logits(&h.narrow(0, n - 1, 1)?)?.into_iter().next().ok_or_else(|| anyhow!("빈 로짓"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::accel;
    use crate::ai::attention::attend;
    use crate::ai::residency::{Budget, Group, ModelId, ModelSpec, Phase, Residency};
    use crate::ai::verify::{det_normal, rel_err};
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use crate::ai::weights::WeightFile;
    use candle_core::quantized::gguf_file::Value;
    use candle_core::quantized::GgmlDType;
    use std::sync::Arc;

    #[test]
    fn pair_rotation_matches_the_definition() -> Result<()> {
        let (h, n, d) = (2usize, 3usize, 8usize);
        let x = Tensor::from_vec(det_normal(h * n * d, 1), (h, n, d), &Device::Cpu)?;
        let c = Tensor::from_vec(det_normal(n * d / 2, 2), (n, d / 2), &Device::Cpu)?;
        let s = Tensor::from_vec(det_normal(n * d / 2, 3), (n, d / 2), &Device::Cpu)?;
        let out = rotate_pairs(&x, &c, &s)?.flatten_all()?.to_vec1::<f32>()?;
        let xs = x.flatten_all()?.to_vec1::<f32>()?;
        let (cs, ss) = (c.flatten_all()?.to_vec1::<f32>()?, s.flatten_all()?.to_vec1::<f32>()?);
        for head in 0..h {
            for t in 0..n {
                for i in 0..d / 2 {
                    let b = (head * n + t) * d + 2 * i;
                    let (a0, a1) = (xs[b], xs[b + 1]);
                    let (ci, si) = (cs[t * d / 2 + i], ss[t * d / 2 + i]);
                    assert!((out[b] - (a0 * ci - a1 * si)).abs() < 1e-5);
                    assert!((out[b + 1] - (a1 * ci + a0 * si)).abs() < 1e-5);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn top1_gives_the_argmax_and_its_probability() {
        let (i, p) = top1(&[0.0, 3.0, 1.0]);
        assert_eq!(i, 1);
        let z = 1.0 + (-3.0f32).exp() + (-2.0f32).exp();
        assert!((p - 1.0 / z).abs() < 1e-6);
    }

    #[test]
    fn attention_in_one_shot_equals_attention_with_a_cache() -> Result<()> {
        // 토큰 6개를 한 번에 계산한 결과와, 앞 4개를 캐시에 넣고 뒤 2개를 계산한 결과가 같아야 합니다.
        let (h, hkv, d) = (4usize, 2usize, 8usize);
        let t = 6;
        let dev = Device::Cpu;
        let q = Tensor::from_vec(det_normal(h * t * d, 7), (h, t, d), &dev)?;
        let k = Tensor::from_vec(det_normal(hkv * t * d, 8), (hkv, t, d), &dev)?;
        let v = Tensor::from_vec(det_normal(hkv * t * d, 9), (hkv, t, d), &dev)?;
        let kt = k.transpose(1, 2)?.contiguous()?;
        let whole = attend(&q, &kt, &v, &[], &StepMask::new(AttnMode::Causal, t, h / hkv, &dev)?, 0.5)?;
        let mut lc = LayerCache::default();
        lc.push(&kt.narrow(2, 0, 3)?.contiguous()?, &v.narrow(1, 0, 3)?.contiguous()?)?;
        lc.push(&kt.narrow(2, 3, 1)?.contiguous()?, &v.narrow(1, 3, 1)?.contiguous()?)?;
        let tail = attend(
            &q.narrow(1, 4, 2)?.contiguous()?,
            &kt.narrow(2, 4, 2)?.contiguous()?,
            &v.narrow(1, 4, 2)?.contiguous()?,
            &lc.segments(),
            &StepMask::new(AttnMode::Causal, 2, h / hkv, &dev)?,
            0.5,
        )?;
        let err = rel_err(&tail, &whole.narrow(1, 4, 2)?)?;
        assert!(err < 1e-5, "{err}");
        Ok(())
    }

    /// 층 2개짜리 작은 디코더 (가중치는 F32)
    struct Mini {
        cfg: GocrConfig,
        blocks: std::sync::Arc<Loaded>,
        output: std::sync::Arc<Loaded>,
        device: Device,
        _dir: TempDir,
    }

    fn mini() -> Mini {
        mini_on(&Device::Cpu)
    }

    fn mini_on(device: &Device) -> Mini {
        let cfg = GocrConfig {
            n_layers: 2,
            hidden: 16,
            ffn: 32,
            vocab: 50,
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
            image_token: 40,
            image_start: 41,
            image_end: 42,
            eos: vec![49],
        };
        let (d, ffn, hq, hk) = (cfg.hidden, cfg.ffn, cfg.n_heads * cfg.head_dim, cfg.n_kv_heads * cfg.head_dim);
        let mut t: Vec<(String, Tensor, GgmlDType)> = Vec::new();
        let mut seed = 40u64;
        let mut add = |name: String, shape: Vec<usize>, gain: f32| {
            seed += 1;
            let n: usize = shape.iter().product();
            let v: Vec<f32> = det_normal(n, seed).into_iter().map(|x| x * gain).collect();
            t.push((name, Tensor::from_vec(v, shape, &Device::Cpu).unwrap(), GgmlDType::F32));
        };
        for i in 0..cfg.n_layers {
            for n in ["attn_norm", "post_attn_norm", "ffn_norm", "post_ffn_norm"] {
                add(format!("blk.{i}.{n}.weight"), vec![d], 0.5);
            }
            add(format!("blk.{i}.attn_q.weight"), vec![hq, d], 0.2);
            add(format!("blk.{i}.attn_k.weight"), vec![hk, d], 0.2);
            add(format!("blk.{i}.attn_v.weight"), vec![hk, d], 0.2);
            add(format!("blk.{i}.attn_output.weight"), vec![d, hq], 0.2);
            add(format!("blk.{i}.ffn_gate_up.weight"), vec![2 * ffn, d], 0.2);
            add(format!("blk.{i}.ffn_down.weight"), vec![d, ffn], 0.2);
        }
        add("output_norm.weight".into(), vec![d], 0.5);
        add("output.weight".into(), vec![cfg.vocab, d], 0.2);
        let dir = TempDir::new("gocr-text");
        let path = dir.0.join("mini.gguf");
        write_gguf(&path, &[("general.architecture", Value::String("gocr".into()))], &t).unwrap();
        let file = Arc::new(WeightFile::open(&path).unwrap());
        let spec = ModelSpec::new(ModelId::Gocr, file);
        let mut res = Residency::new(device.clone(), Budget::unlimited());
        let mut h = res.enter(&spec, &Phase::new("언어", [Group::blocks(0..cfg.n_layers), Group::Output])).unwrap();
        let output = h.pop().unwrap();
        let blocks = h.pop().unwrap();
        Mini { cfg, blocks, output, device: device.clone(), _dir: dir }
    }

    impl Mini {
        fn dec(&self) -> Decoder<'_> {
            Decoder { cfg: &self.cfg, blocks: &self.blocks, output: &self.output }
        }

        fn embed(&self, n: usize, seed: u64) -> Tensor {
            Tensor::from_vec(det_normal(n * self.cfg.hidden, seed), (n, self.cfg.hidden), &self.device).unwrap()
        }
    }

    fn positions(from: usize, n: usize) -> Vec<Pos> {
        (from..from + n).map(|i| [i as u32; 3]).collect()
    }

    /// 프리필 뒤에 토큰을 조금씩 넣은 결과가 한 번에 읽은 결과와 같아야 합니다. `TAIL_LIMIT` 를 넘겨 합치는 경로까지 지납니다.
    #[test]
    fn reading_in_pieces_equals_reading_everything_at_once() -> Result<()> {
        let m = mini();
        let dec = m.dec();
        let total = crate::ai::attention::TAIL_LIMIT + 30;
        let x = m.embed(total, 5);
        let (whole, _) = dec.forward(&KvCache::new(m.cfg.n_layers), &x, &positions(0, total), AttnMode::Causal)?;

        let mut cache = KvCache::new(m.cfg.n_layers);
        let mut at = 0;
        let mut got = Vec::new();
        for n in std::iter::once(9).chain(std::iter::repeat_n(7, total / 7)) {
            let n = n.min(total - at);
            if n == 0 {
                break;
            }
            let (h, kv) = dec.forward(&cache, &x.narrow(0, at, n)?.contiguous()?, &positions(at, n), AttnMode::Causal)?;
            cache.commit(&kv, n)?;
            got.push(h);
            at += n;
        }
        assert_eq!(cache.len(), total);
        let refs: Vec<&Tensor> = got.iter().collect();
        let err = rel_err(&Tensor::cat(&refs, 0)?, &whole)?;
        assert!(err < 1e-4, "{err}");
        Ok(())
    }

    /// 프리필 뒤에 한 토큰씩 읽은 로짓이 GPU 와 CPU 에서 같아야 합니다.
    #[test]
    #[ignore = "GPU 가 필요합니다"]
    fn device_decoder_matches_the_cpu() -> Result<()> {
        let read = |m: &Mini| -> Result<Tensor> {
            let dec = m.dec();
            let x = m.embed(12, 5);
            let mut cache = KvCache::new(m.cfg.n_layers);
            let (_, kv) = dec.forward(&cache, &x.narrow(0, 0, 9)?.contiguous()?, &positions(0, 9), AttnMode::Causal)?;
            cache.commit(&kv, 9)?;
            let mut h = None;
            for at in 9..12 {
                let (y, kv) = dec.forward(&cache, &x.narrow(0, at, 1)?.contiguous()?, &positions(at, 1), AttnMode::Causal)?;
                cache.commit(&kv, 1)?;
                h = Some(y);
            }
            let logits = dec.last_logits(&h.expect("마지막 은닉 상태"))?;
            Ok(Tensor::from_vec(logits, m.cfg.vocab, &Device::Cpu)?)
        };
        let device = accel::test_gpu();
        let err = rel_err(&read(&mini_on(&device))?, &read(&mini())?)?;
        assert!(err < 1e-3, "{err}");
        println!("[DEVICE-CHECK] {} gocr 언어: CPU 대비 상대오차 {err:.1e}", accel::backend_name(&device));
        Ok(())
    }

    /// FP8 로 압축한 캐시로 읽어도 원본 캐시와 가깝고, 앞부분만 넣어도 문맥이 어긋나지 않습니다.
    #[test]
    fn a_compressed_cache_reads_close_to_the_exact_one() -> Result<()> {
        let m = mini();
        let dec = m.dec();
        let total = 40usize;
        let x = m.embed(total, 31);
        let read = |codec: Option<Arc<KvCodec>>| -> Result<Tensor> {
            let mut cache = KvCache::with_codec(m.cfg.n_layers, codec);
            let (_, kv) = dec.forward(&cache, &x.narrow(0, 0, 30)?.contiguous()?, &positions(0, 30), AttnMode::Causal)?;
            cache.commit(&kv, 30)?;
            // 세 토큰을 읽고 하나만 넣은 뒤, 한 토큰씩 읽습니다.
            let (_, kv) = dec.forward(&cache, &x.narrow(0, 30, 3)?.contiguous()?, &positions(30, 3), AttnMode::Causal)?;
            cache.commit(&kv, 1)?;
            let mut h = None;
            for at in 31..total {
                let (y, kv) = dec.forward(&cache, &x.narrow(0, at, 1)?.contiguous()?, &positions(at, 1), AttnMode::Causal)?;
                cache.commit(&kv, 1)?;
                h = Some(y);
            }
            assert_eq!(cache.len(), total);
            Ok(h.expect("마지막 은닉 상태"))
        };
        let exact = read(None)?;
        let got = read(Some(Arc::new(KvCodec::new(m.cfg.head_dim, &Device::Cpu)?)))?;
        let err = rel_err(&got, &exact)?;
        assert!(err < 0.1, "{err}");
        Ok(())
    }

    /// 검증에서 앞부분만 채택하면, 다음 계산은 채택한 토큰만 문맥으로 봐야 합니다.
    #[test]
    fn a_partial_commit_keeps_only_the_accepted_tokens() -> Result<()> {
        let m = mini();
        let dec = m.dec();
        let (head, tail) = (6usize, 4usize);
        let x = m.embed(head + tail + 1, 6);
        let mut cache = KvCache::new(m.cfg.n_layers);
        let (_, kv) = dec.forward(&cache, &x.narrow(0, 0, head)?.contiguous()?, &positions(0, head), AttnMode::Causal)?;
        cache.commit(&kv, head)?;
        let (_, kv) = dec.forward(&cache, &x.narrow(0, head, tail)?.contiguous()?, &positions(head, tail), AttnMode::Causal)?;
        cache.commit(&kv, 2)?;
        assert_eq!(cache.len(), head + 2);
        // 채택한 두 토큰 뒤에 새 토큰을 잇습니다.
        let next = x.narrow(0, head + tail, 1)?.contiguous()?;
        let (h, _) = dec.forward(&cache, &next, &positions(head + 2, 1), AttnMode::Causal)?;

        let along = Tensor::cat(&[x.narrow(0, 0, head + 2)?, next.clone()], 0)?.contiguous()?;
        let (whole, _) = dec.forward(&KvCache::new(m.cfg.n_layers), &along, &positions(0, head + 3), AttnMode::Causal)?;
        let err = rel_err(&h, &whole.narrow(0, head + 2, 1)?)?;
        assert!(err < 1e-4, "{err}");
        assert!(cache.commit(&kv_of(&dec, &cache, &next)?, 2).is_err(), "계산한 것보다 많이 넣을 수 없습니다");
        Ok(())
    }

    fn kv_of(dec: &Decoder<'_>, cache: &KvCache, x: &Tensor) -> Result<PendingKv> {
        Ok(dec.forward(cache, x, &positions(cache.len(), 1), AttnMode::Causal)?.1)
    }

    #[test]
    fn a_draft_does_not_touch_the_cache() -> Result<()> {
        let m = mini();
        let dec = m.dec();
        let x = m.embed(5, 8);
        let mut cache = KvCache::new(m.cfg.n_layers);
        let (_, kv) = dec.forward(&cache, &x, &positions(0, 5), AttnMode::Causal)?;
        cache.commit(&kv, 5)?;
        let draft = m.embed(3, 9);
        let (a, _) = dec.forward(&cache, &draft, &positions(5, 3), AttnMode::Block)?;
        let (b, _) = dec.forward(&cache, &draft, &positions(5, 3), AttnMode::Block)?;
        assert_eq!(cache.len(), 5);
        assert!(rel_err(&a, &b)? < 1e-7, "커밋하지 않으면 다음 계산에 영향이 없습니다");
        let (c, _) = dec.forward(&cache, &draft, &positions(5, 3), AttnMode::Causal)?;
        assert!(rel_err(&a, &c)? > 1e-4, "블록과 인과 주의는 결과가 다릅니다");
        assert_eq!(dec.logits(&a)?.len(), 3);
        assert_eq!(dec.logits(&a)?[0].len(), m.cfg.vocab);
        Ok(())
    }

    #[test]
    fn block_mode_lets_a_token_see_later_tokens_of_its_block() -> Result<()> {
        let (h, hkv, d, t) = (2usize, 2usize, 4usize, 3usize);
        let dev = Device::Cpu;
        let q = Tensor::from_vec(det_normal(h * t * d, 17), (h, t, d), &dev)?;
        let k = Tensor::from_vec(det_normal(hkv * t * d, 18), (hkv, t, d), &dev)?;
        let v = Tensor::from_vec(det_normal(hkv * t * d, 19), (hkv, t, d), &dev)?;
        let mut v2 = v.flatten_all()?.to_vec1::<f32>()?;
        for head in 0..hkv {
            for i in 0..d {
                v2[(head * t + t - 1) * d + i] += 3.0;
            }
        }
        let v2 = Tensor::from_vec(v2, (hkv, t, d), &dev)?;
        let kt = k.transpose(1, 2)?.contiguous()?;
        let diff = |mode| -> Result<f32> {
            let mask = StepMask::new(mode, t, h / hkv, &dev)?;
            let a = attend(&q, &kt, &v, &[], &mask, 0.5)?;
            let b = attend(&q, &kt, &v2, &[], &mask, 0.5)?;
            Ok((a.narrow(1, 0, 1)? - b.narrow(1, 0, 1)?)?.abs()?.max_all()?.to_scalar::<f32>()?)
        };
        assert_eq!(diff(AttnMode::Causal)?, 0.0, "인과 모드의 첫 토큰은 뒤 토큰을 보지 않습니다");
        assert!(diff(AttnMode::Block)? > 1e-4, "블록 모드의 첫 토큰은 뒤 토큰을 봅니다");
        Ok(())
    }
}
