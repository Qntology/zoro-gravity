//! 시험 전용: 텐서 연산 없이 단순 반복문(f64)으로 쓴 zoro 백본 기준 구현과, 그것과 같은 가중치로 GGUF 를 쓰는 작은 모델 생성기.
//! 연산 구현(`ops`, `model`)이 모양·인덱스·연산 순서를 틀리지 않았는지 보는 용도입니다.
//! 구조 자체(1+w 정규화, 출력 게이트 등)에 대한 이해는 양쪽이 같으므로, 그 이해가 맞는지는 실제 가중치로 확인합니다 (`examples/zoro_probe.rs`).

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use candle_core::delta_rule::GdnConfig;
use candle_core::quantized::gguf_file::Value;
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Tensor};

use super::config::{LayerKind, ZoroConfig};
use crate::ai::verify::det_normal;
use crate::ai::weights::testing::{write_gguf, TempDir};
use crate::ai::weights::WeightFile;

pub fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

pub fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// 헤드마다 단순 반복문으로 계산하는 인과 주의 (GQA). `q`: [H,T,D], `k`·`v`: [Hkv,T,D]
#[allow(clippy::too_many_arguments)]
pub fn naive_attention(q: &[f32], k: &[f32], v: &[f32], h: usize, hkv: usize, tt: usize, d: usize, scale: f32) -> Vec<f32> {
    let mut out = vec![0f32; h * tt * d];
    for head in 0..h {
        let kv = head / (h / hkv);
        for i in 0..tt {
            let mut sc: Vec<f32> = (0..=i)
                .map(|j| (0..d).map(|x| q[(head * tt + i) * d + x] * k[(kv * tt + j) * d + x]).sum::<f32>() * scale)
                .collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            sc.iter_mut().for_each(|s| *s = (*s - m).exp());
            let z: f32 = sc.iter().sum();
            for x in 0..d {
                out[(head * tt + i) * d + x] = sc.iter().enumerate().map(|(j, p)| p / z * v[(kv * tt + j) * d + x]).sum();
            }
        }
    }
    out
}

/// PyTorch 의 정의를 그대로 옮긴 게이트 델타 규칙 기준 구현 (Conv1d 는 상관: `y[t] = Σ_j w[j]·x[t−(K−1)+j]`).
/// 입력 `mixed`: [T,C], `z`: [T,Hv·Dv], `a`·`b`: [T,Hv], `conv`: [C,K]. 결과 [T,Hv·Dv].
#[allow(clippy::too_many_arguments)]
pub fn naive_gdn(
    mixed: &[f32], z: &[f32], a: &[f32], b: &[f32], conv: &[f32], neg_a: &[f32], dt_bias: &[f32], norm: &[f32],
    cfg: &GdnConfig, tt: usize, eps: f64,
) -> Vec<f32> {
    let (hk, hv, dk, dv, kk) = (cfg.num_k_heads, cfg.num_v_heads, cfg.head_k_dim, cfg.head_v_dim, cfg.conv_kernel);
    let c = cfg.conv_dim();
    let mut x = vec![0f64; tt * c];
    for ti in 0..tt {
        for ch in 0..c {
            let mut s = 0f64;
            for j in 0..kk {
                let src = ti as isize - (kk as isize - 1) + j as isize;
                if src >= 0 {
                    s += conv[ch * kk + j] as f64 * mixed[src as usize * c + ch] as f64;
                }
            }
            x[ti * c + ch] = silu(s);
        }
    }
    let (kd, vd) = (hk * dk, hv * dv);
    let l2 = |v: &[f64]| -> Vec<f64> {
        let n = (v.iter().map(|e| e * e).sum::<f64>() + 1e-6).sqrt();
        v.iter().map(|e| e / n).collect()
    };
    let rep = hv / hk;
    let mut state = vec![vec![0f64; dk * dv]; hv];
    let mut out = vec![0f32; tt * vd];
    for ti in 0..tt {
        let row = &x[ti * c..(ti + 1) * c];
        for h in 0..hv {
            let kh = h / rep;
            let q: Vec<f64> = l2(&row[kh * dk..(kh + 1) * dk]).iter().map(|e| e / (dk as f64).sqrt()).collect();
            let k = l2(&row[kd + kh * dk..kd + (kh + 1) * dk]);
            let v = &row[2 * kd + h * dv..2 * kd + (h + 1) * dv];
            let g = neg_a[h] as f64 * softplus(a[ti * hv + h] as f64 + dt_bias[h] as f64);
            let beta = 1.0 / (1.0 + (-(b[ti * hv + h] as f64)).exp());
            let s = &mut state[h];
            s.iter_mut().for_each(|e| *e *= g.exp());
            let mut delta = vec![0f64; dv];
            for j in 0..dv {
                let kv_mem: f64 = (0..dk).map(|i| s[i * dv + j] * k[i]).sum();
                delta[j] = (v[j] - kv_mem) * beta;
            }
            for i in 0..dk {
                for j in 0..dv {
                    s[i * dv + j] += k[i] * delta[j];
                }
            }
            let o: Vec<f64> = (0..dv).map(|j| (0..dk).map(|i| s[i * dv + j] * q[i]).sum()).collect();
            let rms = (o.iter().map(|e| e * e).sum::<f64>() / dv as f64 + eps).sqrt();
            for j in 0..dv {
                let gate = silu(z[ti * vd + h * dv + j] as f64);
                out[ti * vd + h * dv + j] = (o[j] / rms * norm[j] as f64 * gate) as f32;
            }
        }
    }
    out
}

/// 작은 zoro: 층 4개 (선형 3 + 전체 1), hidden 64, 어휘 300. 가중치는 결정적 난수입니다.
pub struct MiniZoro {
    pub cfg: ZoroConfig,
    /// GGUF 이름 → (값, 모양). 정규화 가중치는 파일에 들어간 그대로(= 1 을 더하기 전) 입니다.
    pub raw: HashMap<String, (Vec<f32>, Vec<usize>)>,
}

impl MiniZoro {
    pub fn config() -> ZoroConfig {
        ZoroConfig {
            n_layers: 4,
            hidden: 64,
            ffn: 128,
            vocab: 300,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 32,
            rms_eps: 1e-6,
            rope_dim: 8,
            rope_theta: 10000.0,
            gdn_k_heads: 2,
            gdn_v_heads: 4,
            gdn_k_dim: 16,
            gdn_v_dim: 16,
            conv_kernel: 4,
            layer_kinds: vec![
                LayerKind::LinearAttention,
                LayerKind::LinearAttention,
                LayerKind::LinearAttention,
                LayerKind::FullAttention,
            ],
        }
    }

    pub fn new() -> Self {
        Self::with_gain(1.0)
    }

    /// 선형 가중치를 `gain` 배 한 모델. 이득이 낮으면 부분 출력이 잔차보다 작아서 작은 수치 오차가 덜 증폭됩니다.
    pub fn with_gain(gain: f32) -> Self {
        let cfg = Self::config();
        let mut raw: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
        let mut seed = 100u64;
        let mut put = |name: String, shape: Vec<usize>, scale: f32, shift: f32| {
            seed += 1;
            let n: usize = shape.iter().product();
            let data: Vec<f32> = det_normal(n, seed).iter().map(|x| shift + scale * x).collect();
            raw.insert(name, (data, shape));
        };
        put("token_embd.weight".into(), vec![cfg.vocab, cfg.hidden], 0.5, 0.0);
        put("output_norm.weight".into(), vec![cfg.hidden], 0.1, 0.0);
        for i in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            put(p("attn_norm.weight"), vec![cfg.hidden], 0.1, 0.0);
            put(p("post_attention_norm.weight"), vec![cfg.hidden], 0.1, 0.0);
            put(p("ffn_gate.weight"), vec![cfg.ffn, cfg.hidden], 0.12, 0.0);
            put(p("ffn_up.weight"), vec![cfg.ffn, cfg.hidden], 0.12, 0.0);
            put(p("ffn_down.weight"), vec![cfg.hidden, cfg.ffn], 0.09, 0.0);
            match cfg.kind(i) {
                LayerKind::LinearAttention => {
                    put(p("attn_qkv.weight"), vec![cfg.gdn_conv_dim(), cfg.hidden], 0.12, 0.0);
                    put(p("attn_gate.weight"), vec![cfg.gdn_value_dim(), cfg.hidden], 0.12, 0.0);
                    put(p("ssm_beta.weight"), vec![cfg.gdn_v_heads, cfg.hidden], 0.2, 0.0);
                    put(p("ssm_alpha.weight"), vec![cfg.gdn_v_heads, cfg.hidden], 0.2, 0.0);
                    put(p("ssm_out.weight"), vec![cfg.hidden, cfg.gdn_value_dim()], 0.12, 0.0);
                    put(p("ssm_conv1d.weight"), vec![cfg.gdn_conv_dim(), 1, cfg.conv_kernel], 0.4, 0.0);
                    put(p("ssm_dt.bias"), vec![cfg.gdn_v_heads], 0.5, 0.0);
                    // A_log: 감쇠율 exp(A_log) 가 0.3 ~ 3 정도
                    put(p("ssm_a"), vec![cfg.gdn_v_heads], 0.4, 0.2);
                    put(p("ssm_norm.weight"), vec![cfg.gdn_v_dim], 0.1, 1.0);
                }
                LayerKind::FullAttention => {
                    put(p("attn_q.weight"), vec![cfg.q_proj_rows(), cfg.hidden], 0.12, 0.0);
                    put(p("attn_k.weight"), vec![cfg.n_kv_heads * cfg.head_dim, cfg.hidden], 0.12, 0.0);
                    put(p("attn_v.weight"), vec![cfg.n_kv_heads * cfg.head_dim, cfg.hidden], 0.12, 0.0);
                    put(p("attn_output.weight"), vec![cfg.hidden, cfg.n_heads * cfg.head_dim], 0.1, 0.0);
                    put(p("attn_q_norm.weight"), vec![cfg.head_dim], 0.1, 0.0);
                    put(p("attn_k_norm.weight"), vec![cfg.head_dim], 0.1, 0.0);
                }
            }
        }
        let mut mini = Self { cfg, raw };
        if gain != 1.0 {
            let names: Vec<String> = mini
                .raw
                .keys()
                .filter(|n| {
                    n.as_str() != "token_embd.weight"
                        && !n.contains("norm")
                        && !n.contains("ssm_conv1d")
                        && !n.ends_with("ssm_a")
                        && !n.ends_with("ssm_dt.bias")
                })
                .cloned()
                .collect();
            for n in names {
                if let Some((data, _)) = mini.raw.get_mut(&n) {
                    data.iter_mut().for_each(|v| *v *= gain);
                }
            }
        }
        mini
    }

    fn get(&self, name: &str) -> &[f32] {
        &self.raw[name].0
    }

    /// 양자화 후보인가: 32 의 배수 열을 가진 2차원 선형 가중치 (정규화 · 작은 게이트 제외)
    fn quantizable(&self, name: &str) -> bool {
        let (_, shape) = &self.raw[name];
        shape.len() == 2
            && shape[0] >= 64
            && shape[1] % 32 == 0
            && !name.contains("norm")
            && !name.contains("ssm_alpha")
            && !name.contains("ssm_beta")
    }

    /// GGUF 를 쓰고 엽니다. `quantize` 가 거짓이면 모두 F32 라서 기준 구현과 같은 값입니다.
    pub fn write(&self, dir: &TempDir, quantize: bool) -> Result<Arc<WeightFile>> {
        self.write_with(dir, if quantize { "mini-q8" } else { "mini-f32" }, &|_| quantize)
    }

    /// `select(이름)` 이 참인 양자화 후보만 Q8_0 으로 쓰고 나머지는 F32 로 씁니다.
    pub fn write_with(&self, dir: &TempDir, file_stem: &str, select: &dyn Fn(&str) -> bool) -> Result<Arc<WeightFile>> {
        let mut tensors: Vec<(String, Tensor, GgmlDType)> = Vec::new();
        let mut names: Vec<&String> = self.raw.keys().collect();
        names.sort();
        for name in names {
            let (data, shape) = &self.raw[name];
            tensors.push((
                name.clone(),
                Tensor::from_vec(data.clone(), shape.clone(), &Device::Cpu)?,
                if self.quantizable(name) && select(name) { GgmlDType::Q8_0 } else { GgmlDType::F32 },
            ));
        }
        let c = &self.cfg;
        let meta: Vec<(&str, Value)> = vec![
            ("general.architecture", Value::String("zoro".into())),
            ("zoro.block_count", Value::U32(c.n_layers as u32)),
            ("zoro.embedding_length", Value::U32(c.hidden as u32)),
            ("zoro.feed_forward_length", Value::U32(c.ffn as u32)),
            ("zoro.vocab_size", Value::U32(c.vocab as u32)),
            ("zoro.attention.head_count", Value::U32(c.n_heads as u32)),
            ("zoro.attention.head_count_kv", Value::U32(c.n_kv_heads as u32)),
            ("zoro.attention.key_length", Value::U32(c.head_dim as u32)),
            ("zoro.attention.layer_norm_rms_epsilon", Value::F32(c.rms_eps)),
            ("zoro.rope.dimension_count", Value::U32(c.rope_dim as u32)),
            ("zoro.rope.freq_base", Value::F32(c.rope_theta as f32)),
            ("zoro.full_attention_interval", Value::U32(4)),
            ("zoro.ssm.conv_kernel", Value::U32(c.conv_kernel as u32)),
            ("zoro.ssm.state_size", Value::U32(c.gdn_k_dim as u32)),
            ("zoro.ssm.group_count", Value::U32(c.gdn_k_heads as u32)),
            ("zoro.ssm.time_step_rank", Value::U32(c.gdn_v_heads as u32)),
            ("zoro.ssm.inner_size", Value::U32(c.gdn_value_dim() as u32)),
        ];
        let path = dir.0.join(format!("{file_stem}.gguf"));
        write_gguf(&path, &meta, &tensors)?;
        Ok(Arc::new(WeightFile::open(&path)?))
    }

    // ---- 기준 순전파 (f64) ----

    fn linear(&self, x: &[f64], t: usize, name: &str) -> Vec<f64> {
        let (w, shape) = &self.raw[name];
        let (out, inn) = (shape[0], shape[1]);
        let mut y = vec![0f64; t * out];
        for ti in 0..t {
            for o in 0..out {
                y[ti * out + o] = (0..inn).map(|i| x[ti * inn + i] * w[o * inn + i] as f64).sum();
            }
        }
        y
    }

    /// 마지막 차원(`dim`)마다 정규화. `w` 는 (1+w) 로 곱합니다.
    fn norm(&self, x: &[f64], dim: usize, name: &str) -> Vec<f64> {
        let w = self.get(name);
        let mut y = vec![0f64; x.len()];
        for (r, row) in x.chunks(dim).enumerate() {
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / dim as f64 + self.cfg.rms_eps as f64).sqrt();
            for c in 0..dim {
                y[r * dim + c] = row[c] * inv * (1.0 + w[c] as f64);
            }
        }
        y
    }

    /// 입력 토큰 → 최종 정규화된 은닉 상태 `[T·hidden]`
    pub fn forward(&self, ids: &[u32]) -> Vec<f32> {
        let c = &self.cfg;
        let t = ids.len();
        let h = c.hidden;
        let emb = self.get("token_embd.weight");
        let mut x: Vec<f64> = ids.iter().flat_map(|&id| (0..h).map(move |j| emb[id as usize * h + j] as f64)).collect();
        for i in 0..c.n_layers {
            let p = |s: &str| format!("blk.{i}.{s}");
            let hn = self.norm(&x, h, &p("attn_norm.weight"));
            let mix = match c.kind(i) {
                LayerKind::LinearAttention => self.ref_linear_attention(&hn, t, i),
                LayerKind::FullAttention => self.ref_full_attention(&hn, t, i),
            };
            x.iter_mut().zip(&mix).for_each(|(a, b)| *a += b);
            let hn = self.norm(&x, h, &p("post_attention_norm.weight"));
            let g = self.linear(&hn, t, &p("ffn_gate.weight"));
            let u = self.linear(&hn, t, &p("ffn_up.weight"));
            let a: Vec<f64> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            let m = self.linear(&a, t, &p("ffn_down.weight"));
            x.iter_mut().zip(&m).for_each(|(a, b)| *a += b);
        }
        self.norm(&x, h, "output_norm.weight").into_iter().map(|v| v as f32).collect()
    }

    fn ref_linear_attention(&self, hn: &[f64], t: usize, i: usize) -> Vec<f64> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{i}.{s}");
        let to32 = |v: Vec<f64>| -> Vec<f32> { v.into_iter().map(|e| e as f32).collect() };
        let mixed = to32(self.linear(hn, t, &p("attn_qkv.weight")));
        let z = to32(self.linear(hn, t, &p("attn_gate.weight")));
        let b = to32(self.linear(hn, t, &p("ssm_beta.weight")));
        let a = to32(self.linear(hn, t, &p("ssm_alpha.weight")));
        let neg_a: Vec<f32> = self.get(&p("ssm_a")).iter().map(|v| -(*v as f64).exp() as f32).collect();
        let mut g = GdnConfig::new(c.gdn_k_heads, c.gdn_v_heads, c.gdn_k_dim, c.gdn_v_dim);
        g.conv_kernel = c.conv_kernel;
        let y = naive_gdn(
            &mixed, &z, &a, &b, self.get(&p("ssm_conv1d.weight")), &neg_a, self.get(&p("ssm_dt.bias")),
            self.get(&p("ssm_norm.weight")), &g, t, c.rms_eps as f64,
        );
        let y: Vec<f64> = y.into_iter().map(|v| v as f64).collect();
        self.linear(&y, t, &p("ssm_out.weight"))
    }

    fn ref_full_attention(&self, hn: &[f64], t: usize, i: usize) -> Vec<f64> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{i}.{s}");
        let (nh, nkv, d) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let qg = self.linear(hn, t, &p("attn_q.weight")); // [T, nh * 2d]
        let kk = self.linear(hn, t, &p("attn_k.weight")); // [T, nkv * d]
        let vv = self.linear(hn, t, &p("attn_v.weight"));
        let mut q = vec![0f64; nh * t * d]; // [H, T, D]
        let mut gate = vec![0f64; t * nh * d];
        for ti in 0..t {
            for h in 0..nh {
                for x in 0..d {
                    q[(h * t + ti) * d + x] = qg[ti * nh * 2 * d + h * 2 * d + x];
                    gate[ti * nh * d + h * d + x] = qg[ti * nh * 2 * d + h * 2 * d + d + x];
                }
            }
        }
        // 헤드별 정규화 → RoPE
        let qn = self.norm(&q, d, &p("attn_q_norm.weight"));
        let mut k = vec![0f64; nkv * t * d];
        let mut v = vec![0f64; nkv * t * d];
        for ti in 0..t {
            for h in 0..nkv {
                for x in 0..d {
                    k[(h * t + ti) * d + x] = kk[ti * nkv * d + h * d + x];
                    v[(h * t + ti) * d + x] = vv[ti * nkv * d + h * d + x];
                }
            }
        }
        let kn = self.norm(&k, d, &p("attn_k_norm.weight"));
        let rot = |x: &[f64], heads: usize| -> Vec<f64> {
            let mut y = x.to_vec();
            let half = c.rope_dim / 2;
            for h in 0..heads {
                for pos in 0..t {
                    let base = (h * t + pos) * d;
                    for j in 0..half {
                        let ang = pos as f64 * c.rope_theta.powf(-(2.0 * j as f64) / c.rope_dim as f64);
                        let (cs, sn) = (ang.cos(), ang.sin());
                        let (x1, x2) = (x[base + j], x[base + half + j]);
                        y[base + j] = x1 * cs - x2 * sn;
                        y[base + half + j] = x2 * cs + x1 * sn;
                    }
                }
            }
            y
        };
        let (qr, kr) = (rot(&qn, nh), rot(&kn, nkv));
        let f32s = |v: &[f64]| -> Vec<f32> { v.iter().map(|e| *e as f32).collect() };
        let o = naive_attention(&f32s(&qr), &f32s(&kr), &f32s(&v), nh, nkv, t, d, 1.0 / (d as f32).sqrt());
        // [H,T,D] → [T, H*D] 에 출력 게이트
        let mut flat = vec![0f64; t * nh * d];
        for h in 0..nh {
            for ti in 0..t {
                for x in 0..d {
                    let g = 1.0 / (1.0 + (-gate[ti * nh * d + h * d + x]).exp());
                    flat[ti * nh * d + h * d + x] = o[(h * t + ti) * d + x] as f64 * g;
                }
            }
        }
        self.linear(&flat, t, &p("attn_output.weight"))
    }
}
