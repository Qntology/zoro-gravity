//! SigLIP2: 문서 페이지에서 읽을 곳을 찾아 자르는 모델. logis-center 의 방식을 따르되 코드는 직접 구현했습니다.
//!
//! - `config`: 구조 값
//! - `image`: NaFlex 전처리
//! - `tower` · `vision` · `text`: 비전 · 텍스트 타워
//! - `probe`: 패치 특징 → 영역 종류 (시각 앵커)
//! - `ink` · `layout`: 픽셀 분석과 조각 나누기
//! - `arena`: 칸마다 종류끼리 겨루는 패치 경쟁 (NMS)
//! - `plan`: 조각마다 읽을지 정하기
//!
//! 단계마다 쓰는 타워만 올립니다. 페이지를 읽을 때는 비전 타워만, 앵커 문구를 새로 인코딩할 때는 텍스트 타워만 올라옵니다.

pub mod arena;
pub mod config;
pub mod image;
pub mod ink;
pub mod layout;
pub mod plan;
pub mod probe;
pub mod text;
mod tower;
pub mod vision;

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::{DType, Device};

use self::config::Siglip2Config;
use self::image::PatchInput;
use self::text::TextTokenizer;
use super::gguf_model::load_active;
use super::residency::{Group, ModelId, ModelSpec, Phase, Residency, SharedResidency};
use super::weights::WeightFile;

/// 텍스트 타워에 한 번에 넣는 문구 수
const TEXT_BATCH: usize = 16;
/// 비전 타워에 한 번에 넣는 페이지 수
const PAGE_BATCH: usize = 8;

pub fn vision_phase() -> Phase {
    Phase::new("siglip2:비전", [Group::Vision])
}

pub fn text_phase() -> Phase {
    Phase::new("siglip2:텍스트", [Group::Text])
}

/// 패치별 특징 격자. 벡터는 모두 L2 정규화되어 있고 호스트 메모리에 있습니다.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchGrid {
    pub rows: usize,
    pub cols: usize,
    pub dim: usize,
    /// `[rows × cols, dim]` 를 행 우선으로 편 값
    pub patches: Vec<f32>,
    pub pooled: Vec<f32>,
    /// 원본 크기 (픽셀)
    pub width: u32,
    pub height: u32,
}

impl PatchGrid {
    pub fn len(&self) -> usize {
        self.rows * self.cols
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn patch(&self, i: usize) -> &[f32] {
        &self.patches[i * self.dim..(i + 1) * self.dim]
    }
}

fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

pub struct Siglip2 {
    spec: ModelSpec,
    cfg: Arc<Siglip2Config>,
    tokenizer: TextTokenizer,
    residency: SharedResidency,
}

impl Siglip2 {
    /// `model_dir/active.json` 이 가리키는 GGUF 를 엽니다. 가중치는 쓸 때 단계별로 올라옵니다.
    pub fn open(model_dir: &Path, residency: SharedResidency) -> Result<Self> {
        let active = load_active(model_dir)?;
        if active.architecture != "siglip2" {
            bail!("siglip2 모델 폴더가 아닙니다: {} ({})", model_dir.display(), active.architecture);
        }
        let file = Arc::new(WeightFile::open(&active.path)?);
        let mmproj = active.open_mmproj()?;
        let cfg = Arc::new(Siglip2Config::from_files(&file, mmproj.as_deref().unwrap_or(&file))?);
        let tokenizer = TextTokenizer::from_file(&model_dir.join("tokenizer.json"), cfg.max_len)?;
        let spec = ModelSpec::new(ModelId::Siglip2, file);
        let spec = match mmproj {
            Some(mmproj) => spec.with_mmproj(mmproj),
            None => spec,
        };
        spec.announce(&vision_phase());
        Ok(Self { spec, cfg, tokenizer, residency })
    }

    pub fn config(&self) -> &Siglip2Config {
        &self.cfg
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// 페이지 한 장을 읽어 특징 격자를 돌려줍니다. 비전 타워만 올립니다.
    pub fn encode_image(&self, input: &PatchInput) -> Result<PatchGrid> {
        Ok(self.encode_images(std::slice::from_ref(input))?.remove(0))
    }

    /// 페이지들을 읽습니다. 격자가 같은 페이지는 묶어서 계산하고, 결과는 입력 순서입니다.
    pub fn encode_images(&self, inputs: &[PatchInput]) -> Result<Vec<PatchGrid>> {
        let mut res = self.residency.lock().unwrap_or_else(|p| p.into_inner());
        self.encode_images_locked(&mut res, inputs)
    }

    pub fn encode_images_locked(&self, res: &mut Residency, inputs: &[PatchInput]) -> Result<Vec<PatchGrid>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        let handles = res.enter(&self.spec, &vision_phase())?;
        let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (i, input) in inputs.iter().enumerate() {
            groups.entry((input.rows, input.cols)).or_default().push(i);
        }
        let d = self.cfg.dim();
        let mut out: Vec<Option<PatchGrid>> = inputs.iter().map(|_| None).collect();
        for members in groups.values() {
            for chunk in members.chunks(PAGE_BATCH) {
                let images: Vec<&PatchInput> = chunk.iter().map(|&i| &inputs[i]).collect();
                let y = vision::encode(&self.cfg, &handles[0], &images, res.device())?;
                let patches = y.patches.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let pooled = y.pooled.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                for (j, &i) in chunk.iter().enumerate() {
                    let input = &inputs[i];
                    let n = input.rows * input.cols * d;
                    let mut p = patches[j * n..(j + 1) * n].to_vec();
                    p.chunks_mut(d).for_each(normalize);
                    let mut q = pooled[j * d..(j + 1) * d].to_vec();
                    normalize(&mut q);
                    out[i] = Some(PatchGrid { rows: input.rows, cols: input.cols, dim: d, patches: p, pooled: q, width: input.width, height: input.height });
                }
            }
        }
        Ok(out.into_iter().flatten().collect())
    }

    /// 문구들을 공유 공간 벡터(L2 정규화)로 바꿉니다. 텍스트 타워만 올립니다.
    pub fn encode_phrases(&self, phrases: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut res = self.residency.lock().unwrap_or_else(|p| p.into_inner());
        self.encode_phrases_locked(&mut res, phrases)
    }

    pub fn encode_phrases_locked(&self, res: &mut Residency, phrases: &[String]) -> Result<Vec<Vec<f32>>> {
        if phrases.is_empty() {
            return Ok(Vec::new());
        }
        let handles = res.enter(&self.spec, &text_phase())?;
        let mut out = Vec::with_capacity(phrases.len());
        for chunk in phrases.chunks(TEXT_BATCH) {
            let ids: Vec<Vec<u32>> = chunk.iter().map(|p| self.tokenizer.encode(p)).collect::<Result<_>>()?;
            let y = text::encode(&self.cfg, &handles[0], &self.spec.file, &ids, res.device())?;
            let flat = y.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            for row in flat.chunks(self.cfg.dim()) {
                let mut v = row.to_vec();
                normalize(&mut v);
                out.push(v);
            }
        }
        Ok(out)
    }

    /// 이미지-문구 일치도 로짓의 배율과 편향 (`logit_scale`, `logit_bias`)
    pub fn logit_params(&self) -> Result<(f32, f32)> {
        let read = |name: &str| -> Result<f32> {
            Ok(self.spec.file.read_dense(name, &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?[0])
        };
        Ok((read("logit_scale")?, read("logit_bias")?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::accel;
    use crate::ai::residency::{Action, Budget, TEXT_TOKEN_TABLE};
    use crate::ai::verify::det_normal;
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use candle_core::quantized::gguf_file::Value;
    use candle_core::quantized::GgmlDType;
    use candle_core::Tensor;
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::Tokenizer;

    const D: usize = 32;
    const HEADS: usize = 2;
    const FFN: usize = 64;
    const PATCH: usize = 2;
    const SIDE: usize = 3;
    const LEN: usize = 5;
    const VOCAB: usize = 40;

    fn small(n: usize, seed: u64, gain: f32) -> Vec<f32> {
        det_normal(n, seed).into_iter().map(|v| v * gain).collect()
    }

    /// 모든 텐서: (이름, 모양, 값)
    fn tensors() -> Vec<(String, Vec<usize>, Vec<f32>)> {
        let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut seed = 100u64;
        let mut add = |name: String, shape: Vec<usize>, gain: f32| {
            seed += 1;
            let n = shape.iter().product();
            t.push((name, shape, small(n, seed, gain)));
        };
        for tower in ["vision_model", "text_model"] {
            let b = format!("{tower}.encoder.layers.0");
            for name in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"] {
                add(format!("{b}.{name}.weight"), vec![D, D], 0.2);
                add(format!("{b}.{name}.bias"), vec![D], 0.1);
            }
            add(format!("{b}.mlp.fc1.weight"), vec![FFN, D], 0.2);
            add(format!("{b}.mlp.fc1.bias"), vec![FFN], 0.1);
            add(format!("{b}.mlp.fc2.weight"), vec![D, FFN], 0.2);
            add(format!("{b}.mlp.fc2.bias"), vec![D], 0.1);
            for name in ["layer_norm1", "layer_norm2"] {
                add(format!("{b}.{name}.weight"), vec![D], 0.5);
                add(format!("{b}.{name}.bias"), vec![D], 0.1);
            }
        }
        add("vision_model.embeddings.patch_embedding.weight".into(), vec![D, 3 * PATCH * PATCH], 0.3);
        add("vision_model.embeddings.patch_embedding.bias".into(), vec![D], 0.1);
        add("vision_model.embeddings.position_embedding.weight".into(), vec![SIDE * SIDE, D], 0.3);
        add("vision_model.post_layernorm.weight".into(), vec![D], 0.5);
        add("vision_model.post_layernorm.bias".into(), vec![D], 0.1);
        add("vision_model.head.probe".into(), vec![1, 1, D], 0.3);
        add("vision_model.head.attention.in_proj_weight".into(), vec![3 * D, D], 0.2);
        add("vision_model.head.attention.in_proj_bias".into(), vec![3 * D], 0.1);
        add("vision_model.head.attention.out_proj.weight".into(), vec![D, D], 0.2);
        add("vision_model.head.attention.out_proj.bias".into(), vec![D], 0.1);
        add("vision_model.head.layernorm.weight".into(), vec![D], 0.5);
        add("vision_model.head.layernorm.bias".into(), vec![D], 0.1);
        add("vision_model.head.mlp.fc1.weight".into(), vec![FFN, D], 0.2);
        add("vision_model.head.mlp.fc1.bias".into(), vec![FFN], 0.1);
        add("vision_model.head.mlp.fc2.weight".into(), vec![D, FFN], 0.2);
        add("vision_model.head.mlp.fc2.bias".into(), vec![D], 0.1);
        add(TEXT_TOKEN_TABLE.into(), vec![VOCAB, D], 0.3);
        add(config::TEXT_POS.into(), vec![LEN, D], 0.3);
        add("text_model.final_layer_norm.weight".into(), vec![D], 0.5);
        add("text_model.final_layer_norm.bias".into(), vec![D], 0.1);
        add("text_model.head.weight".into(), vec![D, D], 0.2);
        add("text_model.head.bias".into(), vec![D], 0.1);
        add("logit_scale".into(), vec![1], 0.1);
        add("logit_bias".into(), vec![1], 0.1);
        t
    }

    fn tensor_list() -> Vec<(String, Tensor, GgmlDType)> {
        tensors()
            .into_iter()
            .map(|(n, shape, v)| {
                let dtype = if n == TEXT_TOKEN_TABLE { GgmlDType::Q8_0 } else { GgmlDType::F32 };
                (n, Tensor::from_vec(v, shape, &Device::Cpu).unwrap(), dtype)
            })
            .collect()
    }

    fn fixture(dir: &TempDir) -> (Siglip2, SharedResidency) {
        fixture_with(dir, tensor_list(), None)
    }

    /// `mmproj` 를 주면 비전 타워는 그 파일에 있는 모델입니다.
    fn fixture_with(dir: &TempDir, list: Vec<(String, Tensor, GgmlDType)>, mmproj: Option<Arc<WeightFile>>) -> (Siglip2, SharedResidency) {
        fixture_on(dir, list, mmproj, &Device::Cpu)
    }

    fn fixture_on(dir: &TempDir, list: Vec<(String, Tensor, GgmlDType)>, mmproj: Option<Arc<WeightFile>>, device: &Device) -> (Siglip2, SharedResidency) {
        let path = dir.0.join("siglip2.gguf");
        let tower = |k: &str, v: u32| (k.to_string(), Value::U32(v));
        let meta: Vec<(String, Value)> = [
            tower("siglip2.vision.block_count", 1),
            tower("siglip2.vision.embedding_length", D as u32),
            tower("siglip2.vision.feed_forward_length", FFN as u32),
            tower("siglip2.vision.attention.head_count", HEADS as u32),
            tower("siglip2.text.block_count", 1),
            tower("siglip2.text.embedding_length", D as u32),
            tower("siglip2.text.feed_forward_length", FFN as u32),
            tower("siglip2.text.attention.head_count", HEADS as u32),
        ]
        .into_iter()
        .collect();
        let refs: Vec<(&str, Value)> = meta.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        write_gguf(&path, &refs, &list).unwrap();
        let file = Arc::new(WeightFile::open(&path).unwrap());
        let cfg = Arc::new(Siglip2Config::from_files(&file, mmproj.as_deref().unwrap_or(&file)).unwrap());
        assert_eq!((cfg.patch, cfg.pos_side, cfg.vocab, cfg.max_len), (PATCH, SIDE, VOCAB, LEN));

        let vocab = [("<pad>", 0u32), ("<eos>", 1), ("<unk>", 2), ("a", 3), ("b", 4)].iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let wl = WordLevel::builder().vocab(vocab).unk_token("<unk>".into()).build().unwrap();
        let mut tk = Tokenizer::new(wl);
        tk.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace {}));
        let residency = Residency::shared(device.clone(), Budget::unlimited());
        let spec = ModelSpec::new(ModelId::Siglip2, file);
        let spec = match mmproj {
            Some(mmproj) => spec.with_mmproj(mmproj),
            None => spec,
        };
        let model = Siglip2 { spec, cfg, tokenizer: TextTokenizer::from_tokenizer(tk, LEN), residency: residency.clone() };
        (model, residency)
    }

    fn input() -> PatchInput {
        let (rows, cols) = (2, 3);
        PatchInput { pixels: det_normal(rows * cols * 3 * PATCH * PATCH, 9), rows, cols, width: 60, height: 40 }
    }

    fn get<'a>(t: &'a [(String, Vec<usize>, Vec<f32>)], name: &str) -> &'a [f32] {
        &t.iter().find(|(n, ..)| n == name).unwrap_or_else(|| panic!("{name}")).2
    }

    /// 행 하나에 `y = W x + b`
    fn linear(t: &[(String, Vec<usize>, Vec<f32>)], name: &str, x: &[f32]) -> Vec<f32> {
        let (w, b) = (get(t, &format!("{name}.weight")), get(t, &format!("{name}.bias")));
        (0..b.len()).map(|i| b[i] + (0..x.len()).map(|j| w[i * x.len() + j] * x[j]).sum::<f32>()).collect()
    }

    fn norm(t: &[(String, Vec<usize>, Vec<f32>)], name: &str, x: &[f32]) -> Vec<f32> {
        let (w, b) = (get(t, &format!("{name}.weight")), get(t, &format!("{name}.bias")));
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let inv = 1.0 / (var as f64 + config::NORM_EPS).sqrt() as f32;
        x.iter().enumerate().map(|(i, v)| (v - mean) * inv * w[i] + b[i]).collect()
    }

    fn gelu(x: f32) -> f32 {
        let a = 0.797_884_6 * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + a.tanh())
    }

    fn mlp(t: &[(String, Vec<usize>, Vec<f32>)], name: &str, x: &[f32]) -> Vec<f32> {
        let h: Vec<f32> = linear(t, &format!("{name}.fc1"), x).into_iter().map(gelu).collect();
        linear(t, &format!("{name}.fc2"), &h)
    }

    /// 헤드마다 `softmax(q kᵀ / √d) v`, 결과를 이어 붙입니다.
    fn attend(q: &[Vec<f32>], k: &[Vec<f32>], v: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let d = D / HEADS;
        q.iter()
            .map(|qr| {
                let mut out = Vec::with_capacity(D);
                for h in 0..HEADS {
                    let qh = &qr[h * d..(h + 1) * d];
                    let scores: Vec<f32> = k.iter().map(|kr| (0..d).map(|i| qh[i] * kr[h * d + i]).sum::<f32>() / (d as f32).sqrt()).collect();
                    let top = scores.iter().copied().fold(f32::MIN, f32::max);
                    let exp: Vec<f32> = scores.iter().map(|s| (s - top).exp()).collect();
                    let sum: f32 = exp.iter().sum();
                    for i in 0..d {
                        out.push(exp.iter().zip(v).map(|(e, vr)| e * vr[h * d + i]).sum::<f32>() / sum);
                    }
                }
                out
            })
            .collect()
    }

    fn encoder_layer(t: &[(String, Vec<usize>, Vec<f32>)], tower: &str, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let b = format!("{tower}.encoder.layers.0");
        let n: Vec<Vec<f32>> = x.iter().map(|r| norm(t, &format!("{b}.layer_norm1"), r)).collect();
        let proj = |name: &str| -> Vec<Vec<f32>> { n.iter().map(|r| linear(t, &format!("{b}.self_attn.{name}"), r)).collect() };
        let o = attend(&proj("q_proj"), &proj("k_proj"), &proj("v_proj"));
        let x: Vec<Vec<f32>> = x.iter().zip(&o).map(|(r, a)| {
            let y = linear(t, &format!("{b}.self_attn.out_proj"), a);
            r.iter().zip(&y).map(|(p, q)| p + q).collect()
        }).collect();
        x.iter()
            .map(|r| {
                let y = mlp(t, &format!("{b}.mlp"), &norm(t, &format!("{b}.layer_norm2"), r));
                r.iter().zip(&y).map(|(p, q)| p + q).collect()
            })
            .collect()
    }

    fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < tol, "{what} [{i}]: {x} != {y}");
        }
    }

    #[test]
    #[ignore = "GPU 가 필요합니다"]
    fn device_towers_match_the_cpu() -> Result<()> {
        let device = accel::test_gpu();
        let (gpu_dir, cpu_dir) = (TempDir::new("siglip2-gpu"), TempDir::new("siglip2-cpu"));
        let (on_gpu, _) = fixture_on(&gpu_dir, tensor_list(), None, &device);
        let (on_cpu, _) = fixture(&cpu_dir);
        let (img, phrases) = (input(), vec!["a b".to_string(), "b".to_string()]);
        let (a, b) = (on_gpu.encode_images(std::slice::from_ref(&img))?, on_cpu.encode_images(std::slice::from_ref(&img))?);
        close(&a[0].patches, &b[0].patches, 1e-3, "패치");
        close(&a[0].pooled, &b[0].pooled, 1e-3, "풀링");
        let (x, y) = (on_gpu.encode_phrases(&phrases)?, on_cpu.encode_phrases(&phrases)?);
        for (p, q) in x.iter().zip(&y) {
            close(p, q, 1e-3, "문구");
        }
        println!("[DEVICE-CHECK] {} siglip2: 비전 · 텍스트 타워가 CPU 와 1e-3 안에서 일치", accel::backend_name(&device));
        Ok(())
    }

    #[test]
    fn the_vision_tower_matches_a_loop_by_loop_forward() -> Result<()> {
        let dir = TempDir::new("siglip2-v");
        let (model, residency) = fixture(&dir);
        let t = tensors();
        let img = input();
        let got = {
            let res = residency.lock().unwrap();
            drop(res);
            let mut res = residency.lock().unwrap();
            let h = res.enter(&model.spec, &vision_phase())?;
            let out = vision::encode(&model.cfg, &h[0], &[&img], &Device::Cpu)?;
            (out.patches.flatten_all()?.to_vec1::<f32>()?, out.pooled.flatten_all()?.to_vec1::<f32>()?)
        };

        // 패치 임베딩 + 격자에 맞춘 위치 임베딩
        let plen = 3 * PATCH * PATCH;
        let pos = image::interpolate_grid(get(&t, config::VISION_POS), SIDE, D, img.rows, img.cols);
        let x: Vec<Vec<f32>> = (0..img.n_patches())
            .map(|i| {
                let e = linear(&t, "vision_model.embeddings.patch_embedding", &img.pixels[i * plen..(i + 1) * plen]);
                e.iter().zip(&pos[i * D..(i + 1) * D]).map(|(a, b)| a + b).collect()
            })
            .collect();
        let x = encoder_layer(&t, "vision_model", &x);
        let hidden: Vec<Vec<f32>> = x.iter().map(|r| norm(&t, "vision_model.post_layernorm", r)).collect();

        let (w, b) = (get(&t, "vision_model.head.attention.in_proj_weight"), get(&t, "vision_model.head.attention.in_proj_bias"));
        let slot = |s: usize, r: &[f32]| -> Vec<f32> { (0..D).map(|i| b[s * D + i] + (0..D).map(|j| w[(s * D + i) * D + j] * r[j]).sum::<f32>()).collect() };
        let residual = |o: Vec<f32>| -> Vec<f32> {
            let h = mlp(&t, "vision_model.head.mlp", &norm(&t, "vision_model.head.layernorm", &o));
            o.iter().zip(&h).map(|(a, b)| a + b).collect()
        };
        // 패치 특징: 패치의 값 투영 → out_proj → 잔차 블록
        let want: Vec<f32> = hidden.iter().flat_map(|r| residual(linear(&t, "vision_model.head.attention.out_proj", &slot(2, r)))).collect();
        close(&got.0, &want, 2e-4, "패치 특징");
        // 풀링: probe 가 패치를 읽은 값 → out_proj → 잔차 블록
        let q = vec![slot(0, get(&t, "vision_model.head.probe"))];
        let k: Vec<Vec<f32>> = hidden.iter().map(|r| slot(1, r)).collect();
        let v: Vec<Vec<f32>> = hidden.iter().map(|r| slot(2, r)).collect();
        let pooled = residual(linear(&t, "vision_model.head.attention.out_proj", &attend(&q, &k, &v)[0]));
        close(&got.1, &pooled, 2e-4, "풀링");
        Ok(())
    }

    #[test]
    fn a_vision_tower_in_its_own_file_reads_the_same() -> Result<()> {
        let dir = TempDir::new("siglip2-mmproj");
        let (whole, _) = fixture(&dir);
        let split = TempDir::new("siglip2-mmproj-split");
        let (vision, rest): (Vec<_>, Vec<_>) = tensor_list().into_iter().partition(|(n, ..)| n.starts_with("vision_model."));
        let path = split.0.join("mmproj-siglip2-bf16.gguf");
        write_gguf(&path, &[("general.architecture", Value::String("siglip2".into()))], &vision).unwrap();
        let (parted, residency) = fixture_with(&split, rest, Some(Arc::new(WeightFile::open(&path)?)));
        assert!(parted.spec.file.names_with_prefix("vision_model.").is_empty());

        let (a, b) = (whole.encode_image(&input())?, parted.encode_image(&input())?);
        close(&a.patches, &b.patches, 1e-6, "패치 특징");
        assert_eq!(residency.lock().unwrap().resident_groups(), vec![Group::Vision]);
        let phrases = ["a b".to_string()];
        close(&whole.encode_phrases(&phrases)?[0], &parted.encode_phrases(&phrases)?[0], 1e-6, "문구");
        assert_eq!(residency.lock().unwrap().resident_groups(), vec![Group::Text], "문구를 읽을 때 비전 파일의 묶음은 내려갑니다");
        Ok(())
    }

    #[test]
    fn pages_read_together_match_pages_read_one_by_one() -> Result<()> {
        let dir = TempDir::new("siglip2-b");
        let (model, _) = fixture(&dir);
        let other = |seed: u64, rows: usize, cols: usize| PatchInput { pixels: det_normal(rows * cols * 3 * PATCH * PATCH, seed), rows, cols, width: 60, height: 40 };
        // 격자가 다른 페이지가 끼어 있어도 순서가 지켜져야 합니다.
        let pages = vec![input(), other(21, 3, 2), other(22, 2, 3), other(23, 2, 3)];
        let together = model.encode_images(&pages)?;
        for (page, got) in pages.iter().zip(&together) {
            let alone = model.encode_image(page)?;
            assert_eq!((got.rows, got.cols), (alone.rows, alone.cols));
            close(&got.patches, &alone.patches, 1e-5, "패치 특징");
            close(&got.pooled, &alone.pooled, 1e-5, "풀링");
        }
        assert_ne!(together[0].patches, together[2].patches, "서로 다른 페이지가 섞이면 안 됩니다");
        assert!(model.encode_images(&[])?.is_empty());
        Ok(())
    }

    #[test]
    fn the_text_tower_reads_every_position_and_pools_the_last_one() -> Result<()> {
        let dir = TempDir::new("siglip2-t");
        let (model, residency) = fixture(&dir);
        let t = tensors();
        let ids: Vec<u32> = vec![3, 4, 1, 0, 0];
        let got = {
            let mut res = residency.lock().unwrap();
            let h = res.enter(&model.spec, &text_phase())?;
            text::encode(&model.cfg, &h[0], &model.spec.file, std::slice::from_ref(&ids), &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?
        };
        // 토큰 임베딩은 Q8_0 이므로 파일에서 읽은 값을 그대로 씁니다.
        let table = model.spec.file.gather_rows(TEXT_TOKEN_TABLE, &ids, &Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let pos = get(&t, config::TEXT_POS);
        let x: Vec<Vec<f32>> = (0..LEN).map(|i| (0..D).map(|j| table[i * D + j] + pos[i * D + j]).collect()).collect();
        let x = encoder_layer(&t, "text_model", &x);
        let last = norm(&t, "text_model.final_layer_norm", &x[LEN - 1]);
        close(&got, &linear(&t, "text_model.head", &last), 2e-4, "텍스트 벡터");
        Ok(())
    }

    #[test]
    fn each_step_loads_only_its_own_tower() -> Result<()> {
        let dir = TempDir::new("siglip2-r");
        let (model, residency) = fixture(&dir);
        let grid = model.encode_image(&input())?;
        assert_eq!(residency.lock().unwrap().resident_groups(), vec![Group::Vision]);
        let v = model.encode_phrases(&["a b".to_string(), "b".to_string()])?;
        assert_eq!(residency.lock().unwrap().resident_groups(), vec![Group::Text], "문구를 읽을 때 비전 타워는 내려갑니다");
        let r = residency.lock().unwrap();
        let log: Vec<(Action, Group)> = r.history().iter().map(|t| (t.action, t.group)).collect();
        assert_eq!(log, vec![(Action::Load, Group::Vision), (Action::Release, Group::Vision), (Action::Load, Group::Text)]);
        drop(r);

        assert_eq!((grid.rows, grid.cols, grid.dim, grid.len()), (2, 3, D, 6));
        let unit = |x: &[f32]| (x.iter().map(|a| a * a).sum::<f32>().sqrt() - 1.0).abs() < 1e-4;
        assert!((0..6).all(|i| unit(grid.patch(i))) && unit(&grid.pooled));
        assert_eq!(v.len(), 2);
        assert!(v.iter().all(|x| unit(x)));
        assert_ne!(v[0], v[1], "문구가 다르면 벡터도 다릅니다");
        assert_eq!(model.encode_phrases(&[])?.len(), 0);
        Ok(())
    }

    #[test]
    fn the_token_table_stays_in_the_file() -> Result<()> {
        let dir = TempDir::new("siglip2-e");
        let (model, residency) = fixture(&dir);
        let names = Group::Text.tensor_names(&model.spec.file);
        assert!(!names.iter().any(|n| n == TEXT_TOKEN_TABLE), "토큰 표는 필요한 행만 읽습니다");
        assert!(names.iter().all(|n| n.starts_with("text_model.")));
        assert!(Group::Vision.tensor_names(&model.spec.file).iter().all(|n| n.starts_with("vision_model.")));
        drop(residency);
        Ok(())
    }

    #[test]
    fn the_logit_parameters_come_from_the_file() -> Result<()> {
        let dir = TempDir::new("siglip2-l");
        let (model, _) = fixture(&dir);
        let t = tensors();
        let (scale, bias) = model.logit_params()?;
        assert_eq!((scale, bias), (get(&t, "logit_scale")[0], get(&t, "logit_bias")[0]));
        Ok(())
    }
}
