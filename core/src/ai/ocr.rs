//! 문서 인식(gocr)의 실행 구조: 이미지 → (비전 가중치만) → 특징 → (비전을 내림) → (언어 가중치만) → 글자.
//!
//! logis-center 의 검색 흐름과 같은 모양입니다. 비전 인코더로 한 번에 인코딩해 두고 즉시 내린 뒤, 언어 모델을 올립니다.
//! - 이미지를 읽는 동안에는 언어 가중치(텍스트 블록 · 출력)가 장치에 없습니다.
//! - 글자를 디코딩하는 동안에는 비전 가중치가 장치에 없습니다. 비전 단계의 결과(특징)만 넘어옵니다.
//! - 토큰 임베딩 표는 올리지 않고 필요한 행만 파일에서 읽습니다.
//! - 이미지가 여러 장이면 비전 단계를 한 번만 거칩니다 (장마다 비전과 언어를 오가며 올렸다 내리지 않습니다).
//!
//! 이 모듈은 단계 전환과 가중치 전달 구조를 맡고, 실제 계산은 `VisionEncoder` · `LanguageDecoder` 구현(`ai::gocr`)이 맡습니다.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde_json::json;

use super::gguf_model::load_active;
use super::residency::{Group, Loaded, ModelId, ModelSpec, Phase, Prepare, Residency, SharedResidency};
use super::trace;
use super::weights::WeightFile;

/// 단계 안에서 콜백이 읽을 수 있는 상주 상태 (읽기 전용). 진단과 시험용입니다.
pub struct PhaseCtx<'a> {
    pub residency: &'a Residency,
}

/// 언어 단계에서 받는 가중치. 비전 가중치를 담는 칸이 아예 없습니다.
pub struct LanguageWeights {
    pub blocks: Arc<Loaded>,
    pub output: Arc<Loaded>,
}

/// 비전 단계: 비전 가중치만 받아 이미지를 특징으로 바꿉니다.
pub trait VisionEncoder: Send + Sync {
    type Input;
    /// 비전이 내려간 뒤에도 남는 결과. 장치 가중치를 가리키지 않는 값(활성값 · 호스트 데이터)이어야 합니다.
    type Features;
    fn encode(&self, ctx: &PhaseCtx<'_>, vision: &Loaded, input: &Self::Input) -> Result<Self::Features>;

    /// 입력을 모두 읽어 입력 순서대로 돌려줍니다. 구현은 입력을 묶어서 계산해도 됩니다.
    fn encode_many(&self, ctx: &PhaseCtx<'_>, vision: &Loaded, inputs: &[Self::Input]) -> Result<Vec<Self::Features>> {
        inputs.iter().map(|i| self.encode(ctx, vision, i)).collect()
    }
}

/// 언어 단계: 언어 가중치만 받아 특징을 글자로 바꿉니다.
pub trait LanguageDecoder: Send + Sync {
    type Features;
    type Output;
    fn decode(&self, ctx: &PhaseCtx<'_>, language: &LanguageWeights, features: Self::Features) -> Result<Self::Output>;
}

/// 비전 단계에 필요한 묶음: 비전 타워 하나
pub fn vision_phase() -> Phase {
    Phase::new("gocr:비전", [Group::Vision])
}

/// 언어 단계에 필요한 묶음: 텍스트 블록 전부, 출력 (임베딩은 행만 파일에서 읽습니다)
pub fn language_phase(blocks: usize) -> Phase {
    Phase::new("gocr:언어", [Group::blocks(0..blocks), Group::Output])
}

pub struct OcrEngine<E: VisionEncoder, D: LanguageDecoder<Features = E::Features>> {
    spec: ModelSpec,
    residency: SharedResidency,
    blocks: usize,
    encoder: E,
    decoder: D,
}

impl<E, D> OcrEngine<E, D>
where
    E: VisionEncoder,
    D: LanguageDecoder<Features = E::Features>,
{
    /// `model_dir/active.json` 이 가리키는 gocr GGUF 를 엽니다. 가중치는 여기서 올리지 않습니다.
    pub fn open(model_dir: &Path, residency: SharedResidency, encoder: E, decoder: D) -> Result<Self> {
        let active = load_active(model_dir)?;
        if active.architecture != "gocr" {
            bail!("gocr 모델 폴더가 아닙니다: {} ({})", model_dir.display(), active.architecture);
        }
        let file = Arc::new(WeightFile::open(&active.path)?);
        let engine = Self::from_file(file, residency, encoder, decoder)?;
        Ok(match active.open_mmproj()? {
            Some(mmproj) => engine.with_mmproj(mmproj),
            None => engine,
        })
    }

    pub fn from_file(file: Arc<WeightFile>, residency: SharedResidency, encoder: E, decoder: D) -> Result<Self> {
        let blocks = file.meta_usize("gocr.block_count").context("gocr GGUF 가 아닙니다")?;
        Ok(Self { spec: ModelSpec::new(ModelId::Gocr, file), residency, blocks, encoder, decoder })
    }

    /// 올릴 때 한 번 하는 가중치 변환을 정합니다.
    pub fn with_prepare(mut self, prepare: Prepare) -> Self {
        self.spec = self.spec.with_prepare(prepare);
        self
    }

    /// 비전 타워를 따로 둔 파일을 붙입니다.
    pub fn with_mmproj(mut self, mmproj: Arc<WeightFile>) -> Self {
        self.spec = self.spec.with_mmproj(mmproj);
        self
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// 쓸 단계의 파일을 배경에서 미리 읽습니다.
    pub fn announce(&self) {
        self.spec.announce(&vision_phase());
        self.spec.announce(&language_phase(self.blocks));
    }

    /// 입력들을 모두 인식합니다. 비전 단계 한 번 → 비전을 내림 → 언어 단계 한 번의 순서입니다.
    /// 호출하는 동안 상주 관리자를 잡고 있어서, 다른 모델(zoro 등)은 이 호출이 끝난 뒤에 올라옵니다.
    pub fn recognize(&self, inputs: &[E::Input]) -> Result<Vec<D::Output>> {
        let mut res = self.residency.lock().unwrap_or_else(|p| p.into_inner());
        self.recognize_locked(&mut res, inputs)
    }

    pub fn recognize_locked(&self, res: &mut Residency, inputs: &[E::Input]) -> Result<Vec<D::Output>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }
        // 1. 비전 단계: 비전 타워만 올립니다 (다른 모델이 올라와 있었다면 그 모델을 먼저 내립니다).
        let t = Instant::now();
        let features = {
            let handles = res.enter(&self.spec, &vision_phase())?;
            self.encoder.encode_many(&PhaseCtx { residency: &*res }, &handles[0], inputs)?
            // 핸들을 여기서 놓아야 다음 단계에서 비전을 내릴 수 있습니다.
        };
        trace::record("gocr", "비전", t.elapsed().as_secs_f64() * 1e3, json!({ "inputs": inputs.len() }));
        // 2. 언어 단계: 비전을 내리고 언어 가중치만 올립니다.
        let t = Instant::now();
        let handles = res.enter(&self.spec, &language_phase(self.blocks))?;
        let weights = LanguageWeights { blocks: Arc::clone(&handles[0]), output: Arc::clone(&handles[1]) };
        drop(handles);
        let mut out = Vec::with_capacity(features.len());
        for f in features {
            out.push(self.decoder.decode(&PhaseCtx { residency: &*res }, &weights, f)?);
        }
        trace::record("gocr", "언어", t.elapsed().as_secs_f64() * 1e3, json!({ "inputs": inputs.len() }));
        Ok(out)
        // `weights` 는 반환과 함께 놓입니다. 다음에 다른 모델이나 단계가 오면 언어 가중치가 내려갑니다.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::residency::{Action, Budget};
    use crate::ai::verify::det_normal;
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use candle_core::quantized::gguf_file::Value;
    use candle_core::quantized::GgmlDType;
    use candle_core::{DType, Device, Tensor};
    use std::sync::Mutex;

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec(det_normal(n, seed), shape.to_vec(), &Device::Cpu).unwrap()
    }

    fn gocr_file(dir: &TempDir) -> Arc<WeightFile> {
        let mut t: Vec<(String, Tensor, GgmlDType)> = vec![
            ("model.visual.blocks.0.attn.qkv.weight".into(), rand(&[96, 32], 1), GgmlDType::BF16),
            ("model.visual.merger.proj.weight".into(), rand(&[32, 32], 2), GgmlDType::BF16),
            ("token_embd.weight".into(), rand(&[64, 32], 3), GgmlDType::Q8_0),
            ("output.weight".into(), rand(&[64, 32], 4), GgmlDType::Q8_0),
            ("output_norm.weight".into(), rand(&[32], 5), GgmlDType::F32),
        ];
        for i in 0..2u64 {
            t.push((format!("blk.{i}.ffn_down.weight"), rand(&[32, 64], 10 + i), GgmlDType::Q8_0));
        }
        let path = dir.0.join("gocr.gguf");
        write_gguf(
            &path,
            &[("general.architecture", Value::String("gocr".into())), ("gocr.block_count", Value::U32(2))],
            &t,
        )
        .unwrap();
        Arc::new(WeightFile::open(&path).unwrap())
    }

    /// 콜백이 불릴 때의 상주 상태를 기록하는 대역
    struct Recorder {
        seen: Mutex<Vec<(String, Vec<Group>)>>,
    }

    impl VisionEncoder for Arc<Recorder> {
        type Input = u32;
        type Features = Vec<f32>;
        fn encode(&self, ctx: &PhaseCtx<'_>, vision: &Loaded, input: &u32) -> Result<Vec<f32>> {
            self.seen.lock().unwrap().push(("encode".into(), ctx.residency.resident_groups()));
            assert_eq!(vision.group(), Group::Vision);
            assert!(vision.has("model.visual.blocks.0.attn.qkv.weight"));
            Ok(vec![*input as f32; 4])
        }
    }

    impl LanguageDecoder for Arc<Recorder> {
        type Features = Vec<f32>;
        type Output = f32;
        fn decode(&self, ctx: &PhaseCtx<'_>, language: &LanguageWeights, f: Vec<f32>) -> Result<f32> {
            self.seen.lock().unwrap().push(("decode".into(), ctx.residency.resident_groups()));
            assert!(language.blocks.has("blk.1.ffn_down.weight"));
            assert!(language.output.has("output.weight") && language.output.has("output_norm.weight"));
            Ok(f.iter().sum())
        }
    }

    fn engine(dir: &TempDir) -> (OcrEngine<Arc<Recorder>, Arc<Recorder>>, Arc<Recorder>, SharedResidency) {
        let rec = Arc::new(Recorder { seen: Mutex::new(Vec::new()) });
        let res = Residency::shared(Device::Cpu, Budget::unlimited());
        let e = OcrEngine::from_file(gocr_file(dir), res.clone(), rec.clone(), rec.clone()).unwrap();
        (e, rec, res)
    }

    #[test]
    fn images_are_read_with_vision_only_and_decoded_with_language_only() -> Result<()> {
        let dir = TempDir::new("ocr-a");
        let (e, rec, res) = engine(&dir);
        let out = e.recognize(&[1, 2, 3])?;
        assert_eq!(out, vec![4.0, 8.0, 12.0], "특징이 단계 사이로 넘어가 순서대로 디코딩됩니다");

        let seen = rec.seen.lock().unwrap();
        let encodes: Vec<_> = seen.iter().filter(|(k, _)| k == "encode").collect();
        let decodes: Vec<_> = seen.iter().filter(|(k, _)| k == "decode").collect();
        assert_eq!((encodes.len(), decodes.len()), (3, 3));
        for (_, groups) in encodes {
            assert_eq!(*groups, vec![Group::Vision], "이미지를 읽는 동안 언어 가중치는 없어야 합니다");
        }
        for (_, groups) in decodes {
            assert_eq!(*groups, vec![Group::blocks(0..2), Group::Output], "디코딩 동안 비전은 내려가 있어야 합니다");
        }
        // 이미지가 여러 장이어도 비전은 한 번만 올라왔다 내려갔습니다.
        let r = res.lock().unwrap();
        let vision_loads = r.history().iter().filter(|t| t.action == Action::Load && t.group == Group::Vision).count();
        let vision_releases = r.history().iter().filter(|t| t.action == Action::Release && t.group == Group::Vision).count();
        assert_eq!((vision_loads, vision_releases), (1, 1));
        Ok(())
    }

    #[test]
    fn no_inputs_load_nothing() -> Result<()> {
        let dir = TempDir::new("ocr-b");
        let (e, _rec, res) = engine(&dir);
        assert!(e.recognize(&[])?.is_empty());
        assert!(res.lock().unwrap().history().is_empty());
        Ok(())
    }

    #[test]
    fn another_model_is_released_before_the_vision_tower_is_loaded() -> Result<()> {
        let dir = TempDir::new("ocr-c");
        let (e, _rec, res) = engine(&dir);
        // 다른 모델(zoro 모양)이 올라와 있는 상태에서 OCR 이 시작됩니다.
        let path = dir.0.join("zoro.gguf");
        write_gguf(
            &path,
            &[("general.architecture", Value::String("zoro".into()))],
            &[("blk.0.attn_norm.weight".into(), Tensor::zeros(32, DType::F32, &Device::Cpu)?, GgmlDType::F32)],
        )?;
        let zoro = ModelSpec::new(ModelId::Zoro, Arc::new(WeightFile::open(&path)?));
        drop(res.lock().unwrap().enter(&zoro, &Phase::new("text", [Group::blocks(0..1)]))?);
        assert_eq!(res.lock().unwrap().resident_model(), Some(ModelId::Zoro));

        e.recognize(&[7])?;
        let r = res.lock().unwrap();
        assert_eq!(r.resident_model(), Some(ModelId::Gocr));
        let log: Vec<(Action, ModelId)> = r.history().iter().map(|t| (t.action, t.model)).collect();
        let zoro_release = log.iter().position(|e| *e == (Action::Release, ModelId::Zoro)).expect("zoro 를 내려야 함");
        let first_gocr_load = log.iter().position(|e| *e == (Action::Load, ModelId::Gocr)).unwrap();
        assert!(zoro_release < first_gocr_load, "gocr 를 올리기 전에 zoro 를 내려야 합니다: {log:?}");
        Ok(())
    }

    #[test]
    fn a_failing_encoder_stops_before_any_language_weight_is_loaded() -> Result<()> {
        struct Broken;
        impl VisionEncoder for Broken {
            type Input = ();
            type Features = ();
            fn encode(&self, _: &PhaseCtx<'_>, _: &Loaded, _: &()) -> Result<()> {
                bail!("깨진 이미지")
            }
        }
        impl LanguageDecoder for Broken {
            type Features = ();
            type Output = ();
            fn decode(&self, _: &PhaseCtx<'_>, _: &LanguageWeights, _: ()) -> Result<()> {
                unreachable!("인코딩이 실패하면 디코딩 단계에 들어가면 안 됩니다")
            }
        }
        let dir = TempDir::new("ocr-d");
        let res = Residency::shared(Device::Cpu, Budget::unlimited());
        let e = OcrEngine::from_file(gocr_file(&dir), res.clone(), Broken, Broken)?;
        assert!(e.recognize(&[()]).is_err());
        let r = res.lock().unwrap();
        assert!(!r.resident_groups().iter().any(|g| matches!(g, Group::Blocks { .. })));
        // 실패해도 상주 관리자는 일관된 상태이고, 다음 호출에서 정리됩니다.
        assert_eq!(r.resident_groups(), vec![Group::Vision]);
        Ok(())
    }

    #[test]
    fn phases_name_exactly_the_groups_each_step_needs() {
        assert_eq!(vision_phase().groups, vec![Group::Vision]);
        assert_eq!(language_phase(16).groups, vec![Group::blocks(0..16), Group::Output]);
    }
}
