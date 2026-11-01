//! zoro(Zoro-1) 백본: 텍스트 순전파를 직접 구현했습니다 (다른 구현을 가져오지 않았습니다).
//!
//! - `config`: GGUF 메타데이터에서 읽는 구조 값
//! - `ops`: 게이트 델타 규칙 선형 주의 (RMSNorm · RoPE · 인과 주의는 `ai::attention`)
//! - `model`: 층 순전파와 단계별 가중치 사용 (`ai::residency`)
//!
//! [`ZoroBackbone`] 이 `ai::pointer::Backbone` 을 구현하고, [`open_decision_model`] 이 `jev_head` 와 묶어 `DecisionModel` 을 만듭니다.
//! 글쓰기는 [`ZoroBackbone::write`] 입니다. 판단과 같은 몸통에 임베딩 표를 출력 투영으로 붙입니다 (명세의 듀얼 헤드).

pub mod config;
pub mod model;
pub mod ops;
#[cfg(test)]
pub(crate) mod reference;

use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};

use self::model::{rows_to_host, GenerateOptions, StreamMode, ZoroModel};
use super::gguf_model::load_active;
use super::jev_head::{JevHead, HIDDEN};
use super::pointer::{Backbone, PointerHidden, PointerModel, PointerPrompt};
use super::residency::SharedResidency;
use super::speculative::{BlockDiffusion, Speculation};
use super::tokenizer::{token_covering, ModelTokenizer};
use super::weights::WeightFile;
use crate::drafting::{through_last_sentence, TextWriter};

pub use config::{LayerKind, ZoroConfig};

/// 글쓰기의 추측을 `마스크:탐색:폭` 으로 정하는 환경 변수. `0:0:0` 이면 한 토큰씩 읽습니다.
pub const SPECULATION_ENV: &str = "ZOROGRAVITY_ZORO_SPECULATION";
/// 끝 토큰 (`<|im_end|>`, `<|endoftext|>`)
const STOP_TOKENS: [&str; 2] = ["<|im_end|>", "<|endoftext|>"];
/// 생각 구간을 비워 바로 답하게 하는 어시스턴트 머리말 (모델의 채팅 템플릿에서 `enable_thinking=false` 일 때의 형태)
const ANSWER_START: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// `마스크:탐색:폭` 형식의 추측 설정을 읽습니다.
pub fn parse_speculation(text: &str) -> Result<Speculation> {
    let mut parts = text.trim().split(':').map(str::parse::<usize>);
    let mut next = || parts.next().and_then(|v| v.ok());
    let parsed = (|| Some(Speculation { masks: next()?, lookup: next()?, width: next()? }))();
    parsed.with_context(|| format!("{SPECULATION_ENV}={text:?}: `마스크:탐색:폭` 형식의 정수여야 합니다"))
}

/// 모델이 쓴 글
#[derive(Debug, Clone, PartialEq)]
pub struct Written {
    pub text: String,
    pub tokens: usize,
    /// 끝 토큰까지 썼는가 (아니면 길이 제한이나 반복으로 멈춤)
    pub finished: bool,
    pub min_confidence: f32,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// 순전파 횟수
    pub reads: usize,
    /// 검증에 넘긴 후보 수와 그중 채택한 수
    pub drafted: usize,
    pub accepted: usize,
}

pub struct ZoroBackbone {
    model: ZoroModel,
    tokenizer: ModelTokenizer,
    /// 글쓰기에 쓰는 추측의 양
    speculation: Speculation,
}

impl ZoroBackbone {
    /// `model_dir/active.json` 이 가리키는 GGUF 와 `model_dir/tokenizer.json` 을 엽니다. 가중치는 여기서 올리지 않고,
    /// 첫 판단에서 필요한 묶음만 올라옵니다.
    pub fn open(model_dir: &Path, residency: SharedResidency) -> Result<Self> {
        Self::open_with(model_dir, residency, StreamMode::from_env()?)
    }

    pub fn open_with(model_dir: &Path, residency: SharedResidency, mode: StreamMode) -> Result<Self> {
        let active = load_active(model_dir)?;
        if active.architecture != "zoro" {
            bail!("zoro 모델 폴더가 아닙니다: {} ({})", model_dir.display(), active.architecture);
        }
        let file = Arc::new(WeightFile::open(&active.path)?);
        let tokenizer = ModelTokenizer::from_file(&model_dir.join("tokenizer.json"))?;
        // GGUF 에 없으면 모델 폴더의 설정을 씁니다 (예전에 변환한 파일).
        let diffusion = BlockDiffusion::resolve(&file, model_dir)?;
        let mut model = ZoroModel::new(file, residency, mode)?.with_diffusion(diffusion)?;
        if let Some(path) = &active.mmproj {
            // 매핑만 합니다. 비전 단계가 없으면 이 파일의 텐서는 읽지 않습니다.
            model = model.with_mmproj(Arc::new(WeightFile::open(path)?));
        }
        model.announce();
        let speculation = match std::env::var(SPECULATION_ENV) {
            Ok(v) => parse_speculation(&v)?,
            Err(_) => model.default_speculation(),
        };
        Ok(Self { model, tokenizer, speculation })
    }

    pub fn model(&self) -> &ZoroModel {
        &self.model
    }

    pub fn tokenizer(&self) -> &ModelTokenizer {
        &self.tokenizer
    }

    /// 지시문과 입력으로 글을 씁니다. 가장 확률이 높은 토큰만 고르므로 같은 입력이면 같은 글이 나옵니다.
    pub fn write(&self, system: &str, user: &str, max_tokens: usize) -> Result<Written> {
        self.write_with(system, user, max_tokens, self.speculation)
    }

    /// 추측의 양을 정해서 씁니다. 얼마이든 글은 같고 시간만 달라집니다.
    pub fn write_with(&self, system: &str, user: &str, max_tokens: usize, speculation: Speculation) -> Result<Written> {
        let prompt = format!("<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n{ANSWER_START}");
        let ids = self.tokenizer.encode(&prompt)?.ids;
        let stop = STOP_TOKENS.iter().filter_map(|t| self.tokenizer.token_to_id(t)).collect();
        let out = self.model.generate(&ids, &GenerateOptions::new(max_tokens, stop).with_speculation(speculation))?;
        Ok(Written {
            text: self.tokenizer.decode(&out.tokens)?,
            tokens: out.tokens.len(),
            finished: out.finished,
            min_confidence: out.confidences.iter().copied().fold(1.0, f32::min),
            prefill_ms: out.prefill_ms,
            decode_ms: out.decode_ms,
            reads: out.reads,
            drafted: out.drafted,
            accepted: out.accepted,
        })
    }
}

impl Backbone for ZoroBackbone {
    fn encode(&self, prompt: &PointerPrompt) -> Result<PointerHidden> {
        let enc = self.tokenizer.encode(&prompt.text)?;
        let query = token_covering(&enc.offsets, prompt.query_byte)
            .with_context(|| format!("질의 위치(바이트 {})를 덮는 토큰이 없습니다", prompt.query_byte))?;
        let mut rows = vec![query];
        for (i, &b) in prompt.option_bytes.iter().enumerate() {
            rows.push(
                token_covering(&enc.offsets, b)
                    .with_context(|| format!("선택지 {i} 의 위치(바이트 {b})를 덮는 토큰이 없습니다"))?,
            );
        }
        let hidden = self.model.hidden_states(&enc.ids)?;
        let picked = rows_to_host(&hidden, &rows)?;
        if picked.iter().any(|r| r.len() != HIDDEN) {
            bail!("은닉 차원이 {HIDDEN} 이 아닙니다: {}", picked[0].len());
        }
        let mut it = picked.into_iter();
        let query = it.next().expect("질의 행");
        Ok(PointerHidden { query, options: it.flatten().collect() })
    }
}

impl TextWriter for ZoroBackbone {
    /// 길이 제한이나 반복으로 멈춘 글은 마지막으로 완성된 문장까지만 넘깁니다.
    fn complete(&self, system: &str, user: &str, max_tokens: usize) -> Result<String> {
        let w = self.write(system, user, max_tokens)?;
        Ok(if w.finished { w.text } else { through_last_sentence(&w.text).to_string() })
    }
}

/// zoro 백본과 `jev_head` 로 판단 모델을 만듭니다. `residency` 는 gocr 등 다른 모델과 공유해야 "한 번에 한 모델"이 지켜집니다.
pub fn open_decision_model(model_dir: &Path, residency: SharedResidency) -> Result<PointerModel> {
    let backbone = ZoroBackbone::open(model_dir, residency)?;
    let head = JevHead::from_safetensors(&model_dir.join("jev_head.safetensors"))?;
    Ok(PointerModel::new(Box::new(backbone), head))
}
