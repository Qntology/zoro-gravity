//! siglip2 텍스트 타워와 토크나이저. 앵커 문구를 새로 인코딩할 때만 씁니다.
//!
//! 학습과 같이 항상 64 토큰까지 `<pad>` 로 채우고, 패딩 마스크 없이 전체를 읽은 뒤 마지막 위치를 풀링합니다
//! (모델의 `tokenizer_config.json` 이 `input_ids` 만 넘기도록 되어 있습니다).

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use tokenizers::Tokenizer;

use super::config::{Siglip2Config, TEXT_POS};
use super::tower::Stack;
use crate::ai::residency::{Loaded, TEXT_TOKEN_TABLE};
use crate::ai::weights::WeightFile;

const PAD: u32 = 0;

/// 소문자로 바꾸고, 글자 · 숫자 · 공백 밖의 문자는 공백으로 바꾼 뒤 공백을 하나로 줄입니다 (학습 때의 정규화).
pub fn canonicalize(text: &str) -> String {
    let kept: String = text
        .chars()
        .flat_map(|c| if c.is_alphanumeric() || c.is_whitespace() { c.to_lowercase().collect::<Vec<_>>() } else { vec![' '] })
        .collect();
    kept.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub struct TextTokenizer {
    inner: Tokenizer,
    max_len: usize,
}

impl TextTokenizer {
    pub fn from_file(path: &Path, max_len: usize) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| format!("토크나이저를 읽을 수 없습니다: {}", path.display()))?;
        Ok(Self { inner, max_len })
    }

    pub fn from_tokenizer(inner: Tokenizer, max_len: usize) -> Self {
        Self { inner, max_len }
    }

    /// 정확히 `max_len` 개의 토큰
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let e = self.inner.encode(canonicalize(text), true).map_err(|e| anyhow!("{e}"))?;
        let mut ids = e.get_ids().to_vec();
        ids.truncate(self.max_len);
        ids.resize(self.max_len, PAD);
        Ok(ids)
    }
}

/// 토큰열 묶음 → `[B, D]` (L2 정규화 전). 토큰 임베딩은 올리지 않고 쓰는 행만 파일에서 읽습니다.
pub fn encode(cfg: &Siglip2Config, w: &Loaded, file: &WeightFile, batch: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let (b, t, d) = (batch.len(), cfg.max_len, cfg.text.hidden);
    let flat: Vec<u32> = batch.iter().flatten().copied().collect();
    let x = file.gather_rows(TEXT_TOKEN_TABLE, &flat, device)?.reshape((b, t, d))?;
    let s = Stack { w, prefix: "text_model", cfg: &cfg.text };
    let pos = w.dense(TEXT_POS)?.to_dtype(DType::F32)?.unsqueeze(0)?;
    let x = s.encoder(x.broadcast_add(&pos)?)?;
    let x = s.norm("final_layer_norm", &x)?;
    let last = x.narrow(1, t - 1, 1)?.squeeze(1)?.contiguous()?;
    s.linear("head", &last)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_text_is_lowercase_without_punctuation() {
        assert_eq!(canonicalize("A Photo, of 2 CATS!"), "a photo of 2 cats");
        assert_eq!(canonicalize("  성명(자필)  서명 / 인 "), "성명 자필 서명 인");
        assert_eq!(canonicalize("계좌-번호"), "계좌 번호");
        assert_eq!(canonicalize("!!!"), "");
    }
}
