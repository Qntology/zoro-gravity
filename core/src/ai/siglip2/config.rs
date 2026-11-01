//! siglip2 구조 값. 층 수와 크기는 GGUF 메타데이터에서, 패치 크기 · 위치 격자 · 어휘 수는 텐서 모양에서 읽습니다.

use anyhow::{bail, Context, Result};

use crate::ai::residency::TEXT_TOKEN_TABLE;
use crate::ai::weights::WeightFile;

pub const PATCH_EMBED: &str = "vision_model.embeddings.patch_embedding.weight";
pub const VISION_POS: &str = "vision_model.embeddings.position_embedding.weight";
pub const TEXT_POS: &str = "text_model.embeddings.position_embedding.weight";
/// LayerNorm eps (원본 설정의 기본값)
pub const NORM_EPS: f64 = 1e-6;
/// 전처리 설정의 패치 상한. 위치 임베딩 표의 크기와 같습니다.
pub const DEFAULT_MAX_PATCHES: usize = 256;

#[derive(Debug, Clone, PartialEq)]
pub struct Tower {
    pub layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub heads: usize,
}

impl Tower {
    fn from_file(file: &WeightFile, name: &str) -> Result<Self> {
        let key = |k: &str| file.meta_usize(&format!("siglip2.{name}.{k}"));
        let tower = Self {
            layers: key("block_count")?,
            hidden: key("embedding_length")?,
            ffn: key("feed_forward_length")?,
            heads: key("attention.head_count")?,
        };
        if tower.heads == 0 || !tower.hidden.is_multiple_of(tower.heads) {
            bail!("{name} 타워: hidden {} 이 헤드 {} 로 나누어지지 않습니다", tower.hidden, tower.heads);
        }
        Ok(tower)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Siglip2Config {
    pub vision: Tower,
    pub text: Tower,
    pub patch: usize,
    /// 위치 임베딩 표의 한 변 (256 패치면 16)
    pub pos_side: usize,
    pub vocab: usize,
    /// 텍스트 길이 (학습과 같이 항상 이 길이까지 채웁니다)
    pub max_len: usize,
}

impl Siglip2Config {
    /// 모든 텐서가 한 파일에 있을 때
    pub fn from_file(file: &WeightFile) -> Result<Self> {
        Self::from_files(file, file)
    }

    /// `vision_file` 은 비전 타워 텐서가 있는 파일입니다 (`mmproj-siglip2-bf16.gguf`).
    pub fn from_files(file: &WeightFile, vision_file: &WeightFile) -> Result<Self> {
        let vision = Tower::from_file(file, "vision").context("siglip2 GGUF 가 아닙니다")?;
        let text = Tower::from_file(file, "text").context("siglip2 GGUF 가 아닙니다")?;

        let embed = vision_file.info(PATCH_EMBED)?;
        let [_, input] = embed.dims[..] else {
            bail!("패치 임베딩 모양이 예상과 다릅니다: {:?}", embed.dims);
        };
        let patch = (input / 3).isqrt();
        if 3 * patch * patch != input {
            bail!("패치 임베딩 입력 {input} 이 3 × p² 꼴이 아닙니다");
        }

        let pos = vision_file.info(VISION_POS)?;
        let pos_side = pos.dims[0].isqrt();
        if pos_side * pos_side != pos.dims[0] {
            bail!("위치 임베딩 {} 개가 정사각 격자가 아닙니다", pos.dims[0]);
        }

        let vocab = file.info(TEXT_TOKEN_TABLE)?.dims[0];
        let max_len = file.info(TEXT_POS)?.dims[0];
        Ok(Self { vision, text, patch, pos_side, vocab, max_len })
    }

    /// 패치 하나의 입력 길이 (`p × p × 3`)
    pub fn patch_len(&self) -> usize {
        self.patch * self.patch * 3
    }

    /// 코사인 비교에 쓰는 공유 공간의 차원
    pub fn dim(&self) -> usize {
        self.vision.hidden
    }
}
