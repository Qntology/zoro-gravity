//! gocr(GravityOCR) 의 구조 값. GGUF 메타데이터(`gocr.*`)와 원본 config 전문(`zorogravity.source_config_json`)에서 읽습니다.

use anyhow::{anyhow, bail, Context, Result};

use crate::ai::weights::WeightFile;

#[derive(Debug, Clone, PartialEq)]
pub struct GocrConfig {
    // 언어 디코더
    pub n_layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    /// 시간 · 높이 · 너비 축에 나눠 주는 회전 주파수 수 (합은 head_dim / 2)
    pub rope_sections: [usize; 3],
    // 비전 인코더
    pub v_depth: usize,
    pub v_hidden: usize,
    pub v_ffn: usize,
    pub v_heads: usize,
    pub v_patch: usize,
    pub v_merge: usize,
    pub v_out: usize,
    pub v_rms_eps: f32,
    // 특수 토큰
    pub image_token: u32,
    pub image_start: u32,
    pub image_end: u32,
    pub eos: Vec<u32>,
}

fn json_usize(v: &serde_json::Value, key: &str) -> Result<usize> {
    v[key].as_u64().map(|x| x as usize).ok_or_else(|| anyhow!("config 에 {key} 가 없습니다"))
}

fn json_u32(v: &serde_json::Value, key: &str) -> Result<u32> {
    v[key].as_u64().map(|x| x as u32).ok_or_else(|| anyhow!("config 에 {key} 가 없습니다"))
}

impl GocrConfig {
    pub fn from_file(file: &WeightFile) -> Result<Self> {
        let arch = file.meta_str("general.architecture")?;
        if arch != "gocr" {
            bail!("gocr GGUF 가 아닙니다: general.architecture={arch}");
        }
        let json: serde_json::Value =
            serde_json::from_str(file.meta_str("zorogravity.source_config_json")?).context("zorogravity.source_config_json 이 JSON 이 아닙니다")?;
        let text = &json["text_config"];
        let vision = &json["vision_config"];
        let n = |key: &str| file.meta_usize(&format!("gocr.{key}"));
        let sections = text["rope_parameters"]["mrope_section"]
            .as_array()
            .filter(|a| a.len() == 3)
            .ok_or_else(|| anyhow!("config 에 mrope_section 세 값이 없습니다"))?;
        let section = |i: usize| sections[i].as_u64().map(|v| v as usize).ok_or_else(|| anyhow!("mrope_section 이 정수가 아닙니다"));
        let mut eos: Vec<u32> = text["eos_token_id"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
            .unwrap_or_default();
        if eos.is_empty() {
            eos.push(json_u32(&json, "eos_token_id")?);
        }
        let cfg = Self {
            n_layers: n("block_count")?,
            hidden: n("embedding_length")?,
            ffn: n("feed_forward_length")?,
            vocab: n("vocab_size")?,
            n_heads: n("attention.head_count")?,
            n_kv_heads: n("attention.head_count_kv")?,
            head_dim: n("attention.key_length")?,
            rms_eps: file.meta_f64("gocr.attention.layer_norm_rms_epsilon")? as f32,
            rope_theta: file.meta_f64("gocr.rope.freq_base")?,
            rope_sections: [section(0)?, section(1)?, section(2)?],
            v_depth: json_usize(vision, "depth")?,
            v_hidden: json_usize(vision, "hidden_size")?,
            v_ffn: json_usize(vision, "intermediate_size")?,
            v_heads: json_usize(vision, "num_heads")?,
            v_patch: json_usize(vision, "patch_size")?,
            v_merge: json_usize(vision, "spatial_merge_size")?,
            v_out: json_usize(vision, "out_hidden_size")?,
            v_rms_eps: vision["rms_norm_eps"].as_f64().ok_or_else(|| anyhow!("config 에 vision rms_norm_eps 가 없습니다"))? as f32,
            image_token: json_u32(&json, "image_token_id")?,
            image_start: json_u32(&json, "image_start_token_id")?,
            image_end: json_u32(&json, "image_end_token_id")?,
            eos,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            bail!("질의 헤드 {} 는 KV 헤드 {} 의 배수여야 합니다", self.n_heads, self.n_kv_heads);
        }
        if self.rope_sections.iter().sum::<usize>() * 2 != self.head_dim {
            bail!("회전 구간 {:?} 의 합의 두 배가 head_dim {} 이어야 합니다", self.rope_sections, self.head_dim);
        }
        if self.v_heads == 0 || !self.v_hidden.is_multiple_of(self.v_heads) {
            bail!("비전 차원 {} 는 헤드 {} 의 배수여야 합니다", self.v_hidden, self.v_heads);
        }
        if self.v_out != self.hidden {
            bail!("비전 출력 {} 와 언어 차원 {} 이 다릅니다", self.v_out, self.hidden);
        }
        Ok(())
    }

    pub fn v_head_dim(&self) -> usize {
        self.v_hidden / self.v_heads
    }

    /// 패치 하나의 값 수 (3 채널 × 패치 높이 × 패치 너비)
    pub fn patch_len(&self) -> usize {
        3 * self.v_patch * self.v_patch
    }
}

