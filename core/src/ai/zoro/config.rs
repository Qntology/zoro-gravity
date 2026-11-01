//! zoro 텍스트 백본의 구조 값. GGUF 메타데이터(`zoro.*`)와 원본 config 전문(`zorogravity.source_config_json`)에서 읽습니다.

use anyhow::{bail, Context, Result};

use crate::ai::weights::WeightFile;

/// 층 하나가 쓰는 토큰 섞기 방식
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// 게이트 델타 규칙 선형 주의 (순환 상태가 있어 문맥 길이에 상관없이 층당 일정한 상태)
    LinearAttention,
    /// 출력 게이트가 있는 전체 주의
    FullAttention,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ZoroConfig {
    pub n_layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub vocab: usize,
    /// 전체 주의의 질의 헤드 수
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    /// 회전을 적용하는 앞쪽 차원 수 (head_dim × partial_rotary_factor)
    pub rope_dim: usize,
    pub rope_theta: f64,
    pub gdn_k_heads: usize,
    pub gdn_v_heads: usize,
    pub gdn_k_dim: usize,
    pub gdn_v_dim: usize,
    pub conv_kernel: usize,
    pub layer_kinds: Vec<LayerKind>,
}

impl ZoroConfig {
    pub fn from_file(file: &WeightFile) -> Result<Self> {
        let arch = file.meta_str("general.architecture")?;
        if arch != "zoro" {
            bail!("zoro GGUF 가 아닙니다: general.architecture={arch}");
        }
        let n = |key: &str| file.meta_usize(&format!("zoro.{key}"));
        let n_layers = n("block_count")?;
        let interval = n("full_attention_interval")?;
        let gdn_v_heads = n("ssm.time_step_rank")?;
        let inner = n("ssm.inner_size")?;
        if gdn_v_heads == 0 || !inner.is_multiple_of(gdn_v_heads) {
            bail!("ssm.inner_size {inner} 는 ssm.time_step_rank {gdn_v_heads} 로 나누어져야 합니다");
        }
        let cfg = Self {
            n_layers,
            hidden: n("embedding_length")?,
            ffn: n("feed_forward_length")?,
            vocab: n("vocab_size")?,
            n_heads: n("attention.head_count")?,
            n_kv_heads: n("attention.head_count_kv")?,
            head_dim: n("attention.key_length")?,
            rms_eps: file.meta_f64("zoro.attention.layer_norm_rms_epsilon")? as f32,
            rope_dim: n("rope.dimension_count")?,
            rope_theta: file.meta_f64("zoro.rope.freq_base")?,
            gdn_k_heads: n("ssm.group_count")?,
            gdn_v_heads,
            gdn_k_dim: n("ssm.state_size")?,
            gdn_v_dim: inner / gdn_v_heads,
            conv_kernel: n("ssm.conv_kernel")?,
            layer_kinds: layer_kinds(file, n_layers, interval)?,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.n_layers == 0 || self.layer_kinds.len() != self.n_layers {
            bail!("층 수 {} 와 층 종류 {}개가 맞지 않습니다", self.n_layers, self.layer_kinds.len());
        }
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            bail!("질의 헤드 {} 는 KV 헤드 {} 의 배수여야 합니다", self.n_heads, self.n_kv_heads);
        }
        if self.rope_dim == 0 || !self.rope_dim.is_multiple_of(2) || self.rope_dim > self.head_dim {
            bail!("회전 차원 {} 는 head_dim {} 이하의 짝수여야 합니다", self.rope_dim, self.head_dim);
        }
        if self.gdn_k_heads == 0 || !self.gdn_v_heads.is_multiple_of(self.gdn_k_heads) {
            bail!("선형 주의 V 헤드 {} 는 K 헤드 {} 의 배수여야 합니다", self.gdn_v_heads, self.gdn_k_heads);
        }
        if self.conv_kernel == 0 {
            bail!("컨볼루션 폭이 0 입니다");
        }
        Ok(())
    }

    /// 전체 주의 층에서 질의 투영의 출력 수 (질의 + 출력 게이트)
    pub fn q_proj_rows(&self) -> usize {
        self.n_heads * self.head_dim * 2
    }

    pub fn gdn_key_dim(&self) -> usize {
        self.gdn_k_heads * self.gdn_k_dim
    }

    pub fn gdn_value_dim(&self) -> usize {
        self.gdn_v_heads * self.gdn_v_dim
    }

    /// 선형 주의 입력 투영(q, k, v)의 출력 수
    pub fn gdn_conv_dim(&self) -> usize {
        2 * self.gdn_key_dim() + self.gdn_value_dim()
    }

    pub fn kind(&self, layer: usize) -> LayerKind {
        self.layer_kinds[layer]
    }
}

/// 원본 config 의 `layer_types` 가 있으면 그것을, 없으면 `full_attention_interval` 규칙 (매 interval 번째 층이 전체 주의)을 씁니다.
fn layer_kinds(file: &WeightFile, n_layers: usize, interval: usize) -> Result<Vec<LayerKind>> {
    if let Ok(text) = file.meta_str("zorogravity.source_config_json") {
        let json: serde_json::Value = serde_json::from_str(text).context("zorogravity.source_config_json 이 JSON 이 아닙니다")?;
        let types = json["text_config"]["layer_types"].as_array().or_else(|| json["layer_types"].as_array());
        if let Some(types) = types {
            if types.len() != n_layers {
                bail!("layer_types {}개와 층 수 {n_layers} 가 다릅니다", types.len());
            }
            return types
                .iter()
                .map(|t| match t.as_str() {
                    Some("linear_attention") => Ok(LayerKind::LinearAttention),
                    Some("full_attention") => Ok(LayerKind::FullAttention),
                    other => bail!("알 수 없는 층 종류: {other:?}"),
                })
                .collect();
        }
    }
    if interval == 0 {
        bail!("full_attention_interval 이 0 입니다");
    }
    Ok((0..n_layers)
        .map(|i| if (i + 1) % interval == 0 { LayerKind::FullAttention } else { LayerKind::LinearAttention })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use candle_core::quantized::gguf_file::Value;

    pub(crate) fn mini_meta(layer_types: Option<&str>) -> Vec<(&'static str, Value)> {
        let mut m = vec![
            ("general.architecture", Value::String("zoro".into())),
            ("zoro.block_count", Value::U32(4)),
            ("zoro.embedding_length", Value::U32(64)),
            ("zoro.feed_forward_length", Value::U32(128)),
            ("zoro.vocab_size", Value::U32(300)),
            ("zoro.attention.head_count", Value::U32(4)),
            ("zoro.attention.head_count_kv", Value::U32(2)),
            ("zoro.attention.key_length", Value::U32(32)),
            ("zoro.attention.layer_norm_rms_epsilon", Value::F32(1e-6)),
            ("zoro.rope.dimension_count", Value::U32(8)),
            ("zoro.rope.freq_base", Value::F32(10000.0)),
            ("zoro.full_attention_interval", Value::U32(4)),
            ("zoro.ssm.conv_kernel", Value::U32(4)),
            ("zoro.ssm.state_size", Value::U32(16)),
            ("zoro.ssm.group_count", Value::U32(2)),
            ("zoro.ssm.time_step_rank", Value::U32(4)),
            ("zoro.ssm.inner_size", Value::U32(64)),
        ];
        if let Some(t) = layer_types {
            m.push(("zorogravity.source_config_json", Value::String(format!(r#"{{"text_config":{{"layer_types":{t}}}}}"#))));
        }
        m
    }

    fn open(meta: &[(&str, Value)]) -> (TempDir, WeightFile) {
        let dir = TempDir::new("zcfg");
        let path = dir.0.join("m.gguf");
        let dummy = candle_core::Tensor::zeros(32, candle_core::DType::F32, &candle_core::Device::Cpu).unwrap();
        write_gguf(&path, meta, &[("output_norm.weight".into(), dummy, candle_core::quantized::GgmlDType::F32)]).unwrap();
        let f = WeightFile::open(&path).unwrap();
        (dir, f)
    }

    #[test]
    fn reads_the_structure_and_uses_the_interval_rule_without_layer_types() -> Result<()> {
        let (_d, f) = open(&mini_meta(None));
        let c = ZoroConfig::from_file(&f)?;
        assert_eq!((c.n_layers, c.hidden, c.ffn, c.vocab), (4, 64, 128, 300));
        assert_eq!((c.n_heads, c.n_kv_heads, c.head_dim, c.rope_dim), (4, 2, 32, 8));
        assert_eq!((c.gdn_k_heads, c.gdn_v_heads, c.gdn_k_dim, c.gdn_v_dim), (2, 4, 16, 16));
        assert_eq!(c.layer_kinds, vec![
            LayerKind::LinearAttention, LayerKind::LinearAttention, LayerKind::LinearAttention, LayerKind::FullAttention
        ]);
        assert_eq!((c.q_proj_rows(), c.gdn_key_dim(), c.gdn_value_dim(), c.gdn_conv_dim()), (256, 32, 64, 128));
        Ok(())
    }

    #[test]
    fn the_source_layer_types_win_over_the_interval_rule() -> Result<()> {
        let (_d, f) = open(&mini_meta(Some(r#"["full_attention","linear_attention","linear_attention","full_attention"]"#)));
        let c = ZoroConfig::from_file(&f)?;
        assert_eq!(c.kind(0), LayerKind::FullAttention);
        assert_eq!(c.kind(1), LayerKind::LinearAttention);
        let (_d, f) = open(&mini_meta(Some(r#"["full_attention"]"#)));
        assert!(ZoroConfig::from_file(&f).is_err(), "층 수와 길이가 다르면 거부");
        let (_d, f) = open(&mini_meta(Some(r#"["full_attention","x","x","x"]"#)));
        assert!(ZoroConfig::from_file(&f).is_err(), "알 수 없는 층 종류는 거부");
        Ok(())
    }

    #[test]
    fn rejects_other_architectures_and_inconsistent_values() {
        let mut m = mini_meta(None);
        m[0] = ("general.architecture", Value::String("gocr".into()));
        let (_d, f) = open(&m);
        assert!(ZoroConfig::from_file(&f).is_err());

        let mut m = mini_meta(None);
        m.retain(|(k, _)| *k != "zoro.attention.head_count_kv");
        m.push(("zoro.attention.head_count_kv", Value::U32(3)));
        let (_d, f) = open(&m);
        assert!(ZoroConfig::from_file(&f).is_err(), "4 헤드는 3 KV 헤드로 나눌 수 없음");

        let mut m = mini_meta(None);
        m.retain(|(k, _)| *k != "zoro.rope.dimension_count");
        m.push(("zoro.rope.dimension_count", Value::U32(7)));
        let (_d, f) = open(&m);
        assert!(ZoroConfig::from_file(&f).is_err(), "회전 차원은 짝수");
    }
}
