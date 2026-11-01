//! zorogravity-gguf: safetensors 모델을 GGUF 로 바꾸고, 양자화 정밀도를 점검합니다.
//!
//! 사용법
//!   zorogravity-gguf convert <모델 폴더> <q4|q8> <출력.gguf>   변환만 합니다. 같은 폴더에 mmproj 파일도 씁니다.
//!   zorogravity-gguf verify  <모델 폴더> <파일.gguf> [보고서.json]   원본과 비교해 정밀도를 점검합니다. mmproj 도 함께 봅니다.
//!   zorogravity-gguf auto    <모델 폴더>   q4 와 q8 을 모두 만들고, q4 가 점검을 통과하면 q4, 아니면 q8 을 고릅니다.
//!
//! 규칙
//! - 선형 가중치: q4 는 Q4_K (열 수가 256 의 배수일 때), q8 은 Q8_0. 맞지 않으면 Q8_0 또는 BF16.
//! - 비전 타워(model.visual.*, vision_model.*): BF16 그대로. 본 파일에 넣지 않고 `mmproj-<모델>-bf16.gguf` 로 따로 씁니다.
//! - 정규화·게이트·컨볼루션·작은 행렬: F32.
//! - 텐서 이름은 llama.cpp 의 블록 이름 관례를 따릅니다 (blk.N.attn_q.weight, token_embd.weight 등). 구조 이름은 모델 이름(zoro, gocr)을 씁니다.
//!   siglip2 는 두 타워가 같은 층 이름을 쓰고 층 번호도 겹쳐서 원본 이름을 그대로 둡니다.
//! - 원본 config.json 전문을 메타데이터(zorogravity.source_config_json)에 넣어, 나중에 구조를 다시 읽을 수 있게 합니다.
//!
//! 정밀도 점검 (프록시): 양자화된 선형층마다
//! - 가중치 상대오차 ‖Ŵ−W‖/‖W‖
//! - 출력 상대오차 ‖XŴᵀ−XWᵀ‖/‖XWᵀ‖ (X 는 고정 시드의 가우시안 입력 4행)
//!
//! 기준: 출력 상대오차의 평균 5% 이하, 최대 10% 이하면 통과. 실제 디코딩 품질 점검은 모델 순전파가 생긴 뒤 따로 해야 합니다.

use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor};
use serde_json::{json, Value as Json};

/// 출력 상대오차 기준 (선험적으로 정함)
const ACT_MEAN_LIMIT: f32 = 0.05;
const ACT_MAX_LIMIT: f32 = 0.10;
/// 출력 점검에 쓰는 입력 행 수
const PROBE_ROWS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Quant {
    Q4,
    Q8,
}

impl Quant {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "q4" => Ok(Self::Q4),
            "q8" => Ok(Self::Q8),
            _ => bail!("양자화는 q4 또는 q8 이어야 합니다: {s}"),
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Self::Q4 => "q4",
            Self::Q8 => "q8",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Q4 => "Q4_K",
            Self::Q8 => "Q8_0",
        }
    }

    /// 선형 가중치의 GGUF 형식. 열 수가 블록 크기로 나누어지지 않으면 다음 후보로 넘어갑니다.
    fn linear_dtype(self, cols: usize) -> GgmlDType {
        match self {
            Self::Q4 if cols.is_multiple_of(256) => GgmlDType::Q4K,
            _ if cols.is_multiple_of(32) => GgmlDType::Q8_0,
            _ => GgmlDType::BF16,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arch {
    Zoro,
    Gocr,
    Siglip2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Keep {
    /// 양자화 대상 선형 가중치
    Linear,
    /// 정규화·게이트·컨볼루션 등 작은 텐서는 F32
    Small,
    /// 비전 타워는 BF16 그대로
    Vision,
}

impl Arch {
    fn name(self) -> &'static str {
        match self {
            Self::Zoro => "zoro",
            Self::Gocr => "gocr",
            Self::Siglip2 => "siglip2",
        }
    }

    /// 비전 타워를 따로 둔 파일 이름. 양자화 버전마다 같은 BF16 타워를 복사하지 않습니다.
    fn mmproj_file(self) -> String {
        format!("mmproj-{}-bf16.gguf", self.name())
    }
}

fn arch_of(config: &Json) -> Result<Arch> {
    match config["model_type"].as_str() {
        // config.json 의 model_type 은 가중치가 따르는 구조를 알려 줄 뿐입니다. 외부 모델을 쓴다는 뜻이 아닙니다.
        Some("qwen3_5") => Ok(Arch::Zoro),
        Some("glm_ocr") => Ok(Arch::Gocr),
        Some("siglip2") => Ok(Arch::Siglip2),
        other => bail!("지원하지 않는 model_type: {other:?}"),
    }
}

/// HF 텐서 이름 → (GGUF 이름, 보관 방식)
fn map_name(arch: Arch, hf: &str) -> Result<(String, Keep)> {
    if arch == Arch::Siglip2 {
        return Ok((hf.to_string(), siglip2_keep(hf)?));
    }
    if hf.starts_with("model.visual.") {
        return Ok((hf.to_string(), Keep::Vision));
    }
    let top = match hf {
        "model.language_model.embed_tokens.weight" => Some(("token_embd.weight", Keep::Linear)),
        "model.language_model.norm.weight" => Some(("output_norm.weight", Keep::Small)),
        "lm_head.weight" => Some(("output.weight", Keep::Linear)),
        _ => None,
    };
    if let Some((n, k)) = top {
        return Ok((n.to_string(), k));
    }
    if let Some(rest) = hf.strip_prefix("model.language_model.layers.") {
        let (idx, suffix) = rest
            .split_once('.')
            .ok_or_else(|| anyhow!("레이어 번호를 읽을 수 없습니다: {hf}"))?;
        let mapped = match arch {
            Arch::Zoro => zoro_layer(suffix),
            Arch::Gocr => gocr_layer(suffix),
            Arch::Siglip2 => None,
        };
        let (n, k) = mapped.ok_or_else(|| anyhow!("매핑되지 않은 텐서: {hf}"))?;
        return Ok((format!("blk.{idx}.{n}"), k));
    }
    bail!("매핑되지 않은 텐서: {hf}")
}

/// siglip2 텐서의 보관 방식. 비전 타워는 BF16, 텍스트 타워는 행렬만 양자화하고 편향 · 정규화 · 위치 임베딩은 F32 입니다.
fn siglip2_keep(hf: &str) -> Result<Keep> {
    if hf.starts_with("vision_model.") {
        return Ok(Keep::Vision);
    }
    if matches!(hf, "logit_scale" | "logit_bias") {
        return Ok(Keep::Small);
    }
    let Some(rest) = hf.strip_prefix("text_model.") else {
        bail!("매핑되지 않은 텐서: {hf}");
    };
    Ok(if rest.ends_with(".bias") || rest.contains("layer_norm") || rest == "embeddings.position_embedding.weight" {
        Keep::Small
    } else {
        Keep::Linear
    })
}

fn zoro_layer(s: &str) -> Option<(&'static str, Keep)> {
    use Keep::*;
    Some(match s {
        "input_layernorm.weight" => ("attn_norm.weight", Small),
        "post_attention_layernorm.weight" => ("post_attention_norm.weight", Small),
        "linear_attn.in_proj_qkv.weight" => ("attn_qkv.weight", Linear),
        "linear_attn.in_proj_z.weight" => ("attn_gate.weight", Linear),
        "linear_attn.in_proj_b.weight" => ("ssm_beta.weight", Small),
        "linear_attn.in_proj_a.weight" => ("ssm_alpha.weight", Small),
        "linear_attn.out_proj.weight" => ("ssm_out.weight", Linear),
        "linear_attn.conv1d.weight" => ("ssm_conv1d.weight", Small),
        "linear_attn.dt_bias" => ("ssm_dt.bias", Small),
        "linear_attn.A_log" => ("ssm_a", Small),
        "linear_attn.norm.weight" => ("ssm_norm.weight", Small),
        "self_attn.q_proj.weight" => ("attn_q.weight", Linear),
        "self_attn.k_proj.weight" => ("attn_k.weight", Linear),
        "self_attn.v_proj.weight" => ("attn_v.weight", Linear),
        "self_attn.o_proj.weight" => ("attn_output.weight", Linear),
        "self_attn.q_norm.weight" => ("attn_q_norm.weight", Small),
        "self_attn.k_norm.weight" => ("attn_k_norm.weight", Small),
        "mlp.gate_proj.weight" => ("ffn_gate.weight", Linear),
        "mlp.up_proj.weight" => ("ffn_up.weight", Linear),
        "mlp.down_proj.weight" => ("ffn_down.weight", Linear),
        _ => return None,
    })
}

fn gocr_layer(s: &str) -> Option<(&'static str, Keep)> {
    use Keep::*;
    Some(match s {
        "input_layernorm.weight" => ("attn_norm.weight", Small),
        "self_attn.q_proj.weight" => ("attn_q.weight", Linear),
        "self_attn.k_proj.weight" => ("attn_k.weight", Linear),
        "self_attn.v_proj.weight" => ("attn_v.weight", Linear),
        "self_attn.o_proj.weight" => ("attn_output.weight", Linear),
        "post_self_attn_layernorm.weight" => ("post_attn_norm.weight", Small),
        "post_attention_layernorm.weight" => ("ffn_norm.weight", Small),
        "mlp.gate_up_proj.weight" => ("ffn_gate_up.weight", Linear),
        "mlp.down_proj.weight" => ("ffn_down.weight", Linear),
        "post_mlp_layernorm.weight" => ("post_ffn_norm.weight", Small),
        _ => return None,
    })
}

/// 텐서 하나를 어떤 형식으로 저장할지 정합니다. (GGUF 형식, 저장할 텐서)
fn plan(keep: Keep, quant: Quant, t: &Tensor) -> Result<(GgmlDType, Tensor)> {
    match keep {
        Keep::Vision => Ok((GgmlDType::BF16, t.clone())),
        Keep::Small => Ok((GgmlDType::F32, t.to_dtype(DType::F32)?)),
        Keep::Linear => {
            let dims = t.dims();
            if dims.len() != 2 || dims[0] < 64 {
                // 1차원이거나 행이 아주 적은 행렬(게이트 등)은 정밀도를 지킵니다
                return Ok((GgmlDType::F32, t.to_dtype(DType::F32)?));
            }
            Ok((quant.linear_dtype(dims[1]), t.clone()))
        }
    }
}

/// 샤드 파일 목록. index.json 이 있으면 거기서, 없으면 폴더의 *.safetensors 를 씁니다.
fn shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let idx: Json = serde_json::from_str(&fs::read_to_string(&index)?)?;
        let map = idx["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("index.json 에 weight_map 이 없습니다"))?;
        let mut files: Vec<String> = map.values().filter_map(|v| v.as_str().map(str::to_string)).collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
        .collect();
    files.sort();
    if files.is_empty() {
        bail!("safetensors 파일이 없습니다: {}", dir.display());
    }
    Ok(files)
}

fn read_config(dir: &Path) -> Result<Json> {
    let text = fs::read_to_string(dir.join("config.json")).context("config.json 을 읽을 수 없습니다")?;
    Ok(serde_json::from_str(&text)?)
}

fn cfg_u32(obj: &Json, key: &str) -> Result<u32> {
    obj[key]
        .as_u64()
        .map(|v| v as u32)
        .ok_or_else(|| anyhow!("config 에 {key} 가 없습니다"))
}

fn cfg_f32(obj: &Json, key: &str) -> Result<f32> {
    obj[key]
        .as_f64()
        .map(|v| v as f32)
        .ok_or_else(|| anyhow!("config 에 {key} 가 없습니다"))
}

fn i32_array(v: &Json) -> gguf_file::Value {
    let items = v
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_i64()).map(|x| gguf_file::Value::I32(x as i32)).collect())
        .unwrap_or_default();
    gguf_file::Value::Array(items)
}

/// GGUF 메타데이터. 구조 값은 config 에서 옮기고, 원본 config 전문도 함께 넣습니다.
fn metadata(arch: Arch, config: &Json, quant: Quant) -> Result<Vec<(String, gguf_file::Value)>> {
    use gguf_file::Value as V;
    if arch == Arch::Siglip2 {
        return siglip2_metadata(config, quant);
    }
    let t = if config["text_config"].is_object() { &config["text_config"] } else { config };
    let rp = &t["rope_parameters"];
    let name = arch.name();
    // 원본 config 가 선언한 구조 이름은 출처로만 기록합니다.
    let source = config["architectures"][0].as_str().unwrap_or("unknown");
    let mut m: Vec<(String, V)> = Vec::new();
    m.push(("general.architecture".into(), V::String(name.into())));
    m.push(("general.name".into(), V::String(format!("zorogravity-{name}-{}", quant.tag()))));
    m.push((format!("{name}.block_count"), V::U32(cfg_u32(t, "num_hidden_layers")?)));
    m.push((format!("{name}.embedding_length"), V::U32(cfg_u32(t, "hidden_size")?)));
    m.push((format!("{name}.feed_forward_length"), V::U32(cfg_u32(t, "intermediate_size")?)));
    m.push((format!("{name}.context_length"), V::U32(cfg_u32(t, "max_position_embeddings")?)));
    m.push((format!("{name}.vocab_size"), V::U32(cfg_u32(t, "vocab_size")?)));
    m.push((format!("{name}.attention.head_count"), V::U32(cfg_u32(t, "num_attention_heads")?)));
    m.push((format!("{name}.attention.head_count_kv"), V::U32(cfg_u32(t, "num_key_value_heads")?)));
    let head_dim = cfg_u32(t, "head_dim")?;
    m.push((format!("{name}.attention.key_length"), V::U32(head_dim)));
    m.push((format!("{name}.attention.layer_norm_rms_epsilon"), V::F32(cfg_f32(t, "rms_norm_eps")?)));
    let partial = rp["partial_rotary_factor"].as_f64().unwrap_or(1.0);
    m.push((format!("{name}.rope.dimension_count"), V::U32((head_dim as f64 * partial).round() as u32)));
    m.push((format!("{name}.rope.freq_base"), V::F32(cfg_f32(rp, "rope_theta")?)));
    m.push((format!("{name}.rope.dimension_sections"), i32_array(&rp["mrope_section"])));
    if arch == Arch::Zoro {
        m.push((format!("{name}.full_attention_interval"), V::U32(cfg_u32(t, "full_attention_interval")?)));
        m.push((format!("{name}.ssm.conv_kernel"), V::U32(cfg_u32(t, "linear_conv_kernel_dim")?)));
        m.push((format!("{name}.ssm.state_size"), V::U32(cfg_u32(t, "linear_key_head_dim")?)));
        m.push((format!("{name}.ssm.group_count"), V::U32(cfg_u32(t, "linear_num_key_heads")?)));
        m.push((format!("{name}.ssm.time_step_rank"), V::U32(cfg_u32(t, "linear_num_value_heads")?)));
        let inner = cfg_u32(t, "linear_value_head_dim")? * cfg_u32(t, "linear_num_value_heads")?;
        m.push((format!("{name}.ssm.inner_size"), V::U32(inner)));
    }
    m.push(("zorogravity.quant".into(), V::String(quant.label().into())));
    m.push(("zorogravity.mmproj".into(), V::String(arch.mmproj_file())));
    m.push(("zorogravity.source_architecture".into(), V::String(source.into())));
    m.push(("zorogravity.source_config_json".into(), V::String(serde_json::to_string(config)?)));
    Ok(m)
}

/// 블록 디퓨전 설정을 담는 메타데이터. 이름은 `zorogravity_core::ai::speculative` 와 같아야 합니다.
const META_BLOCK_SIZE: &str = "zorogravity.block_diffusion.size";
const META_MASK_ID: &str = "zorogravity.block_diffusion.mask_id";

/// `block_diffusion.json` 을 텍스트 GGUF 로 옮깁니다. 초안은 언어 가중치만 쓰므로 파일 하나로 쓸 수 있고, mmproj 와는 관계가 없습니다.
fn block_diffusion_metadata(dir: &Path) -> Result<Vec<(String, gguf_file::Value)>> {
    let path = dir.join("block_diffusion.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let json: Json = serde_json::from_str(&fs::read_to_string(&path)?).with_context(|| format!("{} 를 읽을 수 없습니다", path.display()))?;
    Ok(vec![
        (META_BLOCK_SIZE.into(), gguf_file::Value::U32(cfg_u32(&json, "bd_size")?)),
        (META_MASK_ID.into(), gguf_file::Value::U32(cfg_u32(&json, "mask_id")?)),
    ])
}

/// 마스크 토큰이 임베딩 표 안에 있는가. 설정이 없는 모델은 볼 것이 없습니다.
fn mask_is_in_the_table(content: &gguf_file::Content) -> bool {
    let Some(gguf_file::Value::U32(mask)) = content.metadata.get(META_MASK_ID) else {
        return !content.metadata.contains_key(META_BLOCK_SIZE);
    };
    content.tensor_infos.get("token_embd.weight").is_some_and(|t| (*mask as usize) < t.shape.dims()[0])
}

/// mmproj 파일의 메타데이터. 구조 이름은 본 모델과 같고 `general.type` 으로 구별합니다.
/// 텐서 이름은 원본(`model.visual.*`)을 그대로 두므로 llama.cpp 의 `clip` 형식과는 다릅니다.
fn mmproj_metadata(arch: Arch, config: &Json) -> Result<Vec<(String, gguf_file::Value)>> {
    use gguf_file::Value as V;
    let name = arch.name();
    let v = &config["vision_config"];
    // 층 · 헤드 수의 이름은 모델마다 다르고, siglip2 는 패치 크기 등을 config 에 적지 않습니다.
    let depth = cfg_u32(v, "depth").or_else(|_| cfg_u32(v, "num_hidden_layers"))?;
    let heads = cfg_u32(v, "num_heads").or_else(|_| cfg_u32(v, "num_attention_heads"))?;
    let mut m: Vec<(String, V)> = vec![
        ("general.architecture".into(), V::String(name.into())),
        ("general.type".into(), V::String("mmproj".into())),
        ("general.name".into(), V::String(format!("zorogravity-{name}-mmproj-bf16"))),
        (format!("{name}.vision.block_count"), V::U32(depth)),
        (format!("{name}.vision.embedding_length"), V::U32(cfg_u32(v, "hidden_size")?)),
        (format!("{name}.vision.feed_forward_length"), V::U32(cfg_u32(v, "intermediate_size")?)),
        (format!("{name}.vision.attention.head_count"), V::U32(heads)),
    ];
    for (key, field) in [
        ("patch_size", "patch_size"),
        ("spatial_merge_size", "spatial_merge_size"),
        ("temporal_patch_size", "temporal_patch_size"),
        ("projection_dim", "out_hidden_size"),
    ] {
        if let Ok(value) = cfg_u32(v, field) {
            m.push((format!("{name}.vision.{key}"), V::U32(value)));
        }
    }
    m.push(("zorogravity.vision_dtype".into(), V::String("bf16".into())));
    m.push(("zorogravity.source_architecture".into(), V::String(config["architectures"][0].as_str().unwrap_or("unknown").into())));
    m.push(("zorogravity.source_config_json".into(), V::String(serde_json::to_string(config)?)));
    Ok(m)
}

/// siglip2 는 비전 · 텍스트 타워의 층 수와 크기를 따로 적습니다. 패치 크기 · 위치 격자 · 어휘 수는 텐서 모양에서 읽습니다.
fn siglip2_metadata(config: &Json, quant: Quant) -> Result<Vec<(String, gguf_file::Value)>> {
    use gguf_file::Value as V;
    let mut m: Vec<(String, V)> = vec![
        ("general.architecture".into(), V::String("siglip2".into())),
        ("general.name".into(), V::String(format!("zorogravity-siglip2-{}", quant.tag()))),
    ];
    for (tower, c) in [("vision", &config["vision_config"]), ("text", &config["text_config"])] {
        m.push((format!("siglip2.{tower}.block_count"), V::U32(cfg_u32(c, "num_hidden_layers")?)));
        m.push((format!("siglip2.{tower}.embedding_length"), V::U32(cfg_u32(c, "hidden_size")?)));
        m.push((format!("siglip2.{tower}.feed_forward_length"), V::U32(cfg_u32(c, "intermediate_size")?)));
        m.push((format!("siglip2.{tower}.attention.head_count"), V::U32(cfg_u32(c, "num_attention_heads")?)));
    }
    let t = &config["text_config"];
    // 공용 도구(`load_active`)가 읽는 값입니다.
    m.push(("siglip2.block_count".into(), V::U32(cfg_u32(&config["vision_config"], "num_hidden_layers")?)));
    m.push(("siglip2.embedding_length".into(), V::U32(cfg_u32(t, "hidden_size")?)));
    m.push(("siglip2.vocab_size".into(), V::U32(cfg_u32(t, "vocab_size")?)));
    m.push(("zorogravity.quant".into(), V::String(quant.label().into())));
    m.push(("zorogravity.mmproj".into(), V::String(Arch::Siglip2.mmproj_file())));
    m.push(("zorogravity.source_architecture".into(), V::String(config["architectures"][0].as_str().unwrap_or("unknown").into())));
    m.push(("zorogravity.source_config_json".into(), V::String(serde_json::to_string(config)?)));
    Ok(m)
}

struct ConvertSummary {
    tensors: usize,
    bytes: u64,
    by_type: BTreeMap<String, usize>,
    /// (파일, 텐서 수, 바이트)
    mmproj: (PathBuf, usize, u64),
}

impl ConvertSummary {
    fn print(&self) {
        println!("   텐서 {}개 · {:.2} GB · 형식별 {:?}", self.tensors, self.bytes as f64 / 1e9, self.by_type);
        let (path, tensors, bytes) = &self.mmproj;
        println!("   비전 {}: 텐서 {tensors}개 · {:.2} GB", path.display(), *bytes as f64 / 1e9);
    }
}

fn write_file(out: &Path, meta: &[(String, gguf_file::Value)], tensors: &[(String, QTensor)]) -> Result<u64> {
    let meta_refs: Vec<(&str, &gguf_file::Value)> = meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensor_refs: Vec<(&str, &QTensor)> = tensors.iter().map(|(n, q)| (n.as_str(), q)).collect();
    let mut file = File::create(out).with_context(|| format!("출력 파일을 만들 수 없습니다: {}", out.display()))?;
    gguf_file::write(&mut file, &meta_refs, &tensor_refs)?;
    drop(file);
    Ok(fs::metadata(out)?.len())
}

fn convert(dir: &Path, quant: Quant, out: &Path) -> Result<ConvertSummary> {
    let config = read_config(dir)?;
    let arch = arch_of(&config)?;
    let mut tensors: Vec<(String, QTensor)> = Vec::new();
    let mut vision: Vec<(String, QTensor)> = Vec::new();
    let mut by_type: BTreeMap<String, usize> = BTreeMap::new();
    let mut reshaped: Vec<String> = Vec::new();
    for shard in shard_files(dir)? {
        let loaded = candle_core::safetensors::load(&shard, &Device::Cpu)
            .with_context(|| format!("샤드를 읽을 수 없습니다: {}", shard.display()))?;
        let mut names: Vec<&String> = loaded.keys().collect();
        names.sort();
        for name in names {
            let (gguf_name, keep) = map_name(arch, name)?;
            let (dtype, mut t) = plan(keep, quant, &loaded[name])?;
            if t.rank() > 4 {
                // GGUF 는 차원 4개까지입니다. 시간축과 채널을 한 축으로 합치면 메모리 배치가 그대로입니다 (예: 3D 패치 임베딩).
                let orig: Vec<String> = t.dims().iter().map(|d| d.to_string()).collect();
                reshaped.push(format!("{name}:{}", orig.join("x")));
                while t.rank() > 4 {
                    let mut dims = t.dims().to_vec();
                    let merged = dims.remove(1) * dims.remove(1);
                    dims.insert(1, merged);
                    t = t.reshape(dims)?;
                }
            }
            let q = QTensor::quantize(&t, dtype).with_context(|| format!("양자화 실패: {name}"))?;
            if keep == Keep::Vision {
                vision.push((gguf_name, q));
            } else {
                *by_type.entry(format!("{dtype:?}")).or_default() += 1;
                tensors.push((gguf_name, q));
            }
        }
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));
    vision.sort_by(|a, b| a.0.cmp(&b.0));
    let mut meta = metadata(arch, &config, quant)?;
    meta.extend(block_diffusion_metadata(dir)?);
    // 차원을 합친 텐서는 비전 타워뿐입니다.
    let mut vision_meta = mmproj_metadata(arch, &config)?;
    if !reshaped.is_empty() {
        let items = reshaped.into_iter().map(gguf_file::Value::String).collect();
        vision_meta.push(("zorogravity.reshaped_tensors".into(), gguf_file::Value::Array(items)));
    }
    let bytes = write_file(out, &meta, &tensors)?;
    let path = out.with_file_name(arch.mmproj_file());
    let size = write_file(&path, &vision_meta, &vision)?;
    Ok(ConvertSummary { tensors: tensors.len(), bytes, by_type, mmproj: (path, vision.len(), size) })
}

#[derive(Debug, Clone)]
struct LinearCheck {
    name: String,
    dtype: String,
    weight_err: f32,
    /// 토큰 임베딩은 행 조회라서 출력 상대오차를 재지 않습니다
    act_err: Option<f32>,
}

struct VerifySummary {
    checks: Vec<LinearCheck>,
    gguf_tensors: usize,
    expected_tensors: usize,
    missing: Vec<String>,
    unmapped: Vec<String>,
    /// mmproj 에서 원본과 비교한 비전 텐서 수와 값이 다른 텐서 수
    vision_checked: usize,
    vision_changed: usize,
    /// 마스크 토큰이 임베딩 표 안에 있는가
    mask_ok: bool,
    pass: bool,
    max_act: f32,
    mean_act: f32,
    max_weight: f32,
    mean_weight: f32,
}

/// `a` 와 기준 `b` 의 상대오차 ‖a−b‖/‖b‖
fn rel(a: &Tensor, b: &Tensor) -> Result<f32> {
    let d = (a - b)?.sqr()?.sum_all()?.to_scalar::<f32>()?;
    let n = b.sqr()?.sum_all()?.to_scalar::<f32>()?;
    Ok((d / n.max(1e-30)).sqrt())
}

/// 고정 시드의 가우시안 입력 (xorshift + Box–Muller)
fn probe(rows: usize, cols: usize, seed: u64) -> Result<Tensor> {
    let mut s = seed | 1;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let v: Vec<f32> = (0..rows * cols)
        .map(|_| {
            let (u1, u2) = (next(), next());
            ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
        })
        .collect();
    Ok(Tensor::from_vec(v, (rows, cols), &Device::Cpu)?)
}

/// 이름마다 다른 시드 (FNV-1a)
fn seed_of(name: &str) -> u64 {
    name.bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3))
}

fn verify(dir: &Path, gguf: &Path) -> Result<VerifySummary> {
    let config = read_config(dir)?;
    let arch = arch_of(&config)?;
    let mut reader = BufReader::new(File::open(gguf).with_context(|| format!("GGUF 를 열 수 없습니다: {}", gguf.display()))?);
    let content = gguf_file::Content::read(&mut reader)?;
    let dev = Device::Cpu;
    let mmproj_path = gguf.with_file_name(arch.mmproj_file());
    let mut mmproj_reader = BufReader::new(File::open(&mmproj_path).with_context(|| format!("mmproj 를 열 수 없습니다: {}", mmproj_path.display()))?);
    let mmproj = gguf_file::Content::read(&mut mmproj_reader)?;

    let mut expected: Vec<String> = Vec::new();
    let mut unmapped: Vec<String> = Vec::new();
    let mut checks: Vec<LinearCheck> = Vec::new();
    let (mut vision_checked, mut vision_changed) = (0, 0);
    for shard in shard_files(dir)? {
        let loaded = candle_core::safetensors::load(&shard, &dev)?;
        let mut names: Vec<&String> = loaded.keys().collect();
        names.sort();
        for name in names {
            let (gguf_name, keep) = match map_name(arch, name) {
                Ok(v) => v,
                Err(_) => {
                    unmapped.push(name.clone());
                    continue;
                }
            };
            expected.push(gguf_name.clone());
            if keep == Keep::Vision {
                // 비전 타워는 BF16 으로 옮기기만 합니다. 원본이 F32 면 BF16 반올림까지는 같아야 합니다.
                if mmproj.tensor_infos.contains_key(&gguf_name) {
                    let stored = mmproj.tensor(&mut mmproj_reader, &gguf_name, &dev)?.dequantize(&dev)?.flatten_all()?;
                    let source = loaded[name].to_dtype(DType::BF16)?.to_dtype(DType::F32)?.flatten_all()?;
                    vision_checked += 1;
                    vision_changed += (rel(&stored, &source)? != 0.0) as usize;
                }
                continue;
            }
            let Some(info) = content.tensor_infos.get(&gguf_name) else {
                continue;
            };
            let quantized = matches!(info.ggml_dtype, GgmlDType::Q4K | GgmlDType::Q8_0);
            if keep != Keep::Linear || !quantized {
                continue;
            }
            let w = loaded[name].to_dtype(DType::F32)?;
            let q = content.tensor(&mut reader, &gguf_name, &dev)?;
            let w_hat = q.dequantize(&dev)?;
            let weight_err = rel(&w_hat, &w)?;
            let act_err = if matches!(gguf_name.as_str(), "token_embd.weight" | "text_model.embeddings.token_embedding.weight") {
                None
            } else {
                let x = probe(PROBE_ROWS, w.dim(1)?, seed_of(&gguf_name))?;
                let y = x.matmul(&w.t()?)?;
                let yh = x.matmul(&w_hat.t()?)?;
                Some(rel(&yh, &y)?)
            };
            checks.push(LinearCheck {
                name: gguf_name,
                dtype: format!("{:?}", info.ggml_dtype),
                weight_err,
                act_err,
            });
        }
    }

    let missing: Vec<String> = expected
        .iter()
        .filter(|n| !content.tensor_infos.contains_key(*n) && !mmproj.tensor_infos.contains_key(*n))
        .cloned()
        .collect();
    let gguf_tensors = content.tensor_infos.len() + mmproj.tensor_infos.len();
    let mask_ok = mask_is_in_the_table(&content);
    let acts: Vec<f32> = checks.iter().filter_map(|c| c.act_err).collect();
    let max_act = acts.iter().cloned().fold(0f32, f32::max);
    let mean_act = if acts.is_empty() { 0.0 } else { acts.iter().sum::<f32>() / acts.len() as f32 };
    let max_weight = checks.iter().map(|c| c.weight_err).fold(0f32, f32::max);
    let mean_weight = if checks.is_empty() {
        0.0
    } else {
        checks.iter().map(|c| c.weight_err).sum::<f32>() / checks.len() as f32
    };
    let pass = missing.is_empty()
        && unmapped.is_empty()
        && expected.len() == gguf_tensors
        && vision_changed == 0
        && mask_ok
        && mean_act <= ACT_MEAN_LIMIT
        && max_act <= ACT_MAX_LIMIT;
    Ok(VerifySummary {
        checks,
        gguf_tensors,
        expected_tensors: expected.len(),
        missing,
        unmapped,
        vision_checked,
        vision_changed,
        mask_ok,
        pass,
        max_act,
        mean_act,
        max_weight,
        mean_weight,
    })
}

fn print_verify(quant_tag: &str, v: &VerifySummary) {
    println!(
        "[{quant_tag}] 텐서 {}개 (원본 {}개) · 누락 {} · 매핑 실패 {}",
        v.gguf_tensors,
        v.expected_tensors,
        v.missing.len(),
        v.unmapped.len()
    );
    if v.vision_checked > 0 {
        println!("[{quant_tag}] mmproj 비전 텐서 {}개 · BF16 원본과 값이 다른 텐서 {}개", v.vision_checked, v.vision_changed);
    }
    println!(
        "[{quant_tag}] 선형층 {}개 점검 · 출력 상대오차 평균 {:.3}% · 최대 {:.3}% · 가중치 상대오차 평균 {:.3}% · 최대 {:.3}%",
        v.checks.len(),
        v.mean_act * 100.0,
        v.max_act * 100.0,
        v.mean_weight * 100.0,
        v.max_weight * 100.0
    );
    let mut worst: Vec<&LinearCheck> = v.checks.iter().filter(|c| c.act_err.is_some()).collect();
    worst.sort_by(|a, b| b.act_err.unwrap_or(0.0).total_cmp(&a.act_err.unwrap_or(0.0)));
    for c in worst.iter().take(3) {
        println!(
            "    최악: {:<40} {:<5} 출력 {:.3}% 가중치 {:.3}%",
            c.name,
            c.dtype,
            c.act_err.unwrap_or(0.0) * 100.0,
            c.weight_err * 100.0
        );
    }
    println!(
        "[{quant_tag}] 판정: {} (기준: 출력 평균 ≤ {:.0}%, 최대 ≤ {:.0}%)",
        if v.pass { "통과" } else { "실패" },
        ACT_MEAN_LIMIT * 100.0,
        ACT_MAX_LIMIT * 100.0
    );
}

fn write_report(path: &Path, quant: Quant, v: &VerifySummary) -> Result<()> {
    let rows: Vec<Json> = v
        .checks
        .iter()
        .map(|c| {
            json!({
                "tensor": c.name,
                "dtype": c.dtype,
                "weight_rel_err": c.weight_err,
                "output_rel_err": c.act_err,
            })
        })
        .collect();
    let report = json!({
        "quant": quant.label(),
        "pass": v.pass,
        "limits": { "output_mean": ACT_MEAN_LIMIT, "output_max": ACT_MAX_LIMIT },
        "output_rel_err": { "mean": v.mean_act, "max": v.max_act },
        "weight_rel_err": { "mean": v.mean_weight, "max": v.max_weight },
        "gguf_tensors": v.gguf_tensors,
        "expected_tensors": v.expected_tensors,
        "missing": v.missing,
        "unmapped": v.unmapped,
        "mmproj_tensors": v.vision_checked,
        "mmproj_changed": v.vision_changed,
        "mask_token_in_table": v.mask_ok,
        "linear_checks": rows,
    });
    fs::write(path, serde_json::to_string_pretty(&report)?)?;
    Ok(())
}

fn auto(dir: &Path) -> Result<()> {
    let name = dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .ok_or_else(|| anyhow!("모델 폴더 이름을 알 수 없습니다"))?;
    // siglip2 는 hidden 1152 가 Q4_K 블록(256)의 배수가 아니라서 q4 가 q8 과 같은 파일이 됩니다.
    let arch = arch_of(&read_config(dir)?)?;
    let quants: &[Quant] = if arch == Arch::Siglip2 { &[Quant::Q8] } else { &[Quant::Q4, Quant::Q8] };
    let mut passed: BTreeMap<&str, bool> = BTreeMap::new();
    for &quant in quants {
        let out = dir.join(format!("{name}-{}.gguf", quant.tag()));
        println!("== {} 변환: {}", quant.label(), out.display());
        convert(dir, quant, &out)?.print();
        let v = verify(dir, &out)?;
        print_verify(quant.tag(), &v);
        write_report(&dir.join(format!("precision-{}.json", quant.tag())), quant, &v)?;
        passed.insert(quant.tag(), v.pass);
    }
    let q4_ok = passed.get("q4").copied();
    let active = if q4_ok == Some(true) { "q4" } else { "q8" };
    let reason = match q4_ok {
        Some(true) => "q4 가 정밀도 기준을 통과했습니다",
        Some(false) => "q4 가 정밀도 기준을 넘어 q8 로 전환했습니다",
        None => "이 모델은 q4 를 만들지 않습니다",
    };
    println!("== 선택: {active} ({reason})");
    let record = json!({
        "model": name,
        "active": active,
        "file": format!("{name}-{active}.gguf"),
        "mmproj": arch.mmproj_file(),
        "reason": reason,
        "q4_pass": q4_ok,
        "q8_pass": passed["q8"],
    });
    fs::write(dir.join("active.json"), serde_json::to_string_pretty(&record)?)?;
    Ok(())
}

/// jev_head.pt (torch zip) 을 F32 safetensors 로 바꿉니다. 파이썬 pickle 을 실행하지 않고, candle 의 제한된 리더로 읽습니다.
fn head(pt: &Path, out: &Path) -> Result<()> {
    // jev_head.pt 의 루트는 {"head": {q.weight, q.bias, k.weight, k.bias}} 입니다 (pickletools 로 확인).
    let reader = candle_core::pickle::PthTensors::new(pt, Some("head"))
        .with_context(|| format!("torch 파일을 읽을 수 없습니다: {}", pt.display()))?;
    let mut names: Vec<&String> = reader.tensor_infos().keys().collect();
    names.sort();
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for name in names {
        let t = reader
            .get(name)?
            .ok_or_else(|| anyhow!("텐서를 읽을 수 없습니다: {name}"))?;
        println!("{name}: {:?} {:?}", t.dims(), t.dtype());
        tensors.insert(name.clone(), t.to_dtype(DType::F32)?);
    }
    candle_core::safetensors::save(&tensors, out)?;
    println!("저장: {} ({}개 텐서, F32)", out.display(), tensors.len());
    Ok(())
}

fn usage() -> ! {
    eprintln!(
        "사용법:\n  zorogravity-gguf convert <모델 폴더> <q4|q8> <출력.gguf>\n  zorogravity-gguf verify  <모델 폴더> <파일.gguf> [보고서.json]\n  zorogravity-gguf auto    <모델 폴더>\n  zorogravity-gguf head    <jev_head.pt> <출력.safetensors>"
    );
    std::process::exit(2)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("convert") if args.len() == 5 => {
            let quant = Quant::parse(&args[3])?;
            convert(Path::new(&args[2]), quant, Path::new(&args[4]))?.print();
            Ok(())
        }
        Some("verify") if args.len() == 4 || args.len() == 5 => {
            let dir = Path::new(&args[2]);
            let gguf = Path::new(&args[3]);
            let v = verify(dir, gguf)?;
            let tag = gguf.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
            print_verify(&tag, &v);
            if let Some(report) = args.get(4) {
                let quant = if tag.ends_with("q8") { Quant::Q8 } else { Quant::Q4 };
                write_report(Path::new(report), quant, &v)?;
            }
            Ok(())
        }
        Some("auto") if args.len() == 3 => auto(Path::new(&args[2])),
        Some("head") if args.len() == 4 => head(Path::new(&args[2]), Path::new(&args[3])),
        _ => usage(),
    }
}
