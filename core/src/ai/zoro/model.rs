//! zoro 텍스트 백본 순전파 (프리필 한 번). 가중치는 단계별로 `Residency` 에서 받아 쓰고, 쓰지 않는 가중치는 올리지 않습니다.
//!
//! 가중치 사용 방식 (docs/weight-residency.md):
//! - 토큰 임베딩 표(약 640 MB)는 올리지 않고, 프롬프트의 토큰 행만 파일에서 읽습니다.
//! - 비전 타워와 출력 투영(lm_head, 임베딩과 공유)은 포인터 판단에 쓰이지 않으므로 올리지 않습니다.
//! - 텍스트 블록은 예산이 넉넉하면 전부 상주, 빠듯하면 구간별로 올렸다 내립니다. 구간 사이에서는 이전 구간을 먼저 내립니다.
//!
//! 판단은 한 번의 프리필이라 KV 캐시와 순환 상태를 층 밖으로 넘기지 않습니다.
//! 글을 쓸 때만([`ZoroModel::generate`]) 층별 상태([`ZoroState`])를 이어 가고, 그때만 출력 투영으로 쓰는 임베딩 표를 올립니다.
//!
//! 글쓰기는 자기 추측 디코딩입니다 (`ai::speculative`). 검증할 토큰 뒤에 마스크를 이어 읽어 다음 후보를 얻고, 상태에는 어긋나기 전까지만 넣습니다.
//! 전체 주의는 키 · 값을 앞에서 잘라 넣으면 되지만 선형 주의의 순환 상태는 자를 수 없어서, 입력을 들고 있다가 그 자리까지 다시 계산합니다.

use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use candle_core::delta_rule::GdnConfig;
use candle_core::{Device, Tensor};
use candle_nn::ops::{sigmoid, silu};

use super::config::{LayerKind, ZoroConfig};
use super::ops::{gated_delta_state, gated_delta_state_on_host, gated_delta_step, gated_delta_step_on_host, gdn_config, gdn_on_device, GdnState, GdnWeights};
use crate::ai::attention::{attend, causal_attention, rms_norm, top1, AttnMode, Causal, LayerCache, Rope, StepMask, ATTN_QUERY_CHUNK};
use crate::ai::residency::{estimate_group_bytes, Budget, Group, Loaded, ModelId, ModelSpec, Phase, Prepare, SharedResidency};
use crate::ai::speculative::{decode_stream, BlockDiffusion, Speculation, StreamModel, StreamOptions};
use crate::ai::weights::WeightFile;

/// `ZOROGRAVITY_ZORO_PROFILE=1` 이면 층 종류별 시간을 표준 오류로 출력합니다 (측정을 위해 층마다 장치 작업을 기다리므로 느려집니다).
fn profile_enabled() -> bool {
    std::env::var("ZOROGRAVITY_ZORO_PROFILE").map(|v| v == "1").unwrap_or(false)
}

/// `ZOROGRAVITY_ZORO_TRACE=1` 이면 층마다 출력의 체크섬을 표준 오류로 냅니다 (재현성 진단용, 느려짐).
fn layer_trace_enabled() -> bool {
    std::env::var("ZOROGRAVITY_ZORO_TRACE").map(|v| v == "1").unwrap_or(false)
}

/// 한 번에 처리하는 토큰 수의 상한. 판단 프롬프트는 이보다 훨씬 짧습니다.
pub const MAX_TOKENS: usize = 8192;
/// 텍스트 블록을 올리는 방식을 정하는 환경 변수: `auto`(기본) · `resident`(= 0) · 숫자 N (블록 N 개씩 올렸다 내림)
pub const STREAM_ENV: &str = "ZOROGRAVITY_ZORO_STREAM_BLOCKS";
/// ROCm: 3행까지는 순전파 비용이 조금만 늘어서 마스크까지 싣는 편이 빠릅니다 (토큰당 35.7 → 30.2 ms).
const ROCM_SPECULATION: Speculation = Speculation { masks: 2, lookup: 2, width: 3 };
/// Vulkan · 블록 디퓨전 설정이 없는 파일: 문맥에서 찾은 후보만 씁니다 (Vulkan 80.0 → 74.9 ms, 마스크를 실으면 88.1 ms).
const LOOKUP_ONLY: Speculation = Speculation { masks: 0, lookup: 2, width: 3 };

/// 올릴 때 한 번 하는 변환.
///
/// - 디코더 RMSNorm 가중치는 0 근처에서 시작하는 형태(`정규화 × (1 + w)`)라서 1 을 더해 둡니다.
/// - `A_log` 는 감쇠율 `−exp(A_log)` 로 바꿔 둡니다.
///
/// 선형 주의 안의 게이트 정규화(`ssm_norm`)는 가중치를 그대로 곱하므로 건드리지 않습니다.
pub const PREPARE: Prepare = Prepare { touches: prepared, apply: prepare };

const ZERO_CENTERED: [&str; 5] =
    [".attn_norm.weight", ".post_attention_norm.weight", ".attn_q_norm.weight", ".attn_k_norm.weight", "output_norm.weight"];

fn prepared(name: &str) -> bool {
    name.ends_with(".ssm_a") || ZERO_CENTERED.iter().any(|s| name.ends_with(s))
}

fn prepare(name: &str, t: Tensor) -> candle_core::Result<Tensor> {
    if name.ends_with(".ssm_a") {
        t.exp()?.neg()
    } else if ZERO_CENTERED.iter().any(|s| name.ends_with(s)) {
        t.affine(1.0, 1.0)
    } else {
        Ok(t)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamMode {
    /// 예산에 모두 들어가면 상주, 아니면 예산에 맞는 구간으로 나눠 스트리밍
    Auto,
    /// 항상 모든 블록을 상주 (예산을 넘으면 오류)
    Resident,
    /// 블록 N 개씩 올렸다 내림
    Chunks(usize),
}

impl StreamMode {
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "0" | "resident" => Ok(Self::Resident),
            n => match n.parse::<usize>() {
                Ok(k) if k > 0 => Ok(Self::Chunks(k)),
                _ => bail!("{STREAM_ENV}={n:?}: auto, resident, 또는 양의 정수여야 합니다"),
            },
        }
    }

    pub fn from_env() -> Result<Self> {
        match std::env::var(STREAM_ENV) {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(Self::Auto),
        }
    }
}

/// 블록별 바이트 수를 `limit` 안에서 이어 붙여 구간으로 나눕니다. 블록 하나가 `limit` 보다 커도 그 블록은 구간 하나가 됩니다
/// (올릴 때 예산 오류로 알려 줍니다).
pub fn chunk_ranges(per_block: &[u64], limit: u64) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let (mut start, mut used) = (0usize, 0u64);
    for (i, &b) in per_block.iter().enumerate() {
        if i > start && used.saturating_add(b) > limit {
            ranges.push(start..i);
            start = i;
            used = 0;
        }
        used = used.saturating_add(b);
    }
    if start < per_block.len() {
        ranges.push(start..per_block.len());
    }
    ranges
}

/// 모든 블록을 구간 하나로
fn whole(n: usize) -> Vec<Range<usize>> {
    std::iter::once(0..n).collect()
}

fn plan_phases(cfg: &ZoroConfig, file: &WeightFile, device: &Device, budget: Budget, mode: StreamMode) -> Result<Vec<Phase>> {
    let n = cfg.n_layers;
    let per_block = (0..n)
        .map(|i| estimate_group_bytes(file, Group::blocks(i..i + 1), device))
        .collect::<Result<Vec<_>>>()?;
    let output = estimate_group_bytes(file, Group::Output, device)?;
    let ranges: Vec<Range<usize>> = match mode {
        StreamMode::Resident => whole(n),
        StreamMode::Chunks(k) => (0..n).step_by(k).map(|s| s..(s + k).min(n)).collect(),
        StreamMode::Auto => match budget.max_bytes() {
            None => whole(n),
            Some(max) if per_block.iter().sum::<u64>() + output <= max => whole(n),
            Some(max) => chunk_ranges(&per_block, max.saturating_sub(output)),
        },
    };
    let last = ranges.len() - 1;
    Ok(ranges
        .into_iter()
        .enumerate()
        .map(|(i, r)| {
            let name = format!("zoro:블록 {}..{}", r.start, r.end);
            let mut groups = vec![Group::blocks(r)];
            if i == last {
                groups.push(Group::Output);
            }
            Phase::new(name, groups)
        })
        .collect())
}

/// 파일의 텐서 모양이 이 구조 값과 맞는지 확인해, 잘못된 파일을 첫 순전파 전에 알립니다.
fn validate_tensors(cfg: &ZoroConfig, file: &WeightFile) -> Result<()> {
    let check = |name: &str, want: &[usize]| -> Result<()> {
        let got = file.info(name)?.dims;
        if got != want {
            bail!("{name} 의 모양이 {want:?} 이어야 하는데 {got:?} 입니다");
        }
        Ok(())
    };
    let h = cfg.hidden;
    check("token_embd.weight", &[cfg.vocab, h])?;
    check("output_norm.weight", &[h])?;
    for i in 0..cfg.n_layers {
        let p = |s: &str| format!("blk.{i}.{s}");
        check(&p("attn_norm.weight"), &[h])?;
        check(&p("post_attention_norm.weight"), &[h])?;
        check(&p("ffn_gate.weight"), &[cfg.ffn, h])?;
        check(&p("ffn_up.weight"), &[cfg.ffn, h])?;
        check(&p("ffn_down.weight"), &[h, cfg.ffn])?;
        match cfg.kind(i) {
            LayerKind::LinearAttention => {
                check(&p("attn_qkv.weight"), &[cfg.gdn_conv_dim(), h])?;
                check(&p("attn_gate.weight"), &[cfg.gdn_value_dim(), h])?;
                check(&p("ssm_beta.weight"), &[cfg.gdn_v_heads, h])?;
                check(&p("ssm_alpha.weight"), &[cfg.gdn_v_heads, h])?;
                check(&p("ssm_out.weight"), &[h, cfg.gdn_value_dim()])?;
                check(&p("ssm_conv1d.weight"), &[cfg.gdn_conv_dim(), 1, cfg.conv_kernel])?;
                check(&p("ssm_dt.bias"), &[cfg.gdn_v_heads])?;
                check(&p("ssm_a"), &[cfg.gdn_v_heads])?;
                check(&p("ssm_norm.weight"), &[cfg.gdn_v_dim])?;
            }
            LayerKind::FullAttention => {
                check(&p("attn_q.weight"), &[cfg.q_proj_rows(), h])?;
                check(&p("attn_k.weight"), &[cfg.n_kv_heads * cfg.head_dim, h])?;
                check(&p("attn_v.weight"), &[cfg.n_kv_heads * cfg.head_dim, h])?;
                check(&p("attn_output.weight"), &[h, cfg.n_heads * cfg.head_dim])?;
                check(&p("attn_q_norm.weight"), &[cfg.head_dim])?;
                check(&p("attn_k_norm.weight"), &[cfg.head_dim])?;
            }
        }
    }
    Ok(())
}

/// 층 하나의 이어 읽기 상태
enum LayerState {
    Linear(Option<GdnState>),
    Full(LayerCache),
}

/// 지금까지 읽은 토큰의 상태. 층마다 전체 주의의 키 · 값이나 선형 주의의 순환 상태를 담습니다.
pub struct ZoroState {
    layers: Vec<LayerState>,
    len: usize,
}

impl ZoroState {
    /// 지금까지 읽은 토큰 수
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// 선형 주의에 넣은 순환 입력. 앞 몇 토큰만 상태에 넣을 때 이 값으로 그 자리의 상태를 다시 계산합니다.
struct GdnInputs {
    mixed: Tensor,
    a: Tensor,
    b: Tensor,
}

/// 아직 상태에 넣지 않은 층 하나의 결과
enum PendingLayer {
    Linear { next: GdnState, inputs: Option<GdnInputs> },
    Full { kt: Tensor, v: Tensor },
}

/// 새 토큰을 읽은 결과. 상태에 넣을지, 앞 몇 개만 넣을지는 호출자가 정합니다 (초안은 버리고, 검증은 어긋나기 전까지만 넣습니다).
struct PendingRead {
    layers: Vec<PendingLayer>,
    n: usize,
}

/// 층 하나를 어떻게 이어 읽는가
#[derive(Clone, Copy)]
struct LayerRead<'a> {
    state: &'a LayerState,
    /// 앞 몇 토큰만 상태에 넣을 수 있도록 순환 입력을 들고 있는가
    partial: bool,
}

/// 글쓰기 설정
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateOptions {
    pub max_tokens: usize,
    /// 이 토큰이 나오면 멈춥니다.
    pub stop: Vec<u32>,
    /// 순전파 하나에 실어 보내는 추측의 양. 얼마이든 같은 글이 나옵니다.
    pub speculation: Speculation,
}

impl GenerateOptions {
    pub fn new(max_tokens: usize, stop: Vec<u32>) -> Self {
        Self { max_tokens, stop, speculation: Speculation::NONE }
    }

    pub fn with_speculation(mut self, speculation: Speculation) -> Self {
        self.speculation = speculation;
        self
    }
}

/// 글쓰기 결과 (토큰 번호)
#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedIds {
    pub tokens: Vec<u32>,
    /// 토큰마다 모델이 준 확률
    pub confidences: Vec<f32>,
    /// 끝 토큰으로 멈췄는가 (아니면 길이 제한이나 반복으로 멈춤)
    pub finished: bool,
    /// 같은 말을 되풀이해서 멈췄는가
    pub looped: bool,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// 순전파 횟수
    pub reads: usize,
    /// 검증에 넘긴 후보 수와 그중 채택한 수
    pub drafted: usize,
    pub accepted: usize,
}

/// 끝이 같은 토큰 묶음의 반복인가. 묶음을 세 번 이상, 합쳐서 `MIN_SPAN` 토큰 이상 되풀이하면 그만 씁니다.
/// 숫자는 한 자가 한 토큰이라 `500,000` 의 `000` 처럼 짧은 반복은 정상입니다.
fn is_looping(tokens: &[u32]) -> bool {
    const MAX_PERIOD: usize = 32;
    const MIN_SPAN: usize = 12;
    (1..=MAX_PERIOD).any(|p| {
        let span = (3 * p).max(MIN_SPAN.div_ceil(p) * p);
        let n = tokens.len();
        n >= span && (p..span).all(|i| tokens[n - 1 - i] == tokens[n - 1 - i + p])
    })
}

/// 전체 주의의 마스크. 층마다 같아서 순전파마다 한 번만 올립니다.
enum Masks {
    /// 상태 없이 프롬프트 전체를 읽을 때
    Whole(Causal),
    /// 상태에 이어 읽을 때
    Stepping(StepMask),
}

/// 순전파 한 번 동안 공유하는 값
struct Ctx {
    rope: Rope,
    masks: Masks,
    gdn: GdnConfig,
    /// 선형 주의의 핵심을 장치 커널로 계산할 수 있는가 (아니면 CPU 커널)
    gdn_native: bool,
}

pub struct ZoroModel {
    cfg: ZoroConfig,
    spec: ModelSpec,
    residency: SharedResidency,
    plan: Vec<Phase>,
    /// 블록 디퓨전 설정. 없으면 초안을 쓰지 않습니다.
    diffusion: Option<BlockDiffusion>,
    /// 장치 커널이 없을 때 쓰는 선형 주의 작은 텐서(층당 약 130 KB)의 CPU 사본. 가중치 상주와 별개로 RAM 에 남습니다.
    host: Mutex<HashMap<usize, Arc<GdnWeights>>>,
}

impl ZoroModel {
    pub fn new(file: Arc<WeightFile>, residency: SharedResidency, mode: StreamMode) -> Result<Self> {
        let cfg = ZoroConfig::from_file(&file)?;
        validate_tensors(&cfg, &file)?;
        let spec = ModelSpec::new(ModelId::Zoro, Arc::clone(&file)).with_prepare(PREPARE);
        let (device, budget) = {
            let r = residency.lock().unwrap_or_else(|p| p.into_inner());
            (r.device().clone(), r.budget())
        };
        let plan = plan_phases(&cfg, &file, &device, budget, mode)?;
        let diffusion = BlockDiffusion::from_file(&file)?.map(|d| d.validate(cfg.vocab)).transpose()?;
        Ok(Self { cfg, spec, residency, plan, diffusion, host: Mutex::new(HashMap::new()) })
    }

    /// 블록 디퓨전 설정을 정합니다.
    pub fn with_diffusion(mut self, diffusion: Option<BlockDiffusion>) -> Result<Self> {
        self.diffusion = diffusion.map(|d| d.validate(self.cfg.vocab)).transpose()?;
        Ok(self)
    }

    /// 이 장치와 파일에 알맞은 추측의 양
    pub fn default_speculation(&self) -> Speculation {
        let device = self.residency.lock().unwrap_or_else(|p| p.into_inner()).device().clone();
        speculation_for(&device, self.diffusion.is_some())
    }

    /// 비전 타워를 따로 둔 파일을 붙입니다.
    pub fn with_mmproj(mut self, mmproj: Arc<WeightFile>) -> Self {
        self.spec = self.spec.with_mmproj(mmproj);
        self
    }

    pub fn config(&self) -> &ZoroConfig {
        &self.cfg
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    /// 단계 계획 (구간이 하나면 상주, 여럿이면 스트리밍)
    pub fn plan(&self) -> &[Phase] {
        &self.plan
    }

    /// 토큰 번호들을 읽어 최종 정규화까지 마친 은닉 상태 `[T, hidden]` 를 장치 위에 돌려줍니다.
    pub fn hidden_states(&self, ids: &[u32]) -> Result<Tensor> {
        Ok(self.hidden_states_pair(ids)?.1)
    }

    /// 최종 정규화 전 · 후의 은닉 상태 `([T, hidden], [T, hidden])`
    pub fn hidden_states_pair(&self, ids: &[u32]) -> Result<(Tensor, Tensor)> {
        if ids.is_empty() {
            bail!("토큰이 없습니다");
        }
        if ids.len() > MAX_TOKENS {
            bail!("토큰 {}개는 한 번에 처리할 수 있는 {MAX_TOKENS}개를 넘습니다", ids.len());
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab) {
            bail!("토큰 번호 {bad} 가 어휘 크기 {} 를 넘습니다", self.cfg.vocab);
        }
        let mut res = self.residency.lock().unwrap_or_else(|p| p.into_inner());
        let device = res.device().clone();
        let t = ids.len();

        // 임베딩 표는 올리지 않고 필요한 행만 파일에서 읽습니다.
        let mut x = self.spec.file.gather_rows("token_embd.weight", ids, &device)?;
        let gdn = gdn_config(&self.cfg);
        let ctx = Ctx {
            rope: Rope::new(t, self.cfg.rope_dim, self.cfg.rope_theta, &device)?,
            masks: Masks::Whole(Causal::new(t, ATTN_QUERY_CHUNK, &device)?),
            gdn_native: gdn_on_device(&device, &gdn),
            gdn,
        };

        let profile = profile_enabled();
        let mut spent = [std::time::Duration::ZERO; 3]; // 선형 주의 층 · 전체 주의 층 · 기타(적재 포함)
        let started = std::time::Instant::now();
        let mut before_norm = None;
        for phase in &self.plan {
            // 이 단계가 쓰는 묶음만 올라온 상태로 만듭니다 (이전 구간은 여기서 내려갑니다).
            let loaded = res.enter(&self.spec, phase)?;
            for l in &loaded {
                match l.group() {
                    Group::Blocks { start, end } => {
                        for i in start as usize..end as usize {
                            let t0 = std::time::Instant::now();
                            x = self.layer(&ctx, l, i, &x, None)?.0;
                            if layer_trace_enabled() {
                                // 진단: 층 출력의 체크섬. 두 번의 실행을 비교해 처음 달라지는 층을 찾습니다.
                                let v = x.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
                                let sum: f64 = v.iter().map(|e| *e as f64).sum();
                                let sq: f64 = v.iter().map(|e| (*e as f64) * (*e as f64)).sum();
                                let bits = v.iter().fold(0u64, |a, e| a.wrapping_mul(1_000_003).wrapping_add(e.to_bits() as u64));
                                eprintln!("[ZOROGRAVITY-TRACE] 층 {i:>2} {:?} 합 {sum:+.9e} 제곱합 {sq:.9e} 비트 {bits:016x}", self.cfg.kind(i));
                            }
                            if profile {
                                // 장치 작업이 끝난 시간을 재려고 값을 한 번 읽습니다 (프로필 모드에서만).
                                let _ = x.narrow(0, 0, 1)?.to_device(&Device::Cpu)?;
                                let k = if self.cfg.kind(i) == LayerKind::LinearAttention { 0 } else { 1 };
                                spent[k] += t0.elapsed();
                            }
                        }
                    }
                    Group::Output => {
                        let normed = rms_norm(&x, l.dense("output_norm.weight")?, self.cfg.rms_eps)?;
                        before_norm = Some(std::mem::replace(&mut x, normed));
                    }
                    other => bail!("zoro 단계에 예상하지 못한 묶음: {other}"),
                }
            }
            // `loaded` 는 이 반복이 끝나면서 놓입니다. 다음 단계에서 이전 구간을 내릴 수 있어야 하기 때문입니다.
        }
        if profile {
            spent[2] = started.elapsed().saturating_sub(spent[0] + spent[1]);
            eprintln!(
                "[ZOROGRAVITY-PROFILE] 토큰 {t}개: 선형 주의 층 합계 {:.0} ms · 전체 주의 층 합계 {:.0} ms · 나머지(적재 · 임베딩 · 정규화) {:.0} ms",
                spent[0].as_secs_f64() * 1e3,
                spent[1].as_secs_f64() * 1e3,
                spent[2].as_secs_f64() * 1e3,
            );
        }
        let before = before_norm.ok_or_else(|| anyhow!("단계 계획에 출력 정규화가 없습니다"))?;
        Ok((before, x))
    }

    /// 쓸 단계의 파일을 배경에서 미리 읽습니다.
    pub fn announce(&self) {
        self.spec.announce(&self.generation_phase());
    }

    /// 글쓰기에 쓰는 묶음: 텍스트 블록 전부 · 출력 정규화 · 임베딩 표 (출력 투영과 공유)
    pub fn generation_phase(&self) -> Phase {
        Phase::new("zoro:글쓰기", [Group::blocks(0..self.cfg.n_layers), Group::Output, Group::Embedding])
    }

    pub fn new_state(&self) -> ZoroState {
        let layers = (0..self.cfg.n_layers)
            .map(|i| match self.cfg.kind(i) {
                LayerKind::LinearAttention => LayerState::Linear(None),
                LayerKind::FullAttention => LayerState::Full(LayerCache::default()),
            })
            .collect();
        ZoroState { layers, len: 0 }
    }

    fn check_ids(&self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() {
            bail!("토큰이 없습니다");
        }
        if let Some(bad) = ids.iter().find(|&&id| id as usize >= self.cfg.vocab) {
            bail!("토큰 번호 {bad} 가 어휘 크기 {} 를 넘습니다", self.cfg.vocab);
        }
        Ok(())
    }

    /// 새 토큰을 이어 읽고 최종 정규화까지 마친 은닉 상태 `[n, hidden]` 를 돌려줍니다.
    pub fn advance(&self, blocks: &Loaded, output: &Loaded, device: &Device, state: &mut ZoroState, ids: &[u32]) -> Result<Tensor> {
        let (h, read) = self.read(blocks, output, device, state, ids, false)?;
        let n = read.n;
        self.commit(blocks, device, state, read, n)?;
        Ok(h)
    }

    /// 새 토큰을 읽습니다. 상태는 바꾸지 않고, 상태에 넣을 값을 돌려줍니다.
    fn read(&self, blocks: &Loaded, output: &Loaded, device: &Device, state: &ZoroState, ids: &[u32], partial: bool) -> Result<(Tensor, PendingRead)> {
        self.check_ids(ids)?;
        let n = ids.len();
        let mut x = self.spec.file.gather_rows("token_embd.weight", ids, device)?;
        let gdn = gdn_config(&self.cfg);
        let ctx = Ctx {
            rope: Rope::span(state.len, n, self.cfg.rope_dim, self.cfg.rope_theta, device)?,
            masks: Masks::Stepping(StepMask::new(AttnMode::Causal, n, self.cfg.n_heads / self.cfg.n_kv_heads, device)?),
            gdn_native: gdn_on_device(device, &gdn),
            gdn,
        };
        let mut layers = Vec::with_capacity(self.cfg.n_layers);
        for i in 0..self.cfg.n_layers {
            let (y, pending) = self.layer(&ctx, blocks, i, &x, Some(LayerRead { state: &state.layers[i], partial }))?;
            x = y;
            layers.push(pending.ok_or_else(|| anyhow!("층 {i} 의 상태가 없습니다"))?);
        }
        Ok((rms_norm(&x, output.dense("output_norm.weight")?, self.cfg.rms_eps)?, PendingRead { layers, n }))
    }

    /// 읽은 토큰 앞 `keep` 개를 상태에 넣습니다. 전부가 아니면 선형 주의는 순환 입력으로 그 자리의 상태를 다시 계산합니다.
    fn commit(&self, blocks: &Loaded, device: &Device, state: &mut ZoroState, read: PendingRead, keep: usize) -> Result<()> {
        if keep > read.n {
            bail!("넣을 토큰 {keep}개가 읽은 토큰 {}개보다 많습니다", read.n);
        }
        if keep == 0 {
            return Ok(());
        }
        let whole = keep == read.n;
        let cut = |x: &Tensor, dim: usize| -> Result<Tensor> { Ok(x.narrow(dim, 0, keep)?.contiguous()?) };
        for (i, (layer, pending)) in state.layers.iter_mut().zip(read.layers).enumerate() {
            match (layer, pending) {
                (LayerState::Full(cache), PendingLayer::Full { kt, v }) if whole => cache.push(&kt, &v)?,
                (LayerState::Full(cache), PendingLayer::Full { kt, v }) => cache.push(&cut(&kt, 2)?, &cut(&v, 1)?)?,
                (LayerState::Linear(s), PendingLayer::Linear { next, .. }) if whole => *s = Some(next),
                (LayerState::Linear(s), PendingLayer::Linear { inputs, .. }) => {
                    let GdnInputs { mixed, a, b } = inputs.ok_or_else(|| anyhow!("층 {i} 은 앞부분만 넣을 수 있게 읽지 않았습니다"))?;
                    *s = Some(self.rewound_state(blocks, device, i, &cut(&mixed, 0)?, &cut(&a, 0)?, &cut(&b, 0)?, s.as_ref())?);
                }
                _ => bail!("층 {i} 의 상태 종류가 층 구조와 다릅니다"),
            }
        }
        state.len += keep;
        Ok(())
    }

    /// `previous` 에서 순환 입력을 이어 읽은 뒤의 선형 주의 상태
    #[allow(clippy::too_many_arguments)]
    fn rewound_state(&self, w: &Loaded, device: &Device, i: usize, mixed: &Tensor, a: &Tensor, b: &Tensor, previous: Option<&GdnState>) -> Result<GdnState> {
        let gdn = gdn_config(&self.cfg);
        if gdn_on_device(device, &gdn) {
            gated_delta_state(mixed, a, b, &self.small_weights(w, i)?, &gdn, previous)
        } else {
            let host = self.host_weights(i, || self.small_weights(w, i))?;
            gated_delta_state_on_host(mixed, a, b, &host, &gdn, previous)
        }
    }

    /// 프롬프트를 읽고 이어서 씁니다. 가장 확률이 높은 토큰을 고르므로 같은 입력이면 같은 글이 나옵니다.
    /// 호출하는 동안 상주 관리자를 잡고 있고, 모든 블록이 함께 있어야 하므로 예산이 모자라면 오류입니다.
    ///
    /// 추측이 있으면 자기 추측 디코딩입니다. 모든 토큰을 자기회귀로 검증하므로 글은 달라지지 않고 시간만 달라집니다.
    /// 초안과 검증은 같은 가중치를 쓰므로 올리는 묶음은 그대로이고, 비전 타워(mmproj)는 어느 쪽에도 쓰이지 않습니다.
    pub fn generate(&self, prompt: &[u32], opts: &GenerateOptions) -> Result<GeneratedIds> {
        self.check_ids(prompt)?;
        if prompt.len() + opts.max_tokens > MAX_TOKENS {
            bail!("프롬프트 {} + 생성 {} 토큰은 상한 {MAX_TOKENS} 을 넘습니다", prompt.len(), opts.max_tokens);
        }
        if opts.speculation.masks > 0 && self.diffusion.is_none() {
            bail!("블록 디퓨전 설정이 없어서 마스크 초안을 쓸 수 없습니다");
        }
        let mut res = self.residency.lock().unwrap_or_else(|p| p.into_inner());
        let device = res.device().clone();
        let loaded = res.enter(&self.spec, &self.generation_phase())?;
        let (blocks, output, table) = (&loaded[0], &loaded[1], &loaded[2]);

        let t = Instant::now();
        let mut state = self.new_state();
        let h = self.advance(blocks, output, &device, &mut state, prompt)?;
        let first = next_tokens(table, &h.narrow(0, h.dim(0)? - 1, 1)?)?[0];
        let prefill_ms = t.elapsed().as_secs_f64() * 1e3;

        let t = Instant::now();
        let mut writer = Writer { model: self, blocks, output, table, device: &device, state, pending: None };
        // 공문서에 들어갈 글이라 모든 토큰을 검증합니다 (조기 종료 없음).
        let opts = StreamOptions { eos: &opts.stop, max_tokens: opts.max_tokens, speculation: opts.speculation, context: prompt, halt: Some(is_looping) };
        let out = decode_stream(&mut writer, first, &opts)?;
        Ok(GeneratedIds {
            finished: !out.truncated,
            looped: out.halted,
            tokens: out.tokens,
            confidences: out.confs,
            prefill_ms,
            decode_ms: t.elapsed().as_secs_f64() * 1e3,
            reads: out.verifies,
            drafted: out.drafted,
            accepted: out.accepted,
        })
    }

    fn layer(&self, ctx: &Ctx, w: &Loaded, i: usize, x: &Tensor, read: Option<LayerRead<'_>>) -> Result<(Tensor, Option<PendingLayer>)> {
        let p = |s: &str| format!("blk.{i}.{s}");
        let eps = self.cfg.rms_eps;
        let h = rms_norm(x, w.dense(&p("attn_norm.weight"))?, eps)?;
        let (mix, pending) = match (self.cfg.kind(i), read) {
            (LayerKind::LinearAttention, None) => self.linear_attention(ctx, w, i, &h, None)?,
            (LayerKind::FullAttention, None) => self.full_attention(ctx, w, i, &h, None)?,
            (LayerKind::LinearAttention, Some(LayerRead { state: LayerState::Linear(s), partial, .. })) => {
                self.linear_attention(ctx, w, i, &h, Some((s.as_ref(), partial)))?
            }
            (LayerKind::FullAttention, Some(LayerRead { state: LayerState::Full(c), .. })) => self.full_attention(ctx, w, i, &h, Some(c))?,
            _ => bail!("층 {i} 의 상태 종류가 층 구조와 다릅니다"),
        };
        let x = (x + mix)?;
        let h = rms_norm(&x, w.dense(&p("post_attention_norm.weight"))?, eps)?;
        let gate = w.linear(&p("ffn_gate.weight"), &h)?;
        let up = w.linear(&p("ffn_up.weight"), &h)?;
        let act = (silu(&gate)? * &up)?;
        let down = w.linear(&p("ffn_down.weight"), &act)?;
        Ok(((x + down)?, pending))
    }

    /// 출력 게이트가 있는 전체 주의. `attn_q` 의 출력은 헤드마다 [질의 | 게이트] 로 나뉩니다.
    fn full_attention(&self, ctx: &Ctx, w: &Loaded, i: usize, h: &Tensor, cache: Option<&LayerCache>) -> Result<(Tensor, Option<PendingLayer>)> {
        let c = &self.cfg;
        let p = |s: &str| format!("blk.{i}.{s}");
        let t = h.dim(0)?;
        let (nh, nkv, d) = (c.n_heads, c.n_kv_heads, c.head_dim);
        let qg = w.linear(&p("attn_q.weight"), h)?.reshape((t, nh, 2 * d))?;
        let q = qg.narrow(2, 0, d)?.contiguous()?;
        let gate = qg.narrow(2, d, d)?.contiguous()?.reshape((t, nh * d))?;
        let k = w.linear(&p("attn_k.weight"), h)?.reshape((t, nkv, d))?;
        let v = w.linear(&p("attn_v.weight"), h)?.reshape((t, nkv, d))?;
        // 질의·키는 헤드별로 정규화한 뒤 RoPE
        let q = rms_norm(&q, w.dense(&p("attn_q_norm.weight"))?, c.rms_eps)?;
        let k = rms_norm(&k, w.dense(&p("attn_k_norm.weight"))?, c.rms_eps)?;
        let q = ctx.rope.apply(&q.transpose(0, 1)?.contiguous()?)?;
        let k = ctx.rope.apply(&k.transpose(0, 1)?.contiguous()?)?;
        let v = v.transpose(0, 1)?.contiguous()?;
        let scale = 1.0 / (d as f64).sqrt();
        let (o, pending) = match (cache, &ctx.masks) {
            (None, Masks::Whole(causal)) => (causal_attention(&q, &k, &v, scale, causal)?, None),
            (Some(cache), Masks::Stepping(mask)) => {
                let kt = k.transpose(1, 2)?.contiguous()?;
                (attend(&q, &kt, &v, &cache.segments(), mask, scale)?, Some(PendingLayer::Full { kt, v }))
            }
            _ => bail!("층 {i} 의 상태와 마스크가 맞지 않습니다"),
        };
        let o = o.transpose(0, 1)?.contiguous()?.reshape((t, nh * d))?;
        let o = (o * sigmoid(&gate)?)?;
        Ok((w.linear(&p("attn_output.weight"), &o)?, pending))
    }

    /// 게이트 델타 규칙 선형 주의. `read` 는 이전 상태와, 앞 몇 토큰만 상태에 넣을 수 있게 읽을지입니다.
    fn linear_attention(&self, ctx: &Ctx, w: &Loaded, i: usize, h: &Tensor, read: Option<(Option<&GdnState>, bool)>) -> Result<(Tensor, Option<PendingLayer>)> {
        let p = |s: &str| format!("blk.{i}.{s}");
        let mixed = w.linear(&p("attn_qkv.weight"), h)?;
        let z = w.linear(&p("attn_gate.weight"), h)?;
        let b = w.linear(&p("ssm_beta.weight"), h)?;
        let a = w.linear(&p("ssm_alpha.weight"), h)?;
        let eps = self.cfg.rms_eps as f64;
        let previous = read.and_then(|(s, _)| s);
        let (y, next) = if ctx.gdn_native {
            gated_delta_step(&mixed, &z, &a, &b, &self.small_weights(w, i)?, &ctx.gdn, eps, previous)?
        } else {
            let host = self.host_weights(i, || self.small_weights(w, i))?;
            gated_delta_step_on_host(&mixed, &z, &a, &b, &host, &ctx.gdn, eps, previous)?
        };
        let y = w.linear(&p("ssm_out.weight"), &y)?;
        Ok((y, read.map(|(_, partial)| PendingLayer::Linear { next, inputs: partial.then_some(GdnInputs { mixed, a, b }) })))
    }

    fn small_weights(&self, w: &Loaded, i: usize) -> Result<GdnWeights> {
        let p = |s: &str| format!("blk.{i}.{s}");
        Ok(GdnWeights {
            conv: w.dense(&p("ssm_conv1d.weight"))?.clone(),
            neg_a: w.dense(&p("ssm_a"))?.clone(),
            dt_bias: w.dense(&p("ssm_dt.bias"))?.clone(),
            norm: w.dense(&p("ssm_norm.weight"))?.clone(),
        })
    }

    fn host_weights(&self, layer: usize, small: impl FnOnce() -> Result<GdnWeights>) -> Result<Arc<GdnWeights>> {
        let mut cache = self.host.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(w) = cache.get(&layer) {
            return Ok(Arc::clone(w));
        }
        let w = Arc::new(small()?.to_device(&Device::Cpu)?);
        cache.insert(layer, Arc::clone(&w));
        Ok(w)
    }
}

/// 잰 장치에서만 추측합니다. 순전파 비용이 행 수에 비례하면 추측이 오히려 손해이기 때문입니다.
fn speculation_for(device: &Device, has_masks: bool) -> Speculation {
    if device.is_rocm() && has_masks {
        ROCM_SPECULATION
    } else if device.is_rocm() || device.is_vulkan() {
        LOOKUP_ONLY
    } else {
        Speculation::NONE
    }
}

/// 은닉 상태의 행마다 다음 토큰과 그 확률을 고릅니다. 출력 투영은 임베딩 표입니다 (`tie_word_embeddings`).
fn next_tokens(table: &Loaded, h: &Tensor) -> Result<Vec<(u32, f32)>> {
    let y = table.linear("token_embd.weight", &h.contiguous()?)?;
    let vocab = y.dim(1)?;
    let y = y.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    Ok(y.chunks(vocab).map(top1).collect())
}

/// 글쓰기 한 번의 가중치와 상태를 [`BlockModel`] 로 묶습니다.
struct Writer<'a> {
    model: &'a ZoroModel,
    blocks: &'a Loaded,
    output: &'a Loaded,
    table: &'a Loaded,
    device: &'a Device,
    state: ZoroState,
    pending: Option<PendingRead>,
}

impl StreamModel for Writer<'_> {
    fn read(&mut self, tokens: &[u32], masks: usize) -> Result<Vec<(u32, f32)>> {
        let mut ids = tokens.to_vec();
        if masks > 0 {
            let mask = self.model.diffusion.map(|d| d.mask_id).ok_or_else(|| anyhow!("블록 디퓨전 설정이 없습니다"))?;
            ids.resize(tokens.len() + masks, mask);
        }
        let (h, read) = self.model.read(self.blocks, self.output, self.device, &self.state, &ids, ids.len() > 1)?;
        self.pending = Some(read);
        next_tokens(self.table, &h)
    }

    fn commit(&mut self, keep: usize) -> Result<()> {
        let read = self.pending.take().ok_or_else(|| anyhow!("검증한 토큰이 없습니다"))?;
        self.model.commit(self.blocks, self.device, &mut self.state, read, keep)
    }
}

/// 은닉 상태의 선택한 행을 CPU 로 가져옵니다. 수치가 깨졌으면 (NaN · 무한대) 오류입니다. 깨진 값으로 판단하지 않기 위해서입니다.
pub fn rows_to_host(hidden: &Tensor, rows: &[usize]) -> Result<Vec<Vec<f32>>> {
    let idx: Vec<u32> = rows.iter().map(|&r| r as u32).collect();
    let picked = hidden.index_select(&Tensor::new(idx, hidden.device())?, 0)?.to_device(&Device::Cpu)?;
    let out = picked.to_vec2::<f32>()?;
    if out.iter().flatten().any(|v| !v.is_finite()) {
        return Err(anyhow!("백본 출력에 NaN 또는 무한대가 있습니다. 이 결과로 판단하지 않습니다"));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::residency::{Action, Residency};
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use crate::ai::zoro::reference::MiniZoro;
    use candle_core::quantized::gguf_file::Value;
    use candle_core::quantized::GgmlDType;

    const IDS: [u32; 11] = [5, 17, 250, 3, 99, 0, 42, 299, 120, 7, 61];

    fn cpu_residency() -> SharedResidency {
        Residency::shared(Device::Cpu, Budget::unlimited())
    }

    fn model(dir: &TempDir, quantize: bool, mode: StreamMode, res: SharedResidency) -> (MiniZoro, ZoroModel) {
        let mini = MiniZoro::new();
        let file = mini.write(dir, quantize).unwrap();
        let m = ZoroModel::new(file, res, mode).unwrap();
        (mini, m)
    }

    fn run(m: &ZoroModel, ids: &[u32]) -> Vec<f32> {
        m.hidden_states(ids).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let d: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let n: f32 = b.iter().map(|y| y * y).sum();
        (d / n).sqrt()
    }

    #[test]
    fn the_forward_pass_matches_the_scalar_reference() {
        let dir = TempDir::new("zm-a");
        let (mini, m) = model(&dir, false, StreamMode::Resident, cpu_residency());
        let got = run(&m, &IDS);
        let want = mini.forward(&IDS);
        assert_eq!(got.len(), IDS.len() * 64);
        let err = rel(&got, &want);
        assert!(err < 1e-4, "기준 구현과의 상대오차 {err}");
        // 모든 층 종류를 지났는지: 선형 3 + 전체 1
        assert_eq!(m.config().layer_kinds.iter().filter(|k| **k == LayerKind::FullAttention).count(), 1);
    }

    #[test]
    fn a_quantized_file_stays_within_quantization_error() {
        // 이득이 낮은 모델: 부분 출력이 잔차보다 작아서 8비트 양자화 오차가 증폭되지 않습니다.
        let dir = TempDir::new("zm-b");
        let mini = MiniZoro::with_gain(0.3);
        let file = mini.write(&dir, true).unwrap();
        let m = ZoroModel::new(file, cpu_residency(), StreamMode::Resident).unwrap();
        let err = rel(&run(&m, &IDS), &mini.forward(&IDS));
        assert!(err < 0.03, "q8 파일의 상대오차 {err}");
    }

    /// 진단: 텐서 종류별로 Q8_0 을 하나씩만 켜고 F32 기준과의 오차를 봅니다.
    #[test]
    #[ignore]
    fn which_tensor_group_dominates_the_quantization_error() {
        let dir = TempDir::new("zm-diag");
        let mini = MiniZoro::new();
        let want = mini.forward(&IDS);
        type Select<'a> = &'a dyn Fn(&str) -> bool;
        let groups: [(&str, Select); 8] = [
            ("임베딩", &|n| n == "token_embd.weight"),
            ("ffn", &|n| n.contains("ffn_")),
            ("attn_qkv", &|n| n.contains("attn_qkv")),
            ("attn_gate(z)", &|n| n.contains("attn_gate")),
            ("ssm_out", &|n| n.contains("ssm_out")),
            ("attn_q/k/v", &|n| n.contains("attn_q.") || n.contains("attn_k.") || n.contains("attn_v.")),
            ("attn_output", &|n| n.contains("attn_output")),
            ("전부", &|_| true),
        ];
        for (n, (label, select)) in groups.into_iter().enumerate() {
            let file = mini.write_with(&dir, &format!("diag-{n}"), select).unwrap();
            let m = ZoroModel::new(file, cpu_residency(), StreamMode::Resident).unwrap();
            println!("{label:<14} 상대오차 {:.4}", rel(&run(&m, &IDS), &want));
        }
    }

    #[test]
    fn the_output_depends_on_the_tokens_and_only_on_earlier_tokens() {
        let dir = TempDir::new("zm-c");
        let (_mini, m) = model(&dir, false, StreamMode::Resident, cpu_residency());
        let a = run(&m, &IDS);
        let mut ids2 = IDS;
        ids2[IDS.len() - 1] = 200;
        let b = run(&m, &ids2);
        let h = 64;
        // 마지막 토큰만 바꾸면 앞 토큰들의 은닉 상태는 그대로입니다 (인과성은 선형 주의 · 전체 주의 모두에서)
        assert_eq!(a[..h * (IDS.len() - 1)], b[..h * (IDS.len() - 1)]);
        assert!(rel(&a[h * (IDS.len() - 1)..], &b[h * (IDS.len() - 1)..]) > 1e-2);
    }

    #[test]
    fn streaming_gives_the_same_numbers_and_never_holds_more_than_one_chunk() {
        let dir_a = TempDir::new("zm-d1");
        let dir_b = TempDir::new("zm-d2");
        let (_m1, resident) = model(&dir_a, true, StreamMode::Resident, cpu_residency());
        let stream_res = cpu_residency();
        let (_m2, stream) = model(&dir_b, true, StreamMode::Chunks(1), stream_res.clone());
        assert_eq!(resident.plan().len(), 1);
        assert_eq!(stream.plan().len(), 4);
        assert!(stream.plan()[3].groups.contains(&Group::Output), "출력 정규화는 마지막 구간에 붙습니다");
        assert!(!stream.plan()[0].groups.contains(&Group::Output));

        assert_eq!(run(&resident, &IDS), run(&stream, &IDS), "같은 연산이므로 값이 같아야 합니다");

        let res = stream_res.lock().unwrap();
        let one_block = (0..4)
            .map(|i| estimate_group_bytes(&stream.spec().file, Group::blocks(i..i + 1), &Device::Cpu).unwrap())
            .max()
            .unwrap();
        // 어느 시점에도 상주량이 (가장 큰 블록 하나 + 출력) 을 넘지 않았는지 기록으로 되짚습니다.
        let mut live = 0u64;
        let mut peak = 0u64;
        for t in res.history() {
            match t.action {
                Action::Load => live += t.bytes,
                Action::Release => live -= t.bytes,
            }
            peak = peak.max(live);
        }
        let output = estimate_group_bytes(&stream.spec().file, Group::Output, &Device::Cpu).unwrap();
        assert!(peak <= one_block + output, "최대 상주 {peak} > 가장 큰 블록 {one_block} + 출력 {output}");
    }

    #[test]
    fn a_tight_budget_makes_the_auto_plan_stream_and_a_roomy_one_stays_resident() {
        let dir = TempDir::new("zm-e");
        let mini = MiniZoro::new();
        let file = mini.write(&dir, true).unwrap();
        let one = estimate_group_bytes(&file, Group::blocks(0..1), &Device::Cpu).unwrap();
        let out = estimate_group_bytes(&file, Group::Output, &Device::Cpu).unwrap();
        let roomy = Residency::shared(Device::Cpu, Budget::bytes(one * 8));
        assert_eq!(ZoroModel::new(Arc::clone(&file), roomy, StreamMode::Auto).unwrap().plan().len(), 1);
        let tight = Residency::shared(Device::Cpu, Budget::bytes(one * 2 + out));
        let m = ZoroModel::new(Arc::clone(&file), tight.clone(), StreamMode::Auto).unwrap();
        assert!(m.plan().len() > 1, "{:?}", m.plan());
        // 빠듯한 예산에서도 끝까지 계산되고, 상주량은 예산을 넘지 않습니다.
        let got = run(&m, &IDS);
        assert!(got.iter().all(|v| v.is_finite()));
        let res = tight.lock().unwrap();
        assert!(res.resident_bytes() <= one * 2 + out);
        // 한 블록도 못 올리는 예산이면 오류입니다.
        let hopeless = Residency::shared(Device::Cpu, Budget::bytes(one / 2));
        let m = ZoroModel::new(file, hopeless, StreamMode::Auto).unwrap();
        assert!(m.hidden_states(&IDS).is_err());
    }

    #[test]
    fn only_the_needed_weights_are_ever_loaded() {
        // 비전 타워가 든 파일이어도 텍스트 판단은 비전도 임베딩 표도 올리지 않습니다.
        let dir = TempDir::new("zm-f");
        let mini = MiniZoro::new();
        let plain = mini.write(&dir, true).unwrap();
        let mut tensors = Vec::new();
        for n in plain.names_with_prefix("") {
            let t = plain.read_dense(&n, &Device::Cpu, candle_core::DType::F32).unwrap();
            let dtype = plain.info(&n).unwrap().dtype;
            tensors.push((n, t, dtype));
        }
        tensors.push((
            "model.visual.blocks.0.attn.qkv.weight".into(),
            Tensor::zeros((96, 32), candle_core::DType::F32, &Device::Cpu).unwrap(),
            GgmlDType::BF16,
        ));
        let path = dir.0.join("with-vision.gguf");
        let meta: Vec<(&str, Value)> = vec![
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
        write_gguf(&path, &meta, &tensors).unwrap();
        let file = Arc::new(WeightFile::open(&path).unwrap());
        assert!(file.names_with_prefix("model.visual.").len() == 1);

        let res = cpu_residency();
        let m = ZoroModel::new(file, res.clone(), StreamMode::Resident).unwrap();
        run(&m, &IDS);
        let res = res.lock().unwrap();
        let loaded: Vec<Group> = res.history().iter().filter(|t| t.action == Action::Load).map(|t| t.group).collect();
        assert!(!loaded.contains(&Group::Vision), "비전은 올라오면 안 됩니다: {loaded:?}");
        assert!(!loaded.contains(&Group::Embedding), "임베딩 표 전체는 올라오면 안 됩니다: {loaded:?}");
        assert_eq!(res.resident_groups(), vec![Group::blocks(0..4), Group::Output]);
    }

    #[test]
    fn another_model_in_between_unloads_zoro_and_the_next_run_reloads_it() {
        let dir = TempDir::new("zm-g");
        let res = cpu_residency();
        let (_mini, m) = model(&dir, true, StreamMode::Resident, res.clone());
        let before = run(&m, &IDS);

        // 다른 모델(여기서는 비전 가중치만 든 작은 gocr 모양 파일)이 장치를 쓰러 옵니다.
        let path = dir.0.join("other.gguf");
        write_gguf(
            &path,
            &[("general.architecture", Value::String("gocr".into()))],
            &[(
                "model.visual.blocks.0.attn.qkv.weight".into(),
                Tensor::zeros((96, 32), candle_core::DType::F32, &Device::Cpu).unwrap(),
                GgmlDType::BF16,
            )],
        )
        .unwrap();
        let other = ModelSpec::new(ModelId::Gocr, Arc::new(WeightFile::open(&path).unwrap()));
        {
            let mut r = res.lock().unwrap();
            drop(r.enter(&other, &Phase::new("vision", [Group::Vision])).unwrap());
            assert_eq!(r.resident_model(), Some(ModelId::Gocr));
            assert_eq!(r.resident_groups(), vec![Group::Vision], "zoro 의 블록은 하나도 남지 않아야 합니다");
        }
        let after = run(&m, &IDS);
        assert_eq!(before, after, "내렸다 다시 올려도 같은 결과여야 합니다");
        let r = res.lock().unwrap();
        assert_eq!(r.resident_model(), Some(ModelId::Zoro));
        assert!(!r.resident_groups().contains(&Group::Vision));
    }

    #[test]
    fn bad_inputs_and_bad_files_are_rejected() {
        let dir = TempDir::new("zm-h");
        let (_mini, m) = model(&dir, false, StreamMode::Resident, cpu_residency());
        assert!(m.hidden_states(&[]).is_err());
        assert!(m.hidden_states(&[300]).is_err(), "어휘 크기와 같은 번호는 범위 밖");
        assert!(m.hidden_states(&vec![1; MAX_TOKENS + 1]).is_err());

        // 모양이 틀린 파일은 첫 순전파 전에 거부합니다. Windows 는 매핑 중인 파일을 덮어쓰지 못하므로 폴더를 따로 씁니다.
        let other = TempDir::new("zm-h2");
        let mini = MiniZoro::new();
        let file = mini.write(&other, false).unwrap();
        let mut tensors = Vec::new();
        for n in file.names_with_prefix("") {
            let mut t = file.read_dense(&n, &Device::Cpu, candle_core::DType::F32).unwrap();
            if n == "blk.1.ffn_up.weight" {
                // 새로 만든 텐서여야 합니다 (narrow 한 뷰는 quantize 가 전체 저장소를 읽습니다)
                t = Tensor::zeros((64, 64), candle_core::DType::F32, &Device::Cpu).unwrap();
            }
            tensors.push((n, t, GgmlDType::F32));
        }
        let path = dir.0.join("broken.gguf");
        let meta: Vec<(&str, Value)> = vec![
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
        write_gguf(&path, &meta, &tensors).unwrap();
        let broken = Arc::new(WeightFile::open(&path).unwrap());
        let err = ZoroModel::new(broken, cpu_residency(), StreamMode::Resident).err().expect("거부되어야 함");
        assert!(err.to_string().contains("blk.1.ffn_up.weight"), "{err}");
    }

    /// 모든 블록과 출력, 임베딩 표를 올린 핸들
    fn weights(m: &ZoroModel, res: &SharedResidency) -> Vec<Arc<Loaded>> {
        res.lock().unwrap().enter(m.spec(), &m.generation_phase()).unwrap()
    }

    #[test]
    fn reading_in_pieces_equals_reading_everything_at_once() {
        let dir = TempDir::new("zm-state");
        let res = cpu_residency();
        let (mini, m) = model(&dir, false, StreamMode::Resident, res.clone());
        let w = weights(&m, &res);
        let mut state = m.new_state();
        let mut parts = Vec::new();
        let mut at = 0;
        for n in [5usize, 1, 1, 3, 1] {
            parts.push(m.advance(&w[0], &w[1], &Device::Cpu, &mut state, &IDS[at..at + n]).unwrap());
            at += n;
        }
        assert_eq!(state.len(), IDS.len());
        let got = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let err = rel(&got, &mini.forward(&IDS));
        assert!(err < 1e-4, "기준 구현과의 상대오차 {err}");
    }

    /// 표로 쓰는 출력 투영까지 포함한 다음 토큰
    fn next_by_full_forward(m: &ZoroModel, ids: &[u32], table: &[f32]) -> u32 {
        let h = run(m, ids);
        let d = m.config().hidden;
        let last = &h[(ids.len() - 1) * d..];
        let score = |row: usize| table[row * d..(row + 1) * d].iter().zip(last).map(|(a, b)| a * b).sum::<f32>();
        (0..m.config().vocab).max_by(|a, b| score(*a).total_cmp(&score(*b))).unwrap() as u32
    }

    #[test]
    fn generating_with_state_writes_the_same_tokens_as_rereading_every_time() {
        let dir = TempDir::new("zm-gen");
        let res = cpu_residency();
        let (_mini, m) = model(&dir, false, StreamMode::Resident, res.clone());
        let table = m.spec().file.read_dense("token_embd.weight", &Device::Cpu, candle_core::DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let prompt = &IDS[..4];
        let got = m.generate(prompt, &GenerateOptions::new(6, vec![])).unwrap();

        // 토큰마다 처음부터 다시 읽는 느린 방식이 기준입니다. 되풀이로 일찍 멈췄으면 그만큼만 비교합니다.
        let mut ids = prompt.to_vec();
        for &t in &got.tokens {
            let want = next_by_full_forward(&m, &ids, &table);
            assert_eq!(t, want, "{} 번째 토큰", ids.len());
            ids.push(want);
        }
        assert!(!got.tokens.is_empty());
        assert!(got.confidences.iter().all(|c| (0.0..=1.0).contains(c)));
        assert_eq!(got.tokens.len(), got.confidences.len());
    }

    #[test]
    fn a_stop_token_ends_the_text_and_the_table_is_loaded_only_for_writing() {
        let dir = TempDir::new("zm-stop");
        let res = cpu_residency();
        let (_mini, m) = model(&dir, false, StreamMode::Resident, res.clone());
        run(&m, &IDS);
        assert!(!res.lock().unwrap().resident_groups().contains(&Group::Embedding), "판단에는 표가 없습니다");

        let first = m.generate(&IDS[..4], &GenerateOptions::new(1, vec![])).unwrap().tokens[0];
        assert!(res.lock().unwrap().resident_groups().contains(&Group::Embedding), "글쓰기에만 표가 올라옵니다");
        let stopped = m.generate(&IDS[..4], &GenerateOptions::new(5, vec![first])).unwrap();
        assert!(stopped.tokens.is_empty() && stopped.finished);

        run(&m, &IDS);
        assert!(!res.lock().unwrap().resident_groups().contains(&Group::Embedding), "판단으로 돌아가면 표를 내립니다");
        assert!(m.generate(&[], &GenerateOptions::new(1, vec![])).is_err());
        assert!(m.generate(&[300], &GenerateOptions::new(1, vec![])).is_err());
        assert!(m.generate(&IDS, &GenerateOptions::new(MAX_TOKENS, vec![])).is_err());
    }

    /// 다음 토큰을 하나 읽은 은닉 상태. 상태가 같으면 이 값이 같습니다.
    fn next_after(m: &ZoroModel, w: &[Arc<Loaded>], state: &mut ZoroState, id: u32) -> Vec<f32> {
        m.advance(&w[0], &w[1], &Device::Cpu, state, &[id]).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
    }

    #[test]
    fn a_read_leaves_the_state_alone_until_it_is_committed() {
        let dir = TempDir::new("zm-read");
        let res = cpu_residency();
        let (_mini, m) = model(&dir, false, StreamMode::Resident, res.clone());
        let w = weights(&m, &res);
        let cpu = Device::Cpu;
        let (mut plain, mut drafted) = (m.new_state(), m.new_state());
        m.advance(&w[0], &w[1], &cpu, &mut plain, &IDS[..5]).unwrap();
        m.advance(&w[0], &w[1], &cpu, &mut drafted, &IDS[..5]).unwrap();
        // 초안을 두 번 읽어도 상태는 그대로이고, 같은 상태에서 다시 읽으면 값이 같습니다.
        let draft = |s: &ZoroState| m.read(&w[0], &w[1], &cpu, s, &[7, 299, 299], false).unwrap().0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(draft(&drafted), draft(&drafted));
        assert_eq!(drafted.len(), 5);
        assert_eq!(next_after(&m, &w, &mut drafted, 7), next_after(&m, &w, &mut plain, 7));
    }

    #[test]
    fn a_row_does_not_depend_on_the_masks_after_it() {
        let dir = TempDir::new("zm-mode");
        let res = cpu_residency();
        let (_mini, m) = model(&dir, false, StreamMode::Resident, res.clone());
        let w = weights(&m, &res);
        let cpu = Device::Cpu;
        let mut state = m.new_state();
        m.advance(&w[0], &w[1], &cpu, &mut state, &IDS[..5]).unwrap();
        // 검증과 초안을 한 순전파에 싣는 근거입니다. 뒤에 마스크가 붙어도 앞 행은 그대로입니다.
        let rows = |ids: &[u32]| m.read(&w[0], &w[1], &cpu, &state, ids, false).unwrap().0.narrow(0, 0, 2).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let (bare, masked) = (rows(&[7, 8]), rows(&[7, 8, 299, 299, 299]));
        assert!(rel(&bare, &masked) < 1e-5, "{}", rel(&bare, &masked));
    }

    #[test]
    fn keeping_part_of_a_read_equals_reading_only_that_part() {
        let dir = TempDir::new("zm-keep");
        let res = cpu_residency();
        let (_mini, m) = model(&dir, false, StreamMode::Resident, res.clone());
        let w = weights(&m, &res);
        let cpu = Device::Cpu;
        for keep in 1..=4usize {
            let (mut parted, mut plain) = (m.new_state(), m.new_state());
            m.advance(&w[0], &w[1], &cpu, &mut parted, &IDS[..5]).unwrap();
            m.advance(&w[0], &w[1], &cpu, &mut plain, &IDS[..5]).unwrap();
            let read = m.read(&w[0], &w[1], &cpu, &parted, &IDS[5..9], true).unwrap().1;
            m.commit(&w[0], &cpu, &mut parted, read, keep).unwrap();
            m.advance(&w[0], &w[1], &cpu, &mut plain, &IDS[5..5 + keep]).unwrap();
            assert_eq!(parted.len(), 5 + keep);
            let (a, b) = (next_after(&m, &w, &mut parted, 3), next_after(&m, &w, &mut plain, 3));
            assert!(rel(&a, &b) < 1e-5, "{keep}개만 넣음: {}", rel(&a, &b));
        }
        // 앞부분만 넣을 수 없게 읽은 결과는 잘라 넣지 못합니다. 틀린 상태로 이어 쓰지 않기 위해서입니다.
        let mut state = m.new_state();
        let read = m.read(&w[0], &w[1], &cpu, &state, &IDS[..4], false).unwrap().1;
        assert!(m.commit(&w[0], &cpu, &mut state, read, 2).is_err());
        let read = m.read(&w[0], &w[1], &cpu, &state, &IDS[..4], true).unwrap().1;
        assert!(m.commit(&w[0], &cpu, &mut state, read, 5).is_err(), "읽은 것보다 많이 넣을 수 없습니다");
    }

    /// 마스크 토큰을 정한 작은 모델
    fn drafting_model(dir: &TempDir, res: SharedResidency) -> ZoroModel {
        let (_mini, m) = model(dir, false, StreamMode::Resident, res);
        m.with_diffusion(Some(BlockDiffusion { bd_size: 4, mask_id: 299 })).unwrap()
    }

    #[test]
    fn every_amount_of_speculation_writes_the_same_tokens() {
        let dir = TempDir::new("zm-spec");
        let res = cpu_residency();
        let m = drafting_model(&dir, res.clone());
        assert_eq!(m.default_speculation(), Speculation::NONE, "재지 않은 장치에서는 추측하지 않습니다");
        let want = m.generate(&IDS[..4], &GenerateOptions::new(24, vec![])).unwrap();
        assert!(want.tokens.len() > 8, "비교할 글이 너무 짧습니다: {:?}", want.tokens);
        assert_eq!(want.reads, want.tokens.len());
        assert_eq!(want.drafted, 0);
        for (masks, lookup, width) in [(1, 0, 0), (2, 0, 0), (4, 0, 0), (0, 3, 0), (1, 3, 0), (3, 6, 0), (2, 2, 3)] {
            let speculation = Speculation { masks, lookup, width };
            let got = m.generate(&IDS[..4], &GenerateOptions::new(24, vec![]).with_speculation(speculation)).unwrap();
            assert_eq!(got.tokens, want.tokens, "{speculation:?}");
            assert_eq!(got.looped, want.looped, "{speculation:?}");
            assert!(got.accepted <= got.drafted);
            for (a, b) in got.confidences.iter().zip(&want.confidences) {
                assert!((a - b).abs() < 1e-4, "{speculation:?}: 확률 {a} / {b}");
            }
        }
    }

    #[test]
    fn a_stop_token_ends_a_speculative_text_at_the_same_place() {
        let dir = TempDir::new("zm-spec-stop");
        let res = cpu_residency();
        let m = drafting_model(&dir, res.clone());
        let free = m.generate(&IDS[..4], &GenerateOptions::new(10, vec![])).unwrap();
        let stop = free.tokens[free.tokens.len() / 2];
        let at = free.tokens.iter().position(|t| *t == stop).unwrap();
        let got = m.generate(&IDS[..4], &GenerateOptions::new(10, vec![stop]).with_speculation(Speculation { masks: 3, lookup: 3, width: 0 })).unwrap();
        assert_eq!(got.tokens, free.tokens[..at]);
        assert!(got.finished);
    }

    #[test]
    fn drafting_loads_nothing_more_than_writing() {
        let dir = TempDir::new("zm-spec-res");
        let res = cpu_residency();
        let m = drafting_model(&dir, res.clone());
        m.generate(&IDS[..4], &GenerateOptions::new(6, vec![])).unwrap();
        let resident = res.lock().unwrap().resident_groups();
        let loads = |r: &SharedResidency| r.lock().unwrap().history().iter().filter(|t| t.action == Action::Load).count();
        let before = loads(&res);
        m.generate(&IDS[..4], &GenerateOptions::new(6, vec![]).with_speculation(Speculation { masks: 3, lookup: 3, width: 0 })).unwrap();
        assert_eq!(loads(&res), before, "같은 가중치라 다시 올리지 않습니다");
        assert_eq!(res.lock().unwrap().resident_groups(), resident);
        assert!(!resident.contains(&Group::Vision), "글쓰기에는 비전 타워가 없습니다");
    }

    #[test]
    fn masks_need_the_diffusion_settings_but_lookup_does_not() {
        let dir = TempDir::new("zm-spec-none");
        let (_mini, m) = model(&dir, false, StreamMode::Resident, cpu_residency());
        let masked = GenerateOptions::new(4, vec![]).with_speculation(Speculation { masks: 2, lookup: 0, width: 0 });
        let err = m.generate(&IDS[..4], &masked).expect_err("거부되어야 함");
        assert!(err.to_string().contains("블록 디퓨전"), "{err}");
        let found = GenerateOptions::new(4, vec![]).with_speculation(Speculation { masks: 0, lookup: 3, width: 0 });
        assert!(m.generate(&IDS[..4], &found).is_ok());
        assert!(m.with_diffusion(Some(BlockDiffusion { bd_size: 8, mask_id: 300 })).is_err(), "마스크가 어휘 밖입니다");
    }

    #[test]
    fn repeated_phrases_are_detected() {
        assert!(is_looping(&[1; 12]));
        assert!(is_looping(&[9, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6, 5, 6]));
        assert!(is_looping(&[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]));
        assert!(!is_looping(&[1, 2, 3, 4, 5, 1, 2, 3, 4, 5]), "두 번은 괜찮습니다");
        assert!(!is_looping(&[5, 0, 0, 7, 0, 0, 0]), "500,000 의 000 은 반복이 아닙니다");
        assert!(!is_looping(&[1; 11]));
        assert!(!is_looping(&[1, 2, 3, 4, 5, 6]));
        assert!(!is_looping(&[]));
    }

    #[test]
    fn chunk_ranges_cover_every_block_exactly_once() {
        assert_eq!(chunk_ranges(&[10, 10, 10, 10], 25), vec![0..2, 2..4]);
        assert_eq!(chunk_ranges(&[10, 10, 10, 10], 100), vec![0..4]);
        assert_eq!(chunk_ranges(&[10, 10, 10, 10], 5), vec![0..1, 1..2, 2..3, 3..4], "블록 하나가 한도보다 커도 구간 하나");
        assert_eq!(chunk_ranges(&[], 5), Vec::<Range<usize>>::new());
        for limit in [1u64, 15, 29, 30, 1000] {
            let sizes = [7u64, 9, 30, 4, 4, 12, 1];
            let r = chunk_ranges(&sizes, limit);
            let covered: Vec<usize> = r.iter().flat_map(|x| x.clone()).collect();
            assert_eq!(covered, (0..sizes.len()).collect::<Vec<_>>(), "한도 {limit}");
        }
    }

    #[test]
    fn stream_mode_parses_the_environment_values() {
        assert_eq!(StreamMode::parse("").unwrap(), StreamMode::Auto);
        assert_eq!(StreamMode::parse("AUTO").unwrap(), StreamMode::Auto);
        assert_eq!(StreamMode::parse("0").unwrap(), StreamMode::Resident);
        assert_eq!(StreamMode::parse("resident").unwrap(), StreamMode::Resident);
        assert_eq!(StreamMode::parse("8").unwrap(), StreamMode::Chunks(8));
        assert!(StreamMode::parse("-1").is_err());
        assert!(StreamMode::parse("many").is_err());
    }

    #[test]
    fn prepare_changes_only_the_zero_centered_norms_and_the_decay_rate() {
        let t = |v: Vec<f32>| Tensor::from_vec(v.clone(), v.len(), &Device::Cpu).unwrap();
        let v = |x: Tensor| x.to_vec1::<f32>().unwrap();
        assert_eq!(v(prepare("blk.0.attn_norm.weight", t(vec![0.0, -0.5])).unwrap()), vec![1.0, 0.5]);
        assert_eq!(v(prepare("blk.7.attn_q_norm.weight", t(vec![0.25])).unwrap()), vec![1.25]);
        assert_eq!(v(prepare("output_norm.weight", t(vec![2.0])).unwrap()), vec![3.0]);
        assert_eq!(v(prepare("blk.0.ssm_norm.weight", t(vec![0.9])).unwrap()), vec![0.9], "게이트 정규화는 직접 곱함");
        let a = v(prepare("blk.0.ssm_a", t(vec![0.0, (2.0f32).ln()])).unwrap());
        assert!((a[0] + 1.0).abs() < 1e-6 && (a[1] + 2.0).abs() < 1e-6, "−exp(A_log): {a:?}");
        assert_eq!(v(prepare("blk.0.ssm_dt.bias", t(vec![3.0])).unwrap()), vec![3.0]);
    }

    #[test]
    fn rows_to_host_refuses_broken_numbers() {
        let ok = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        assert_eq!(rows_to_host(&ok, &[1, 0]).unwrap(), vec![vec![3.0, 4.0], vec![1.0, 2.0]]);
        let bad = Tensor::from_vec(vec![1f32, f32::NAN, 3.0, 4.0], (2, 2), &Device::Cpu).unwrap();
        assert!(rows_to_host(&bad, &[0]).is_err());
        assert!(rows_to_host(&bad, &[1]).is_ok(), "고른 행에만 NaN 이 없으면 통과");
    }
}
