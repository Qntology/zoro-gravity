//! 가중치 상주 관리: 모델은 한 번에 하나만, 그 안에서도 지금 단계가 쓰는 묶음만 장치에 올립니다.
//!
//! logis-center 의 단계 전환 방식을 따릅니다 (구조만 참고했고 코드는 가져오지 않았습니다).
//! - 쓰는 타워만 올립니다. 질의 인코딩에 비전 인코더가 필요 없으면 올리지 않습니다.
//! - 다음 모델이나 다음 단계의 가중치를 올리기 **전에** 이전 것을 먼저 내립니다. 동시 상주가 없습니다.
//! - 내린 뒤에는 장치 동기화 → 작업 버퍼 반환 → 메모리 풀 반환(`settle`)으로 VRAM 을 실제로 돌려받고,
//!   전후의 장치 여유 메모리를 [`Transition`] 으로 기록합니다. 함께 내리는 묶음은 정리를 한 번만 합니다.
//! - 변환이 없는 일반 텐서는 CPU 를 거치지 않고 매핑에서 바로 장치로 올립니다.
//! - 파일은 미리 읽어 둘 수 있습니다 ([`ModelSpec::announce`]). 장치에는 올리지 않으므로 동시 상주가 아닙니다.
//!
//! 단위는 [`Group`] (비전 타워, 텍스트 블록 구간, 출력 정규화 …) 입니다. 단계([`Phase`])는 "지금 필요한 묶음의 전체 목록"이고,
//! [`Residency::enter`] 가 거기에 없는 묶음은 내리고 없는 묶음만 올립니다. 모델이 바뀌면 이전 모델을 통째로 내립니다.
//!
//! 쓰는 쪽이 핸들(`Arc<Loaded>`)을 쥔 채로는 묶음을 내릴 수 없습니다. 내렸다고 기록했는데 메모리가 남는 일을 막기 위해서입니다.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Result};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Module, Tensor};
use serde_json::json;

use super::accel;
use super::trace;
use super::weights::{TensorMeta, WeightFile};

/// 장치에 올릴 가중치 합계의 상한(MB)을 직접 정하는 환경 변수
pub const BUDGET_ENV: &str = "ZOROGRAVITY_VRAM_BUDGET_MB";
/// 활성값·작업 버퍼용으로 비워 둘 장치 메모리(MB). 상한을 직접 정하지 않을 때만 쓰며 기본 1024
pub const RESERVE_ENV: &str = "ZOROGRAVITY_VRAM_RESERVE_MB";
const DEFAULT_RESERVE_MB: u64 = 1024;
const HISTORY_LIMIT: usize = 512;
/// 묶음을 올리기 전에 장치에 남아 있어야 하는 여유 (MB). 포크의 Vulkan 가중치 할당이 512 MB 를 따로 비워 두기 때문에 그보다 조금 크게 잡습니다.
pub const HEADROOM_ENV: &str = "ZOROGRAVITY_VRAM_HEADROOM_MB";
const DEFAULT_HEADROOM_MB: u64 = 640;
/// siglip2 토큰 임베딩 표. [`Group::Embedding`] 과 같은 이유로 올리지 않고 필요한 행만 파일에서 읽습니다.
pub const TEXT_TOKEN_TABLE: &str = "text_model.embeddings.token_embedding.weight";
/// 호스트에 남은 가중치가 있어도 계속 진행하도록 허용하는 환경 변수 (`1`). 수치가 실행마다 달라질 수 있어 진단용으로만 씁니다.
pub const ALLOW_HOST_ENV: &str = "ZOROGRAVITY_ALLOW_HOST_WEIGHTS";

fn env_mb(key: &str) -> Option<u64> {
    std::env::var(key).ok().and_then(|v| v.trim().parse::<u64>().ok())
}

/// 가중치와 KV 캐시를 잡은 뒤에도 장치에 남겨 둘 여유
pub fn headroom_bytes() -> u64 {
    env_mb(HEADROOM_ENV).unwrap_or(DEFAULT_HEADROOM_MB) << 20
}

fn allow_host_weights() -> bool {
    std::env::var(ALLOW_HOST_ENV).map(|v| v == "1").unwrap_or(false)
}

/// 한 번에 하나만 올라올 수 있는 모델
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ModelId {
    Zoro,
    Gocr,
    Siglip2,
}

impl ModelId {
    pub fn name(self) -> &'static str {
        match self {
            Self::Zoro => "zoro",
            Self::Gocr => "gocr",
            Self::Siglip2 => "siglip2",
        }
    }

    pub fn from_architecture(arch: &str) -> Result<Self> {
        match arch {
            "zoro" => Ok(Self::Zoro),
            "gocr" => Ok(Self::Gocr),
            "siglip2" => Ok(Self::Siglip2),
            other => bail!("알 수 없는 모델 구조: {other}"),
        }
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// 함께 올리고 함께 내리는 가중치 묶음
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Group {
    /// 토큰 임베딩 표 전체 (`token_embd.weight`). 몇 행만 필요하면 올리지 말고 [`WeightFile::gather_rows`] 를 쓰세요.
    Embedding,
    /// 텍스트 블록 `[start, end)` (`blk.N.*`)
    Blocks { start: u32, end: u32 },
    /// 출력 정규화와, 있으면 출력 투영 (`output_norm.weight`, `output.weight`)
    Output,
    /// 비전 타워 (`model.visual.*`, `vision_model.*`). 이미지를 읽을 때만 필요합니다.
    Vision,
    /// siglip2 텍스트 타워 (`text_model.*`, 토큰 임베딩 표 제외). 앵커 문구를 새로 인코딩할 때만 필요합니다.
    Text,
}

impl Group {
    pub fn blocks(range: Range<usize>) -> Self {
        Self::Blocks { start: range.start as u32, end: range.end as u32 }
    }

    /// 이 묶음에 속하는 텐서 이름 (정렬됨)
    pub fn tensor_names(&self, file: &WeightFile) -> Vec<String> {
        match *self {
            Self::Embedding => ["token_embd.weight"].iter().filter(|n| file.has(n)).map(|n| n.to_string()).collect(),
            Self::Output => {
                ["output_norm.weight", "output.weight"].iter().filter(|n| file.has(n)).map(|n| n.to_string()).collect()
            }
            Self::Vision => {
                let mut names = file.names_with_prefix("model.visual.");
                names.extend(file.names_with_prefix("vision_model."));
                names
            }
            Self::Text => {
                let mut names = file.names_with_prefix("text_model.");
                names.retain(|n| n != TEXT_TOKEN_TABLE);
                names
            }
            Self::Blocks { start, end } => (start..end).flat_map(|i| file.names_with_prefix(&format!("blk.{i}."))).collect(),
        }
    }
}

impl fmt::Display for Group {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embedding => f.write_str("임베딩"),
            Self::Output => f.write_str("출력"),
            Self::Vision => f.write_str("비전"),
            Self::Text => f.write_str("텍스트 타워"),
            Self::Blocks { start, end } => write!(f, "블록 {start}..{end}"),
        }
    }
}

/// 올린 텐서 하나
pub enum Weight {
    /// 양자화 선형 가중치 (y = x Wᵀ)
    Quant(QMatMul),
    /// 일반 텐서 (정규화·컨볼루션·양자화하지 않은 선형 가중치 …)
    Dense(Tensor),
}

/// 올라온 묶음. 쓰는 동안 `Arc` 핸들을 쥐고, 다 쓰면 놓아야 내릴 수 있습니다.
pub struct Loaded {
    group: Group,
    device_bytes: u64,
    file_bytes: u64,
    weights: HashMap<String, Weight>,
}

impl Loaded {
    pub fn group(&self) -> Group {
        self.group
    }

    /// 장치에서 차지하는 바이트 수 (일반 텐서는 장치 dtype 기준)
    pub fn device_bytes(&self) -> u64 {
        self.device_bytes
    }

    /// 파일에서 차지하던 바이트 수
    pub fn file_bytes(&self) -> u64 {
        self.file_bytes
    }

    pub fn len(&self) -> usize {
        self.weights.len()
    }

    pub fn is_empty(&self) -> bool {
        self.weights.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.weights.keys().map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.weights.contains_key(name)
    }

    pub fn get(&self, name: &str) -> Result<&Weight> {
        self.weights.get(name).ok_or_else(|| anyhow!("{} 묶음에 {name} 텐서가 올라와 있지 않습니다", self.group))
    }

    /// 일반 텐서
    pub fn dense(&self, name: &str) -> Result<&Tensor> {
        match self.get(name)? {
            Weight::Dense(t) => Ok(t),
            Weight::Quant(_) => bail!("{name} 은 양자화 가중치라 일반 텐서로 쓸 수 없습니다"),
        }
    }

    /// `y = x Wᵀ`. 양자화 가중치와 일반 2차원 가중치 모두 받습니다. 일반 가중치가 BF16 이면(ROCm) 행렬곱만 그 dtype 으로 합니다.
    pub fn linear(&self, name: &str, x: &Tensor) -> Result<Tensor> {
        match self.get(name)? {
            // CUDA 의 기본 곱은 활성값을 8비트로 줄입니다. 판단 스케일을 맞춘 ROCm · Vulkan 과 같은 값이 나오도록 정확한 곱을 씁니다.
            Weight::Quant(m) if x.device().is_cuda() => Ok(m.forward_exact(x)?),
            Weight::Quant(m) => Ok(m.forward(x)?),
            Weight::Dense(w) if w.rank() == 2 => dense_linear(x, w),
            Weight::Dense(w) => bail!("{name}: 선형 가중치가 아닙니다 (모양 {:?})", w.dims()),
        }
    }
}

/// `y = x Wᵀ` (일반 가중치). `x` 는 `[…, 열]` 이고, 행렬곱만 가중치의 dtype 으로 합니다.
pub fn dense_linear(x: &Tensor, w: &Tensor) -> Result<Tensor> {
    let mut dims = x.dims().to_vec();
    let last = dims.last_mut().ok_or_else(|| anyhow!("스칼라에는 선형 변환을 할 수 없습니다"))?;
    let cols = *last;
    *last = w.dim(0)?;
    let y = x.reshape(((), cols))?.to_dtype(w.dtype())?.matmul(&w.t()?)?;
    Ok(y.to_dtype(x.dtype())?.reshape(dims)?)
}

/// 일반 텐서를 올릴 때 한 번 적용하는 변환 (예: 정규화 가중치에 1 을 더함). 이름과 텐서를 받습니다.
pub type PrepareFn = fn(&str, Tensor) -> candle_core::Result<Tensor>;

/// 올릴 때의 변환과 그 대상. 대상이 아닌 텐서는 CPU 를 거치지 않고 바로 장치로 올립니다.
#[derive(Clone, Copy)]
pub struct Prepare {
    pub touches: fn(&str) -> bool,
    pub apply: PrepareFn,
}

/// 어느 모델의 어느 파일을 쓰는지. 같은 모델이라도 파일이 다르면 다른 상주로 칩니다.
#[derive(Clone)]
pub struct ModelSpec {
    pub id: ModelId,
    pub file: Arc<WeightFile>,
    /// 비전 타워를 따로 둔 파일 (`mmproj-*.gguf`). 없으면 비전 타워도 `file` 에 있습니다.
    pub mmproj: Option<Arc<WeightFile>>,
    pub prepare: Option<Prepare>,
}

impl ModelSpec {
    pub fn new(id: ModelId, file: Arc<WeightFile>) -> Self {
        Self { id, file, mmproj: None, prepare: None }
    }

    pub fn with_prepare(mut self, prepare: Prepare) -> Self {
        self.prepare = Some(prepare);
        self
    }

    pub fn with_mmproj(mut self, mmproj: Arc<WeightFile>) -> Self {
        self.mmproj = Some(mmproj);
        self
    }

    /// 이 묶음의 텐서가 있는 파일
    pub fn file_of(&self, group: Group) -> &Arc<WeightFile> {
        match (&self.mmproj, group) {
            (Some(mmproj), Group::Vision) => mmproj,
            _ => &self.file,
        }
    }

    /// 곧 쓸 단계를 알립니다. 파일을 배경에서 미리 읽을 뿐 장치에는 아무것도 올리지 않습니다.
    pub fn announce(&self, phase: &Phase) {
        for group in &phase.groups {
            let file = self.file_of(*group);
            file.warm(group.tensor_names(file));
        }
    }
}

/// 한 단계에 필요한 묶음의 전체 목록
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phase {
    pub name: String,
    pub groups: Vec<Group>,
}

impl Phase {
    pub fn new(name: impl Into<String>, groups: impl IntoIterator<Item = Group>) -> Self {
        Self { name: name.into(), groups: groups.into_iter().collect() }
    }

    fn validate(&self) -> Result<()> {
        if self.groups.is_empty() {
            bail!("단계 {} 에 묶음이 없습니다", self.name);
        }
        for (i, a) in self.groups.iter().enumerate() {
            for b in &self.groups[i + 1..] {
                let overlap = match (a, b) {
                    (Group::Blocks { start: s1, end: e1 }, Group::Blocks { start: s2, end: e2 }) => s1 < e2 && s2 < e1,
                    _ => a == b,
                };
                if overlap {
                    bail!("단계 {} 의 묶음 {a} 와 {b} 가 겹칩니다", self.name);
                }
            }
        }
        Ok(())
    }
}

/// 장치에 올려 둘 가중치 합계의 상한
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    max_bytes: Option<u64>,
}

impl Budget {
    pub const fn unlimited() -> Self {
        Self { max_bytes: None }
    }

    pub const fn bytes(max: u64) -> Self {
        Self { max_bytes: Some(max) }
    }

    pub fn max_bytes(&self) -> Option<u64> {
        self.max_bytes
    }

    pub fn allows(&self, resident: u64, extra: u64) -> bool {
        self.max_bytes.is_none_or(|m| resident.saturating_add(extra) <= m)
    }

    /// `ZOROGRAVITY_VRAM_BUDGET_MB` 가 있으면 그 값. 없으면 GPU 는 지금 여유 메모리에서 `ZOROGRAVITY_VRAM_RESERVE_MB`(기본 1024)를 뺀 값,
    /// CPU 는 제한 없음. 모델이 아무것도 올라오지 않은 시점에 부르는 것을 전제로 합니다.
    pub fn auto(device: &Device) -> Self {
        if let Some(mb) = env_mb(BUDGET_ENV) {
            return Self::bytes(mb << 20);
        }
        if device.is_cpu() {
            return Self::unlimited();
        }
        let reserve = env_mb(RESERVE_ENV).unwrap_or(DEFAULT_RESERVE_MB) << 20;
        match device.effective_free_memory() {
            Ok(Some(free)) => Self::bytes((free as u64).saturating_sub(reserve)),
            _ => Self::unlimited(),
        }
    }
}

/// 상한을 넘겨서 올릴 수 없을 때의 오류. 호출자가 구간을 줄여 다시 시도할 수 있게 형식을 갖춥니다.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{group} 묶음 {need_mb} MB 를 올리면 가중치 예산 {max_mb} MB 를 넘습니다 (지금 {resident_mb} MB 상주)", need_mb = .need >> 20, max_mb = .max >> 20, resident_mb = .resident >> 20)]
pub struct OverBudget {
    pub group: Group,
    pub need: u64,
    pub resident: u64,
    pub max: u64,
}

/// 장치 여유 메모리가 모자라 묶음을 올릴 수 없을 때의 오류.
/// 모자란 채로 올리면 포크가 일부 가중치를 조용히 호스트 메모리에 두어서, 그 층만 CPU 경로로 계산되고 수치가 실행마다 달라집니다.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{group} 묶음은 장치 메모리 {need_mb} MB 가 필요한데 여유가 {free_mb} MB 뿐입니다 (여유분 {headroom_mb} MB 포함). 같은 GPU 를 쓰는 다른 프로그램을 닫거나 ZOROGRAVITY_ZORO_STREAM_BLOCKS 로 구간을 줄이세요", need_mb = .need >> 20, free_mb = .free >> 20, headroom_mb = .headroom >> 20)]
pub struct DeviceMemoryShort {
    pub group: Group,
    pub need: u64,
    pub free: u64,
    pub headroom: u64,
}

/// 올린 뒤 줄어든 장치 여유가 예상보다 크게 적을 때의 오류 (일부 가중치가 장치에 올라가지 않았다는 뜻)
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{group} 묶음에서 장치에 올라가야 할 {expected_mb} MB 중 {observed_mb} MB 만 올라갔습니다. 일부 가중치가 호스트에 남아 수치가 달라질 수 있어 올리기를 취소했습니다 (진단용으로 ZOROGRAVITY_ALLOW_HOST_WEIGHTS=1 로 허용할 수 있습니다)", expected_mb = .expected >> 20, observed_mb = .observed >> 20)]
pub struct WeightsLeftOnHost {
    pub group: Group,
    pub expected: u64,
    pub observed: u64,
}

/// 올리기 전 검사: 여유(`free`)가 필요량 + 여유분을 담는가. `free` 를 모르면(CPU 등) 통과.
pub fn check_headroom(group: Group, need: u64, free: Option<u64>, headroom: u64) -> std::result::Result<(), DeviceMemoryShort> {
    match free {
        Some(f) if need > 0 && f < need.saturating_add(headroom) => Err(DeviceMemoryShort { group, need, free: f, headroom }),
        _ => Ok(()),
    }
}

/// 올린 뒤 검사: 줄어든 여유(`observed`)가 예상(`expected`)에서 허용 오차 이상 모자라면 true (일부가 장치에 없다는 뜻).
/// 허용 오차는 예상의 10% 와 64 MB 중 큰 값입니다 (다른 프로그램이 동시에 메모리를 돌려주는 정도는 눈감음).
pub fn weights_missing_from_device(expected: u64, observed: u64) -> bool {
    const FLOOR: u64 = 64 << 20;
    if expected < FLOOR {
        return false;
    }
    observed.saturating_add((expected / 10).max(FLOOR)) < expected
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Load,
    Release,
}

/// 올리고 내린 기록. 전후의 장치 여유 메모리로 반환이 실제로 일어났는지 볼 수 있습니다.
#[derive(Debug, Clone)]
pub struct Transition {
    pub action: Action,
    pub model: ModelId,
    pub group: Group,
    pub bytes: u64,
    pub free_before: Option<u64>,
    pub free_after: Option<u64>,
    /// Vulkan 일반 텐서 버퍼의 총량 전 · 후 (다른 장치는 None). 아래 `Residency::tensor_buffer_bytes` 참고
    pub buffers_before: Option<u64>,
    pub buffers_after: Option<u64>,
    pub millis: u64,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub model: Option<ModelId>,
    pub groups: Vec<(Group, u64)>,
    pub resident_bytes: u64,
    pub free_bytes: Option<u64>,
    pub budget_bytes: Option<u64>,
}

struct Current {
    id: ModelId,
    file: Arc<WeightFile>,
    groups: BTreeMap<Group, Arc<Loaded>>,
}

/// 모델 하나, 단계 하나의 가중치만 장치에 두는 관리자
pub struct Residency {
    device: Device,
    budget: Budget,
    current: Option<Current>,
    history: Vec<Transition>,
}

/// 여러 곳(zoro 백본, gocr 엔진)이 한 관리자를 공유합니다. 그래야 "한 번에 한 모델"이 지켜집니다.
pub type SharedResidency = Arc<Mutex<Residency>>;

impl Residency {
    pub fn new(device: Device, budget: Budget) -> Self {
        Self { device, budget, current: None, history: Vec::new() }
    }

    pub fn shared(device: Device, budget: Budget) -> SharedResidency {
        Arc::new(Mutex::new(Self::new(device, budget)))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn budget(&self) -> Budget {
        self.budget
    }

    pub fn resident_model(&self) -> Option<ModelId> {
        self.current.as_ref().map(|c| c.id)
    }

    pub fn resident_groups(&self) -> Vec<Group> {
        self.current.as_ref().map(|c| c.groups.keys().copied().collect()).unwrap_or_default()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.current.as_ref().map(|c| c.groups.values().map(|l| l.device_bytes).sum()).unwrap_or(0)
    }

    pub fn history(&self) -> &[Transition] {
        &self.history
    }

    /// Vulkan 장치가 쥔 일반 텐서 버퍼의 총량 (재사용 풀 포함). 다른 장치는 None.
    /// Vulkan 에서는 일반 텐서(활성값 · F32 정규화 가중치 · 비전 타워)가 호스트에서 보이는 메모리(RAM)에 잡혀서
    /// `free_bytes` 에 나타나지 않고, 양자화(Q8_0) 가중치만 장치 힙에 올라가 `free_bytes` 로 보입니다.
    pub fn tensor_buffer_bytes(&self) -> Option<u64> {
        match &self.device {
            Device::Vulkan(d) => Some(d.allocated_bytes() as u64),
            _ => None,
        }
    }

    /// 장치의 여유 메모리 (CPU 이거나 조회가 안 되면 None)
    pub fn free_bytes(&self) -> Option<u64> {
        match self.device.mem_info() {
            Ok(Some((free, _))) => Some(free as u64),
            _ => None,
        }
    }

    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            model: self.resident_model(),
            groups: self
                .current
                .as_ref()
                .map(|c| c.groups.iter().map(|(g, l)| (*g, l.device_bytes)).collect())
                .unwrap_or_default(),
            resident_bytes: self.resident_bytes(),
            free_bytes: self.free_bytes(),
            budget_bytes: self.budget.max_bytes(),
        }
    }

    /// 이 단계가 쓰는 묶음만 올라온 상태로 만들고, 그 핸들을 `phase.groups` 순서대로 돌려줍니다.
    ///
    /// 1. 다른 모델이 올라와 있으면 통째로 내립니다 (동시 상주 금지).
    /// 2. 이 단계에 없는 묶음을 내립니다.
    /// 3. 아직 없는 묶음을 올립니다. 상한을 넘으면 [`OverBudget`] 오류입니다.
    pub fn enter(&mut self, spec: &ModelSpec, phase: &Phase) -> Result<Vec<Arc<Loaded>>> {
        phase.validate()?;
        self.switch_model(spec)?;
        let stale: Vec<Group> = self
            .resident_groups()
            .into_iter()
            .filter(|g| !phase.groups.contains(g))
            .collect();
        self.release_groups(&stale)?;
        for g in &phase.groups {
            if !self.is_resident(*g) {
                self.load_one(spec, *g)?;
            }
        }
        let cur = self.current.as_ref().ok_or_else(|| anyhow!("모델이 올라와 있지 않습니다"))?;
        Ok(phase.groups.iter().map(|g| Arc::clone(&cur.groups[g])).collect())
    }

    /// 묶음 하나를 내립니다. 쓰는 곳이 핸들을 쥐고 있으면 오류입니다.
    pub fn release(&mut self, group: Group) -> Result<()> {
        if !self.is_resident(group) {
            return Ok(());
        }
        self.release_groups(&[group])
    }

    /// 올라온 것을 모두 내리고 장치 메모리를 돌려받습니다.
    pub fn release_all(&mut self) -> Result<()> {
        let groups = self.resident_groups();
        if groups.is_empty() {
            // 내릴 것이 없어도 정리합니다. 다른 관리자가 남긴 풀과 버퍼 캐시가 여유 메모리 검사를 흐리지 않게 하기 위해서입니다.
            self.settle();
        }
        self.release_groups(&groups)?;
        self.current = None;
        Ok(())
    }

    fn is_resident(&self, group: Group) -> bool {
        self.current.as_ref().is_some_and(|c| c.groups.contains_key(&group))
    }

    fn switch_model(&mut self, spec: &ModelSpec) -> Result<()> {
        // 같은 모델의 같은 파일이면 이미 올라온 가중치를 그대로 씁니다 (파일을 다시 열어 만든 `WeightFile` 도 같은 파일로 봅니다).
        let same = matches!(&self.current, Some(c) if c.id == spec.id && c.file.path() == spec.file.path());
        if !same {
            // 새 모델의 첫 바이트를 읽기 전에 이전 모델을 모두 내립니다.
            self.release_all()?;
            self.current = Some(Current { id: spec.id, file: Arc::clone(&spec.file), groups: BTreeMap::new() });
        }
        Ok(())
    }

    fn load_one(&mut self, spec: &ModelSpec, group: Group) -> Result<()> {
        let need = estimate_group_bytes(spec.file_of(group), group, &self.device)?;
        let resident = self.resident_bytes();
        if !self.budget.allows(resident, need) {
            return Err(OverBudget { group, need, resident, max: self.budget.max_bytes().unwrap_or(u64::MAX) }.into());
        }
        // 올리기 전에 장치에 실제로 자리가 있는지 봅니다. 다른 프로세스가 같은 GPU 를 쓰고 있으면 예산을 잡은 때와 여유가 달라졌을 수 있습니다.
        let on_device = estimate_device_heap_bytes(spec.file_of(group), group, &self.device)?;
        // Metal 은 다 쓴 작업 버퍼를 동기화할 때 돌려줍니다. 먼저 돌려받아야 올리기 전후의 차이가 이 묶음의 크기가 됩니다.
        if self.device.is_metal() {
            let _ = self.device.synchronize_all();
        }
        let (free_before, buffers_before) = (self.free_bytes(), self.tensor_buffer_bytes());
        if !allow_host_weights() {
            check_headroom(group, on_device, free_before, headroom_bytes())?;
        }
        let t0 = Instant::now();
        let loaded = load_group(spec, group, &self.device)?;
        // 올리는 쪽 복사가 끝났음을 보장해야 이후의 여유 메모리 측정이 의미가 있습니다.
        let _ = self.device.synchronize_all();
        let (free_after, buffers_after) = (self.free_bytes(), self.tensor_buffer_bytes());
        if let (Some(before), Some(after)) = (free_before, free_after) {
            let observed = before.saturating_sub(after);
            if weights_missing_from_device(on_device, observed) && !allow_host_weights() {
                // 올라간 만큼 되돌려 놓고 알립니다.
                drop(loaded);
                self.settle();
                return Err(WeightsLeftOnHost { group, expected: on_device, observed }.into());
            }
        }
        self.record(Transition {
            action: Action::Load,
            model: spec.id,
            group,
            bytes: loaded.device_bytes,
            free_before,
            free_after,
            buffers_before,
            buffers_after,
            millis: t0.elapsed().as_millis() as u64,
        });
        let cur = self.current.as_mut().ok_or_else(|| anyhow!("모델이 올라와 있지 않습니다"))?;
        cur.groups.insert(group, Arc::new(loaded));
        Ok(())
    }

    /// 묶음들을 함께 내립니다. 정리는 전체에 한 번만 하고, 쓰는 곳이 남은 묶음이 하나라도 있으면 아무것도 내리지 않습니다.
    /// 기록의 여유 메모리와 시간은 함께 내린 묶음 전체의 값입니다 (시간은 묶음 수로 나눕니다).
    fn release_groups(&mut self, groups: &[Group]) -> Result<()> {
        if groups.is_empty() {
            return Ok(());
        }
        let (free_before, buffers_before) = (self.free_bytes(), self.tensor_buffer_bytes());
        let t0 = Instant::now();
        let cur = self.current.as_mut().ok_or_else(|| anyhow!("모델이 올라와 있지 않습니다"))?;
        for group in groups {
            let loaded = cur.groups.get(group).ok_or_else(|| anyhow!("{group} 묶음이 올라와 있지 않습니다"))?;
            let holders = Arc::strong_count(loaded) - 1;
            if holders > 0 {
                bail!("{group} 묶음을 아직 쓰는 곳이 있어 내릴 수 없습니다 (핸들 {holders}개). 핸들을 놓은 뒤에 내리세요");
            }
        }
        let model = cur.id;
        let released: Vec<(Group, u64)> = groups.iter().filter_map(|g| cur.groups.remove(g).map(|l| (*g, l.device_bytes))).collect();
        self.settle();
        let (free_after, buffers_after) = (self.free_bytes(), self.tensor_buffer_bytes());
        let millis = t0.elapsed().as_millis() as u64 / released.len().max(1) as u64;
        for (group, bytes) in released {
            self.record(Transition { action: Action::Release, model, group, bytes, free_before, free_after, buffers_before, buffers_after, millis });
        }
        Ok(())
    }

    /// 동기화 → 작업 버퍼 반환 → 메모리 풀 반환 (logis-center 의 `settle_gpu_memory` 와 같은 순서)
    fn settle(&self) {
        if self.device.is_cpu() {
            return;
        }
        let _ = self.device.bind_to_thread();
        if let Err(e) = self.device.synchronize_all() {
            eprintln!("[ZOROGRAVITY-MEM] 장치 동기화 실패: {e}");
        }
        if let Err(e) = self.device.release_workspaces() {
            eprintln!("[ZOROGRAVITY-MEM] 작업 버퍼 반환 실패: {e}");
        }
        if let Err(e) = self.device.trim_memory_pool(0) {
            eprintln!("[ZOROGRAVITY-MEM] 메모리 풀 반환 실패: {e}");
        }
    }

    fn record(&mut self, t: Transition) {
        let step = match t.action {
            Action::Load => "올림",
            Action::Release => "내림",
        };
        trace::record(t.model.name(), step, t.millis as f64, json!({ "group": t.group.to_string(), "bytes": t.bytes, "free_after": t.free_after }));
        if self.history.len() >= HISTORY_LIMIT {
            self.history.drain(..HISTORY_LIMIT / 2);
        }
        self.history.push(t);
    }
}

/// 텐서를 장치에 어떤 모양으로 둘지
enum Plan {
    Quant,
    Dense(DType),
}

fn plan_tensor(meta: &TensorMeta, device: &Device) -> Plan {
    if meta.is_float() {
        // F32 로 저장된 것은 그대로, BF16 · F16 으로 저장된 것은 장치 dtype 정책(ROCm 은 BF16, 그 밖은 F32)을 따릅니다.
        if meta.dtype == candle_core::quantized::GgmlDType::F32 {
            Plan::Dense(DType::F32)
        } else {
            Plan::Dense(accel::model_dtype(device))
        }
    } else if meta.dims.len() == 2 {
        Plan::Quant
    } else {
        Plan::Dense(DType::F32)
    }
}

fn device_bytes_of(meta: &TensorMeta, device: &Device) -> u64 {
    match plan_tensor(meta, device) {
        Plan::Quant => meta.bytes,
        Plan::Dense(dtype) => (meta.elems() * dtype.size_in_bytes()) as u64,
    }
}

/// 묶음이 **장치 힙**에 차지할 바이트 수 (Vulkan 은 일반 텐서가 호스트 메모리라 제외, Q8_0 은 블록마다 스케일을 F32 로 펴서 34 → 36 바이트).
/// 장치 여유 메모리(`mem_info`)와 비교하는 값입니다.
pub fn estimate_device_heap_bytes(file: &WeightFile, group: Group, device: &Device) -> Result<u64> {
    if device.is_cpu() {
        return Ok(0);
    }
    let mut total = 0u64;
    for n in group.tensor_names(file) {
        let meta = file.info(&n)?;
        total += match (plan_tensor(&meta, device), device) {
            (Plan::Quant, Device::Vulkan(_)) => {
                if meta.dtype == candle_core::quantized::GgmlDType::Q8_0 { meta.bytes / 34 * 36 } else { 0 }
            }
            (Plan::Quant, _) => meta.bytes,
            (Plan::Dense(_), Device::Vulkan(_)) => 0,
            (Plan::Dense(dtype), _) => (meta.elems() * dtype.size_in_bytes()) as u64,
        };
    }
    Ok(total)
}

/// 묶음을 `device` 에 올렸을 때의 바이트 수를 파일 목차만으로 계산합니다 (아무것도 읽지 않음).
pub fn estimate_group_bytes(file: &WeightFile, group: Group, device: &Device) -> Result<u64> {
    let names = group.tensor_names(file);
    if names.is_empty() {
        bail!("{group} 에 해당하는 텐서가 파일에 없습니다");
    }
    let mut total = 0u64;
    for n in names {
        total += device_bytes_of(&file.info(&n)?, device);
    }
    Ok(total)
}

fn load_group(spec: &ModelSpec, group: Group, device: &Device) -> Result<Loaded> {
    let file = spec.file_of(group);
    let names = group.tensor_names(file);
    if names.is_empty() {
        bail!("{group} 에 해당하는 텐서가 파일에 없습니다");
    }
    file.warm(names.clone());
    let mut weights = HashMap::with_capacity(names.len());
    let (mut device_bytes, mut file_bytes) = (0u64, 0u64);
    for name in names {
        let meta = file.info(&name)?;
        file_bytes += meta.bytes;
        device_bytes += device_bytes_of(&meta, device);
        let weight = match plan_tensor(&meta, device) {
            Plan::Quant => Weight::Quant(QMatMul::from_qtensor(file.read_qtensor(&name, device)?)?),
            // 변환은 CPU 에서 한 번만 하고 결과를 장치로 옮깁니다 (장치에서 작은 연산을 여러 번 돌리지 않습니다).
            Plan::Dense(dtype) => match spec.prepare.filter(|p| (p.touches)(&name)) {
                Some(p) => Weight::Dense((p.apply)(&name, file.read_dense(&name, &Device::Cpu, dtype)?)?.to_device(device)?),
                None => Weight::Dense(file.read_dense(&name, device, dtype)?),
            },
        };
        weights.insert(name, weight);
    }
    Ok(Loaded { group, device_bytes, file_bytes, weights })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::weights::testing::{write_gguf, TempDir};
    use crate::ai::verify::det_normal;
    use candle_core::quantized::gguf_file::Value;
    use candle_core::quantized::GgmlDType;

    fn rand(shape: &[usize], seed: u64) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_vec(det_normal(n, seed), shape.to_vec(), &Device::Cpu).unwrap()
    }

    /// 작은 zoro 모양 파일: 블록 4 개, 임베딩, 출력 정규화, 비전 2 개
    fn zoro_like(dir: &TempDir) -> Arc<WeightFile> {
        let mut t: Vec<(String, Tensor, GgmlDType)> = vec![
            ("token_embd.weight".into(), rand(&[64, 64], 1), GgmlDType::Q8_0),
            ("output_norm.weight".into(), rand(&[64], 2), GgmlDType::F32),
            ("model.visual.blocks.0.attn.qkv.weight".into(), rand(&[96, 32], 3), GgmlDType::BF16),
            ("model.visual.patch_embed.proj.weight".into(), rand(&[8, 3, 2, 4], 4), GgmlDType::BF16),
        ];
        for i in 0..4u64 {
            t.push((format!("blk.{i}.ffn_up.weight"), rand(&[128, 64], 10 + i), GgmlDType::Q8_0));
            t.push((format!("blk.{i}.attn_norm.weight"), rand(&[64], 20 + i), GgmlDType::F32));
        }
        let path = dir.0.join("zoro.gguf");
        write_gguf(&path, &[("general.architecture", Value::String("zoro".into()))], &t).unwrap();
        Arc::new(WeightFile::open(&path).unwrap())
    }

    fn gocr_like(dir: &TempDir) -> Arc<WeightFile> {
        let mut t: Vec<(String, Tensor, GgmlDType)> = vec![
            ("token_embd.weight".into(), rand(&[64, 32], 31), GgmlDType::Q8_0),
            ("output.weight".into(), rand(&[64, 32], 32), GgmlDType::Q8_0),
            ("output_norm.weight".into(), rand(&[32], 33), GgmlDType::F32),
            ("model.visual.blocks.0.attn.qkv.weight".into(), rand(&[96, 32], 34), GgmlDType::BF16),
        ];
        for i in 0..2u64 {
            t.push((format!("blk.{i}.ffn_down.weight"), rand(&[32, 64], 40 + i), GgmlDType::Q8_0));
        }
        let path = dir.0.join("gocr.gguf");
        write_gguf(&path, &[("general.architecture", Value::String("gocr".into()))], &t).unwrap();
        Arc::new(WeightFile::open(&path).unwrap())
    }

    fn phase(name: &str, groups: &[Group]) -> Phase {
        Phase::new(name, groups.iter().copied())
    }

    /// `weights_missing_from_device` 가 보는 크기(64 MB 이상)의 묶음을 GPU 에 올리고 내립니다.
    /// 앞 계산의 작업 버퍼가 남아 있어도 올리기가 거부되면 안 되고, 내린 메모리는 돌아와야 합니다.
    #[test]
    #[ignore = "GPU 가 필요합니다"]
    fn device_loading_is_measured_and_releasing_returns_the_memory() -> Result<()> {
        let device = accel::test_gpu();
        let dir = TempDir::new("res-gpu");
        let tensors: Vec<(String, Tensor, GgmlDType)> = (0..2)
            .map(|i| (format!("blk.{i}.ffn_up.weight"), Tensor::zeros((8192, 4096), DType::F32, &Device::Cpu).unwrap(), GgmlDType::F32))
            .collect();
        let path = dir.0.join("big.gguf");
        write_gguf(&path, &[("general.architecture", Value::String("zoro".into()))], &tensors)?;
        let file = Arc::new(WeightFile::open(&path)?);
        let spec = ModelSpec::new(ModelId::Zoro, Arc::clone(&file));
        let mut res = Residency::new(device.clone(), Budget::unlimited());
        res.release_all()?;
        let start = res.free_bytes();

        let front = res.enter(&spec, &phase("앞", &[Group::blocks(0..1)]))?;
        // 가중치와 같은 크기의 결과를 읽고 버립니다. Metal 은 이 버퍼를 다음 동기화 때까지 쥐고 있습니다.
        front[0].dense("blk.0.ffn_up.weight")?.sqr()?.sum_all()?.to_scalar::<f32>()?;
        drop(front);
        drop(res.enter(&spec, &phase("앞뒤", &[Group::blocks(0..1), Group::blocks(1..2)]))?);
        assert_eq!(res.resident_groups(), vec![Group::blocks(0..1), Group::blocks(1..2)]);
        res.release_all()?;

        // Vulkan 의 일반 텐서는 호스트 메모리에 있어서 장치 여유로는 볼 수 없습니다.
        if estimate_device_heap_bytes(&file, Group::blocks(0..1), &device)? > 0 {
            let (before, after) = (start.expect("장치 여유"), res.free_bytes().expect("장치 여유"));
            assert!(after + (16 << 20) >= before, "내린 메모리가 돌아오지 않았습니다: {} → {} MB", before >> 20, after >> 20);
        }
        println!("[DEVICE-CHECK] {} 상주 관리: 128 MiB 묶음 2개를 올리고 내림, 여유 {:?} → {:?} MB", accel::backend_name(&device), start.map(|b| b >> 20), res.free_bytes().map(|b| b >> 20));
        Ok(())
    }

    #[test]
    fn the_vision_tower_comes_from_the_mmproj_file() -> Result<()> {
        let dir = TempDir::new("res-mmproj");
        let text = dir.0.join("zoro-q8.gguf");
        let arch = [("general.architecture", Value::String("zoro".into()))];
        write_gguf(
            &text,
            &arch,
            &[
                ("blk.0.ffn_up.weight".into(), rand(&[128, 64], 1), GgmlDType::Q8_0),
                ("output_norm.weight".into(), rand(&[64], 2), GgmlDType::F32),
            ],
        )?;
        let vision = dir.0.join("mmproj-zoro-bf16.gguf");
        write_gguf(&vision, &arch, &[("model.visual.blocks.0.attn.qkv.weight".into(), rand(&[96, 32], 3), GgmlDType::BF16)])?;
        let spec = ModelSpec::new(ModelId::Zoro, Arc::new(WeightFile::open(&text)?));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());

        let err = res.enter(&spec, &phase("비전", &[Group::Vision])).err().expect("mmproj 없이는 비전이 없습니다");
        assert!(err.to_string().contains("비전"), "{err}");

        let spec = spec.with_mmproj(Arc::new(WeightFile::open(&vision)?));
        let h = res.enter(&spec, &phase("언어", &[Group::blocks(0..1), Group::Output]))?;
        assert!(h[0].has("blk.0.ffn_up.weight"));
        drop(h);
        let h = res.enter(&spec, &phase("비전", &[Group::Vision]))?;
        assert!(h[0].has("model.visual.blocks.0.attn.qkv.weight"));
        assert_eq!(h[0].dense("model.visual.blocks.0.attn.qkv.weight")?.dims(), &[96, 32]);
        drop(h);
        assert_eq!(res.resident_groups(), vec![Group::Vision], "언어 묶음은 내려갑니다");
        Ok(())
    }

    #[test]
    fn a_phase_loads_only_its_own_groups() -> Result<()> {
        let dir = TempDir::new("res-a");
        let zoro = ModelSpec::new(ModelId::Zoro, zoro_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let h = res.enter(&zoro, &phase("text", &[Group::blocks(0..2), Group::Output]))?;
        assert_eq!(h.len(), 2);
        assert_eq!(h[0].group(), Group::blocks(0..2));
        assert!(h[0].has("blk.1.ffn_up.weight") && !h[0].has("blk.2.ffn_up.weight"));
        assert!(h[1].has("output_norm.weight"));
        // 쓰지 않는 비전과 임베딩은 올라오지 않았습니다.
        assert_eq!(res.resident_groups(), vec![Group::blocks(0..2), Group::Output]);
        assert!(zoro.file.info("token_embd.weight")?.bytes > 0);
        let want: u64 = h.iter().map(|l| l.device_bytes()).sum();
        assert_eq!(res.resident_bytes(), want);
        Ok(())
    }

    #[test]
    fn device_bytes_follow_the_planned_storage() -> Result<()> {
        let dir = TempDir::new("res-b");
        let file = zoro_like(&dir);
        // 양자화 선형은 파일 크기 그대로, F32 는 4 바이트, BF16 은 CPU 정책(F32)으로 올라옵니다.
        let blocks = estimate_group_bytes(&file, Group::blocks(0..1), &Device::Cpu)?;
        assert_eq!(blocks, (128 * 64 / 32 * 34 + 64 * 4) as u64);
        let vision = estimate_group_bytes(&file, Group::Vision, &Device::Cpu)?;
        assert_eq!(vision, ((96 * 32 + 8 * 3 * 2 * 4) * 4) as u64, "BF16 으로 저장된 비전은 F32 로 올라옵니다");
        assert!(estimate_group_bytes(&file, Group::blocks(9..10), &Device::Cpu).is_err(), "없는 구간");
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let spec = ModelSpec::new(ModelId::Zoro, file);
        let h = res.enter(&spec, &phase("v", &[Group::Vision]))?;
        assert_eq!(h[0].device_bytes(), vision);
        assert!(h[0].file_bytes() < vision);
        Ok(())
    }

    #[test]
    fn entering_the_next_phase_drops_what_it_does_not_need() -> Result<()> {
        // OCR 같은 흐름: 비전만 → 비전을 내리고 언어만
        let dir = TempDir::new("res-c");
        let spec = ModelSpec::new(ModelId::Gocr, gocr_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let vision = phase("vision", &[Group::Vision]);
        let language = phase("language", &[Group::Embedding, Group::blocks(0..2), Group::Output]);

        drop(res.enter(&spec, &vision)?);
        assert_eq!(res.resident_groups(), vec![Group::Vision]);

        drop(res.enter(&spec, &language)?);
        assert_eq!(res.resident_groups(), vec![Group::Embedding, Group::blocks(0..2), Group::Output]);
        assert!(!res.resident_groups().contains(&Group::Vision), "언어 단계에서는 비전이 내려가 있어야 합니다");

        // 기록 순서: 비전을 올림 → 비전을 내림 → 언어를 올림 (내린 뒤에 올립니다)
        let log: Vec<(Action, Group)> = res.history().iter().map(|t| (t.action, t.group)).collect();
        let release_vision = log.iter().position(|e| *e == (Action::Release, Group::Vision)).unwrap();
        let first_language_load = log.iter().position(|e| e.0 == Action::Load && e.1 != Group::Vision).unwrap();
        assert!(release_vision < first_language_load, "{log:?}");
        Ok(())
    }

    #[test]
    fn a_resident_group_is_kept_when_the_next_phase_still_needs_it() -> Result<()> {
        let dir = TempDir::new("res-d");
        let spec = ModelSpec::new(ModelId::Zoro, zoro_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let p = phase("text", &[Group::blocks(0..4), Group::Output]);
        drop(res.enter(&spec, &p)?);
        let loads = res.history().iter().filter(|t| t.action == Action::Load).count();
        drop(res.enter(&spec, &p)?);
        drop(res.enter(&spec, &p)?);
        assert_eq!(res.history().iter().filter(|t| t.action == Action::Load).count(), loads, "다시 올리지 않습니다");
        assert_eq!(res.history().iter().filter(|t| t.action == Action::Release).count(), 0);
        Ok(())
    }

    #[test]
    fn switching_models_releases_the_previous_model_first() -> Result<()> {
        let dir = TempDir::new("res-e");
        let zoro = ModelSpec::new(ModelId::Zoro, zoro_like(&dir));
        let gocr = ModelSpec::new(ModelId::Gocr, gocr_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());

        drop(res.enter(&zoro, &phase("text", &[Group::blocks(0..4), Group::Output]))?);
        assert_eq!(res.resident_model(), Some(ModelId::Zoro));
        let zoro_bytes = res.resident_bytes();
        assert!(zoro_bytes > 0);

        drop(res.enter(&gocr, &phase("vision", &[Group::Vision]))?);
        assert_eq!(res.resident_model(), Some(ModelId::Gocr));
        assert_eq!(res.resident_groups(), vec![Group::Vision], "zoro 의 블록은 하나도 남지 않아야 합니다");

        // 어느 시점에도 두 모델이 함께 올라와 있지 않았는지 기록으로 되짚습니다.
        let mut resident: HashMap<ModelId, u64> = HashMap::new();
        for t in res.history() {
            let e = resident.entry(t.model).or_default();
            match t.action {
                Action::Load => *e += t.bytes,
                Action::Release => *e -= t.bytes,
            }
            let live: Vec<_> = resident.iter().filter(|(_, b)| **b > 0).map(|(m, _)| *m).collect();
            assert!(live.len() <= 1, "동시 상주: {live:?} · 기록 {:?}", res.history());
        }
        Ok(())
    }

    #[test]
    fn a_group_that_is_still_borrowed_cannot_be_released() -> Result<()> {
        let dir = TempDir::new("res-f");
        let zoro = ModelSpec::new(ModelId::Zoro, zoro_like(&dir));
        let gocr = ModelSpec::new(ModelId::Gocr, gocr_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let held = res.enter(&zoro, &phase("text", &[Group::blocks(0..2)]))?;
        // 핸들을 쥔 채 다른 단계로 가면 메모리가 남으므로 거부합니다.
        let err = res.enter(&zoro, &phase("other", &[Group::blocks(2..4)])).err().expect("거부되어야 함");
        assert!(err.to_string().contains("아직 쓰는 곳"), "{err}");
        let err = res.enter(&gocr, &phase("vision", &[Group::Vision])).err().expect("모델 전환도 거부");
        assert!(err.to_string().contains("아직 쓰는 곳"), "{err}");
        drop(held);
        res.enter(&gocr, &phase("vision", &[Group::Vision]))?;
        assert_eq!(res.resident_model(), Some(ModelId::Gocr));
        Ok(())
    }

    #[test]
    fn the_budget_is_enforced_and_the_error_is_typed() -> Result<()> {
        let dir = TempDir::new("res-g");
        let file = zoro_like(&dir);
        let one_block = estimate_group_bytes(&file, Group::blocks(0..1), &Device::Cpu)?;
        let spec = ModelSpec::new(ModelId::Zoro, file);
        let mut res = Residency::new(Device::Cpu, Budget::bytes(one_block * 2));
        drop(res.enter(&spec, &phase("a", &[Group::blocks(0..2)]))?);
        let err = res.enter(&spec, &phase("b", &[Group::blocks(0..3)])).err().expect("예산 초과");
        let over = err.downcast_ref::<OverBudget>().expect("OverBudget 형식이어야 합니다");
        assert_eq!(over.group, Group::blocks(0..3));
        assert_eq!(over.max, one_block * 2);
        // 구간을 나누어 한 구간씩 쓰면 같은 예산에서 모든 블록을 지나갈 수 있습니다.
        for start in 0..4 {
            drop(res.enter(&spec, &phase("stream", &[Group::blocks(start..start + 1)]))?);
            assert!(res.resident_bytes() <= one_block * 2);
        }
        Ok(())
    }

    #[test]
    fn prepare_runs_once_on_dense_tensors_at_load() -> Result<()> {
        fn is_norm(name: &str) -> bool {
            name.ends_with("norm.weight")
        }
        fn plus_one(_: &str, t: Tensor) -> candle_core::Result<Tensor> {
            t.affine(1.0, 1.0)
        }
        let dir = TempDir::new("res-h");
        let file = zoro_like(&dir);
        let raw = file.read_dense("blk.0.attn_norm.weight", &Device::Cpu, DType::F32)?.to_vec1::<f32>()?;
        let spec = ModelSpec::new(ModelId::Zoro, file).with_prepare(Prepare { touches: is_norm, apply: plus_one });
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let h = res.enter(&spec, &phase("text", &[Group::blocks(0..1)]))?;
        let got = h[0].dense("blk.0.attn_norm.weight")?.to_vec1::<f32>()?;
        for (g, r) in got.iter().zip(&raw) {
            assert!((g - (r + 1.0)).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn tensors_outside_the_prepare_target_are_read_as_stored() -> Result<()> {
        fn nothing(_: &str) -> bool {
            false
        }
        fn broken(_: &str, _: Tensor) -> candle_core::Result<Tensor> {
            unreachable!("대상이 아닌 텐서를 변환했습니다")
        }
        let dir = TempDir::new("res-direct");
        let file = zoro_like(&dir);
        let raw = file.read_dense("blk.0.attn_norm.weight", &Device::Cpu, DType::F32)?.to_vec1::<f32>()?;
        let spec = ModelSpec::new(ModelId::Zoro, file).with_prepare(Prepare { touches: nothing, apply: broken });
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let h = res.enter(&spec, &phase("text", &[Group::blocks(0..1)]))?;
        assert_eq!(h[0].dense("blk.0.attn_norm.weight")?.to_vec1::<f32>()?, raw);
        Ok(())
    }

    #[test]
    fn groups_released_together_are_recorded_one_by_one_and_all_or_nothing() -> Result<()> {
        let dir = TempDir::new("res-batch");
        let spec = ModelSpec::new(ModelId::Zoro, zoro_like(&dir));
        let gocr = ModelSpec::new(ModelId::Gocr, gocr_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let held = res.enter(&spec, &phase("a", &[Group::blocks(0..2), Group::Output, Group::Vision]))?;
        let keep = Arc::clone(&held[1]);
        drop(held);
        // 하나라도 쓰는 곳이 있으면 나머지도 내려가지 않습니다.
        let err = res.enter(&gocr, &phase("b", &[Group::Vision])).err().expect("거부되어야 함");
        assert!(err.to_string().contains("아직 쓰는 곳"), "{err}");
        assert_eq!(res.resident_groups().len(), 3, "아무것도 내려가지 않았습니다");
        drop(keep);
        drop(res.enter(&gocr, &phase("b", &[Group::Vision]))?);
        let released: Vec<Group> = res.history().iter().filter(|t| t.action == Action::Release).map(|t| t.group).collect();
        assert_eq!(released.len(), 3, "내린 묶음마다 기록이 남습니다");
        Ok(())
    }

    #[test]
    fn announcing_a_phase_loads_nothing_onto_the_device() -> Result<()> {
        let dir = TempDir::new("res-announce");
        let spec = ModelSpec::new(ModelId::Zoro, zoro_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        spec.announce(&phase("text", &[Group::blocks(0..4), Group::Output]));
        assert!(res.resident_groups().is_empty());
        assert_eq!(res.resident_bytes(), 0);
        assert!(res.history().is_empty());
        drop(res.enter(&spec, &phase("text", &[Group::blocks(0..4), Group::Output]))?);
        assert_eq!(res.resident_groups().len(), 2);
        Ok(())
    }

    #[test]
    fn linear_works_for_quantized_and_plain_weights() -> Result<()> {
        let dir = TempDir::new("res-i");
        let file = zoro_like(&dir);
        let spec = ModelSpec::new(ModelId::Zoro, file.clone());
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        let h = res.enter(&spec, &phase("text", &[Group::blocks(0..1)]))?;
        let x = rand(&[3, 64], 99);
        let y = h[0].linear("blk.0.ffn_up.weight", &x)?;
        assert_eq!(y.dims(), &[3, 128]);
        // 양자화 오차 안에서 F32 로 푼 가중치의 곱과 같아야 합니다.
        let w = file.read_dense("blk.0.ffn_up.weight", &Device::Cpu, DType::F32)?;
        let want = x.matmul(&w.t()?)?;
        let err = crate::ai::verify::rel_err(&y, &want)?;
        assert!(err < 0.03, "{err}");
        assert!(h[0].linear("blk.0.attn_norm.weight", &x).is_err(), "1차원은 선형 가중치가 아님");
        assert!(h[0].linear("nope", &x).is_err());
        Ok(())
    }

    #[test]
    fn loading_is_refused_when_the_device_has_no_room_instead_of_leaving_weights_on_the_host() {
        let g = Group::blocks(0..4);
        let mb = |n: u64| n << 20;
        // 여유 3000 MB, 필요 2400 MB + 여유분 640 MB = 3040 MB → 모자람
        let err = check_headroom(g, mb(2400), Some(mb(3000)), mb(640)).unwrap_err();
        assert_eq!((err.group, err.need, err.free, err.headroom), (g, mb(2400), mb(3000), mb(640)));
        assert!(err.to_string().contains("2400") && err.to_string().contains("3000"), "{err}");
        assert!(check_headroom(g, mb(2400), Some(mb(3040)), mb(640)).is_ok(), "정확히 맞으면 통과");
        // 알 수 없으면(CPU) 통과, 올릴 것이 장치 힙에 없으면(Vulkan 의 일반 텐서) 통과
        assert!(check_headroom(g, mb(2400), None, mb(640)).is_ok());
        assert!(check_headroom(g, 0, Some(0), mb(640)).is_ok());
    }

    #[test]
    fn missing_weights_are_noticed_with_a_tolerance_for_other_programs() {
        let mb = |n: u64| n << 20;
        assert!(!weights_missing_from_device(mb(3600), mb(3850)), "예상보다 더 줄어든 것은 정상 (스케일 · 작업 버퍼)");
        assert!(!weights_missing_from_device(mb(3600), mb(3300)), "10% 이내 차이는 허용");
        assert!(weights_missing_from_device(mb(3600), mb(2800)), "800 MB 가 모자라면 호스트에 남은 것");
        assert!(weights_missing_from_device(mb(3600), 0), "하나도 안 올라갔음");
        assert!(!weights_missing_from_device(mb(10), 0), "작은 묶음은 측정 오차가 커서 판정하지 않음");
        assert!(!weights_missing_from_device(0, 0));
    }

    #[test]
    fn overlapping_or_empty_phases_are_rejected() {
        assert!(phase("empty", &[]).validate().is_err());
        assert!(phase("dup", &[Group::Vision, Group::Vision]).validate().is_err());
        assert!(phase("overlap", &[Group::blocks(0..4), Group::blocks(3..6)]).validate().is_err());
        assert!(phase("ok", &[Group::blocks(0..3), Group::blocks(3..6), Group::Output]).validate().is_ok());
    }

    #[test]
    fn release_all_empties_the_manager() -> Result<()> {
        let dir = TempDir::new("res-j");
        let spec = ModelSpec::new(ModelId::Zoro, zoro_like(&dir));
        let mut res = Residency::new(Device::Cpu, Budget::unlimited());
        drop(res.enter(&spec, &phase("text", &[Group::blocks(0..4), Group::Output]))?);
        res.release_all()?;
        assert_eq!(res.resident_model(), None);
        assert_eq!(res.resident_bytes(), 0);
        assert!(res.resident_groups().is_empty());
        Ok(())
    }
}
