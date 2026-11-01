//! GGUF 에서 필요한 텐서만 읽는 계층.
//!
//! 파일을 메모리에 매핑해 두고, 텐서 데이터는 요청받은 텐서의 바이트 범위만 빌려 줍니다. 읽은 범위만 OS 페이지 캐시에 올라옵니다.
//! 처음 읽는 범위는 배경 스레드가 미리 읽어 둡니다 ([`WeightFile::warm`]). 전환이 페이지 폴트를 기다리지 않게 하려는 것입니다.
//! 중간 버퍼가 없어서 텐서 하나가 파일에서 장치까지 복사 한 번으로 갑니다. logis-center 의 mmap 로더와 같은 방식입니다.
//! 모델 전체를 올리지 않고 단계가 쓰는 가중치만 올리는 방식(`ai::residency`)의 바닥입니다.
//! 토큰 임베딩처럼 큰 표에서 몇 행만 필요하면 [`WeightFile::gather_rows`] 가 그 행의 바이트만 읽습니다
//! (zoro q8 의 임베딩 표는 약 675 MB 인데, 프롬프트 한 개에는 수백 행이면 됩니다).

use std::collections::HashSet;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use candle_core::quantized::ggml_file::qtensor_from_ggml;
use candle_core::quantized::gguf_file::{Content, TensorInfo, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Tensor, WithDType};
use memmap2::Mmap;
use rayon::prelude::*;

/// BF16 을 F32 로 풀 때 스레드 하나가 맡는 원소 수
const CONVERT_CHUNK: usize = 1 << 16;

/// 텐서 하나의 목차 항목
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: GgmlDType,
    pub dims: Vec<usize>,
    /// 파일에서 차지하는 바이트 수
    pub bytes: u64,
}

impl TensorMeta {
    pub fn elems(&self) -> usize {
        self.dims.iter().product()
    }

    /// F32 · F16 · BF16 처럼 블록 양자화가 아닌 형식인가
    pub fn is_float(&self) -> bool {
        is_float_dtype(self.dtype)
    }
}

pub fn is_float_dtype(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
}

/// 미리 읽기에서 한 범위로 합치는 틈의 상한
const PREFETCH_GAP: usize = 1 << 20;
/// 페이지를 건드리는 간격
const PAGE: usize = 4096;
/// 배경 미리 읽기를 끄는 환경 변수 (`0`)
pub const WARM_ENV: &str = "ZOROGRAVITY_WARM";

#[cfg(windows)]
#[repr(C)]
struct MemoryRange {
    start: *mut std::ffi::c_void,
    bytes: usize,
}

#[cfg(windows)]
extern "system" {
    fn GetCurrentProcess() -> *mut std::ffi::c_void;
    fn PrefetchVirtualMemory(process: *mut std::ffi::c_void, entries: usize, ranges: *const MemoryRange, flags: u32) -> i32;
}

#[cfg(windows)]
fn prefetch_range(map: &Mmap, start: usize, bytes: usize) {
    let range = MemoryRange { start: map[start..].as_ptr() as *mut std::ffi::c_void, bytes };
    // 실패해도 페이지 폴트로 읽히므로 결과를 보지 않습니다.
    unsafe {
        PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
    }
}

#[cfg(unix)]
fn prefetch_range(map: &Mmap, start: usize, bytes: usize) {
    let _ = map.advise_range(memmap2::Advice::WillNeed, start, bytes);
}

/// 저장된 바이트를 다른 원소 형식으로 봅니다. GGUF 는 텐서를 32 바이트 경계에 두고 매핑은 페이지 경계에서 시작합니다.
fn as_elements<'a, T: WithDType>(name: &str, raw: &'a [u8]) -> Result<&'a [T]> {
    // 원소 하나도 바꾸지 않고 형식만 다시 봅니다. `T` 는 패딩이 없는 숫자 형식입니다.
    let (head, body, tail) = unsafe { raw.align_to::<T>() };
    if !head.is_empty() || !tail.is_empty() {
        bail!("{name}: 텐서 위치가 원소 경계에 맞지 않습니다");
    }
    Ok(body)
}

/// BF16 은 F32 의 위쪽 16 비트와 같습니다.
fn bf16_to_f32(values: &[half::bf16]) -> Vec<f32> {
    let mut out = vec![0f32; values.len()];
    out.par_chunks_mut(CONVERT_CHUNK).zip(values.par_chunks(CONVERT_CHUNK)).for_each(|(o, v)| {
        o.iter_mut().zip(v).for_each(|(o, v)| *o = f32::from_bits((v.to_bits() as u32) << 16));
    });
    out
}

/// 열린 GGUF 파일. 파일을 매핑만 해 두고, 텐서 데이터는 쓸 때 그 범위만 읽습니다.
pub struct WeightFile {
    path: PathBuf,
    content: Content,
    map: Mmap,
    /// 이미 미리 읽은 범위
    warmed: Mutex<HashSet<(usize, usize)>>,
}

impl std::fmt::Debug for WeightFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WeightFile")
            .field("path", &self.path)
            .field("tensors", &self.content.tensor_infos.len())
            .finish()
    }
}

impl WeightFile {
    /// 파일 머리만 읽고 나머지는 매핑만 합니다.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("GGUF 를 열 수 없습니다: {}", path.display()))?;
        let content = Content::read(&mut BufReader::new(&file))
            .with_context(|| format!("GGUF 머리를 읽을 수 없습니다: {}", path.display()))?;
        // 모델 파일은 실행 중에 바꾸지 않는다고 봅니다. candle 의 safetensors 읽기도 같습니다.
        let map = unsafe { Mmap::map(&file) }.with_context(|| format!("GGUF 를 매핑할 수 없습니다: {}", path.display()))?;
        Ok(Self { path: path.to_path_buf(), content, map, warmed: Mutex::new(HashSet::new()) })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    // ---- 메타데이터 ----

    pub fn has_meta(&self, key: &str) -> bool {
        self.content.metadata.contains_key(key)
    }

    pub fn meta_str(&self, key: &str) -> Result<&str> {
        match self.content.metadata.get(key) {
            Some(Value::String(s)) => Ok(s),
            Some(other) => bail!("GGUF 메타데이터 {key} 가 문자열이 아닙니다: {other:?}"),
            None => bail!("GGUF 메타데이터 {key} 가 없습니다"),
        }
    }

    fn meta_int(&self, key: &str) -> Result<i128> {
        let v = self.content.metadata.get(key).ok_or_else(|| anyhow!("GGUF 메타데이터 {key} 가 없습니다"))?;
        Ok(match v {
            Value::U8(x) => *x as i128,
            Value::I8(x) => *x as i128,
            Value::U16(x) => *x as i128,
            Value::I16(x) => *x as i128,
            Value::U32(x) => *x as i128,
            Value::I32(x) => *x as i128,
            Value::U64(x) => *x as i128,
            Value::I64(x) => *x as i128,
            other => bail!("GGUF 메타데이터 {key} 가 정수가 아닙니다: {other:?}"),
        })
    }

    pub fn meta_usize(&self, key: &str) -> Result<usize> {
        let v = self.meta_int(key)?;
        usize::try_from(v).map_err(|_| anyhow!("GGUF 메타데이터 {key}={v} 는 음수이거나 너무 큽니다"))
    }

    pub fn meta_f64(&self, key: &str) -> Result<f64> {
        match self.content.metadata.get(key) {
            Some(Value::F32(x)) => Ok(*x as f64),
            Some(Value::F64(x)) => Ok(*x),
            Some(other) => bail!("GGUF 메타데이터 {key} 가 실수가 아닙니다: {other:?}"),
            None => bail!("GGUF 메타데이터 {key} 가 없습니다"),
        }
    }

    // ---- 목차 ----

    pub fn has(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    pub fn info(&self, name: &str) -> Result<TensorMeta> {
        let t = self
            .content
            .tensor_infos
            .get(name)
            .ok_or_else(|| anyhow!("GGUF 에 {name} 텐서가 없습니다 ({})", self.path.display()))?;
        let elems = t.shape.elem_count();
        let bytes = (elems / t.ggml_dtype.block_size() * t.ggml_dtype.type_size()) as u64;
        Ok(TensorMeta { name: name.to_string(), dtype: t.ggml_dtype, dims: t.shape.dims().to_vec(), bytes })
    }

    /// 이름이 `prefix` 로 시작하는 텐서 이름 (정렬됨)
    pub fn names_with_prefix(&self, prefix: &str) -> Vec<String> {
        let mut names: Vec<String> =
            self.content.tensor_infos.keys().filter(|n| n.starts_with(prefix)).cloned().collect();
        names.sort();
        names
    }

    pub fn tensor_count(&self) -> usize {
        self.content.tensor_infos.len()
    }

    /// 파일 안 모든 텐서의 바이트 수 합
    pub fn total_bytes(&self) -> u64 {
        self.content.tensor_infos.keys().filter_map(|n| self.info(n).ok()).map(|m| m.bytes).sum()
    }

    // ---- 읽기 ----

    /// 텐서의 목차 항목과 저장된 바이트. 매핑을 그대로 빌려 줍니다.
    fn stored(&self, name: &str) -> Result<(&TensorInfo, &[u8])> {
        let info = self
            .content
            .tensor_infos
            .get(name)
            .ok_or_else(|| anyhow!("GGUF 에 {name} 텐서가 없습니다 ({})", self.path.display()))?;
        let bytes = info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        let start = (self.content.tensor_data_offset + info.offset) as usize;
        let raw = self.map.get(start..start + bytes).ok_or_else(|| anyhow!("{name}: 파일이 잘렸습니다 ({})", self.path.display()))?;
        Ok((info, raw))
    }

    /// 이 텐서들을 배경에서 미리 읽습니다. 장치에는 아무것도 올리지 않습니다.
    ///
    /// OS 에 미리 읽기를 요청한 뒤 페이지를 하나씩 건드려 이 프로세스에 매핑합니다. Windows 의 미리 읽기는 페이지를 메모리로 가져올 뿐이라,
    /// 처음 읽을 때의 소프트 폴트가 남아 있기 때문입니다 (zoro 전체에 약 2 초). 같은 범위는 한 번만 합니다.
    pub fn warm(self: &Arc<Self>, names: Vec<String>) {
        if std::env::var(WARM_ENV).is_ok_and(|v| v == "0") {
            return;
        }
        let file = Arc::clone(self);
        let started = std::thread::Builder::new().name("zorogravity-warm".into()).spawn(move || file.warm_now(&names));
        if let Err(e) = started {
            eprintln!("[ZOROGRAVITY-MEM] 미리 읽기를 시작하지 못했습니다: {e}");
        }
    }

    /// [`Self::warm`] 의 본체. 호출한 스레드에서 끝까지 읽습니다.
    pub fn warm_now(&self, names: &[String]) {
        let base = self.map.as_ptr() as usize;
        let mut spans: Vec<(usize, usize)> = names
            .iter()
            .filter_map(|n| self.stored(n).ok())
            .map(|(_, raw)| (raw.as_ptr() as usize - base, raw.as_ptr() as usize - base + raw.len()))
            .collect();
        spans.sort_unstable();
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (start, end) in spans {
            match merged.last_mut() {
                Some(last) if start <= last.1 + PREFETCH_GAP => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        let fresh: Vec<(usize, usize)> = {
            let mut done = self.warmed.lock().unwrap_or_else(|p| p.into_inner());
            merged.into_iter().filter(|span| done.insert(*span)).collect()
        };
        for (start, end) in fresh {
            prefetch_range(&self.map, start, end - start);
            let mut sum = 0u8;
            for at in (start..end).step_by(PAGE) {
                // 읽기가 사라지지 않게 합니다. 범위는 위에서 매핑 안임을 확인했습니다.
                sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(self.map.as_ptr().add(at)) });
            }
            std::hint::black_box(sum);
        }
    }

    /// 텐서 하나를 `device` 로 읽습니다 (양자화 형식 그대로). 장치로 가는 복사 한 번뿐입니다.
    pub fn read_qtensor(&self, name: &str, device: &Device) -> Result<QTensor> {
        self.content
            .tensor_from_slice(&self.map, name, device)
            .with_context(|| format!("{name} 텐서를 읽을 수 없습니다"))
    }

    /// 텐서 하나를 풀어서 `dtype` 의 일반 텐서로 `device` 에 둡니다. 정규화·컨볼루션·비전 가중치 같은 일반 텐서용입니다.
    /// 저장 형식 그대로 쓰면 변환 없이 복사만 하고, BF16 을 F32 로 풀 때는 병렬로 합니다.
    pub fn read_dense(&self, name: &str, device: &Device, dtype: DType) -> Result<Tensor> {
        let (info, raw) = self.stored(name)?;
        let dims = info.shape.dims().to_vec();
        match (info.ggml_dtype, dtype) {
            (GgmlDType::F32, DType::F32) => Ok(Tensor::from_slice(as_elements::<f32>(name, raw)?, dims, device)?),
            (GgmlDType::BF16, DType::BF16) => Ok(Tensor::from_slice(as_elements::<half::bf16>(name, raw)?, dims, device)?),
            (GgmlDType::BF16, DType::F32) => Ok(Tensor::from_vec(bf16_to_f32(as_elements(name, raw)?), dims, device)?),
            _ => {
                let t = self.read_qtensor(name, &Device::Cpu)?.dequantize(&Device::Cpu)?;
                let t = if t.dtype() == dtype { t } else { t.to_dtype(dtype)? };
                Ok(t.to_device(device)?)
            }
        }
    }

    /// 2차원 텐서 `name` 에서 `rows` 번 행만 읽어 F32 [행 수, 열 수] 로 `device` 에 둡니다.
    /// 표 전체를 읽지 않고 그 행의 바이트만 파일에서 읽습니다.
    pub fn gather_rows(&self, name: &str, rows: &[u32], device: &Device) -> Result<Tensor> {
        if rows.is_empty() {
            bail!("{name}: 읽을 행이 없습니다");
        }
        let (info, table) = self.stored(name)?;
        let dims = info.shape.dims();
        if dims.len() != 2 {
            bail!("{name}: 행을 고르려면 2차원이어야 합니다 (모양 {dims:?})");
        }
        let (n_rows, cols) = (dims[0], dims[1]);
        let dtype = info.ggml_dtype;
        let (block, size) = (dtype.block_size(), dtype.type_size());
        if cols % block != 0 {
            bail!("{name}: 열 수 {cols} 가 블록 크기 {block} 로 나누어지지 않습니다");
        }
        let row_bytes = cols / block * size;
        // `qtensor_from_ggml` 이 블록으로 다시 보므로 u32 로 정렬된 버퍼에 모읍니다.
        let mut words = vec![0u32; (rows.len() * row_bytes).div_ceil(4)];
        let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
        let raw = &mut raw[..rows.len() * row_bytes];
        for (i, &r) in rows.iter().enumerate() {
            let r = r as usize;
            if r >= n_rows {
                bail!("{name}: 행 {r} 이 범위 밖입니다 (행 수 {n_rows})");
            }
            raw[i * row_bytes..(i + 1) * row_bytes].copy_from_slice(&table[r * row_bytes..(r + 1) * row_bytes]);
        }
        let q = qtensor_from_ggml(dtype, raw, vec![rows.len(), cols], &Device::Cpu)?;
        Ok(q.dequantize(&Device::Cpu)?.to_device(device)?)
    }
}

/// 시험과 예제에서 쓰는 GGUF 를 만드는 도우미 (실제 변환은 `tools/gguf-convert` 가 합니다).
#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use candle_core::quantized::gguf_file;

    pub struct TempDir(pub PathBuf);

    impl TempDir {
        pub fn new(tag: &str) -> Self {
            static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let n = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let dir = std::env::temp_dir().join(format!("zorogravity-{tag}-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("임시 폴더");
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// (이름, 텐서, 저장 형식) 목록으로 GGUF 를 씁니다.
    pub fn write_gguf(
        path: &Path,
        meta: &[(&str, Value)],
        tensors: &[(String, Tensor, GgmlDType)],
    ) -> Result<()> {
        let q: Vec<(String, QTensor)> = tensors
            .iter()
            .map(|(n, t, d)| Ok((n.clone(), QTensor::quantize(t, *d)?)))
            .collect::<Result<_>>()?;
        let refs: Vec<(&str, &QTensor)> = q.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let meta_refs: Vec<(&str, &Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
        let mut f = File::create(path)?;
        gguf_file::write(&mut f, &meta_refs, &refs)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use crate::ai::verify::det_normal;

    fn rand(rows: usize, cols: usize, seed: u64) -> Tensor {
        Tensor::from_vec(det_normal(rows * cols, seed), (rows, cols), &Device::Cpu).unwrap()
    }

    fn sample() -> Result<(TempDir, WeightFile)> {
        let dir = TempDir::new("weights");
        let path = dir.0.join("m.gguf");
        write_gguf(
            &path,
            &[("general.architecture", Value::String("zoro".into())), ("zoro.block_count", Value::U32(2))],
            &[
                ("token_embd.weight".into(), rand(40, 64, 1), GgmlDType::Q8_0),
                ("blk.0.attn_norm.weight".into(), rand(1, 64, 2).reshape(64)?, GgmlDType::F32),
                ("blk.1.ffn_up.weight".into(), rand(32, 64, 3), GgmlDType::Q8_0),
                ("blk.10.ffn_up.weight".into(), rand(32, 64, 4), GgmlDType::Q8_0),
                ("output_norm.weight".into(), rand(1, 64, 5).reshape(64)?, GgmlDType::F32),
            ],
        )?;
        let file = WeightFile::open(&path)?;
        Ok((dir, file))
    }

    #[test]
    fn the_index_is_readable_without_reading_any_tensor_data() -> Result<()> {
        let (_dir, f) = sample()?;
        assert_eq!(f.tensor_count(), 5);
        assert_eq!(f.meta_str("general.architecture")?, "zoro");
        assert_eq!(f.meta_usize("zoro.block_count")?, 2);
        assert!(f.meta_usize("general.architecture").is_err());
        assert!(f.meta_str("zoro.block_count").is_err());
        assert!(f.meta_str("missing").is_err());

        let m = f.info("token_embd.weight")?;
        assert_eq!((m.dims.as_slice(), m.dtype), (&[40usize, 64][..], GgmlDType::Q8_0));
        // Q8_0: 32 개마다 34 바이트
        assert_eq!(m.bytes, 40 * 64 / 32 * 34);
        assert!(!m.is_float());
        assert!(f.info("blk.0.attn_norm.weight")?.is_float());
        assert_eq!(f.info("blk.0.attn_norm.weight")?.bytes, 64 * 4);
        assert!(f.info("nope").is_err());
        assert_eq!(f.total_bytes(), (40 * 64 / 32 * 34 + 64 * 4 + 2 * (32 * 64 / 32 * 34) + 64 * 4) as u64);
        Ok(())
    }

    #[test]
    fn prefix_listing_does_not_confuse_block_1_with_block_10() -> Result<()> {
        let (_dir, f) = sample()?;
        assert_eq!(f.names_with_prefix("blk.1."), vec!["blk.1.ffn_up.weight"]);
        assert_eq!(f.names_with_prefix("blk.10."), vec!["blk.10.ffn_up.weight"]);
        assert_eq!(f.names_with_prefix("blk.").len(), 3);
        Ok(())
    }

    #[test]
    fn gathered_rows_equal_the_same_rows_of_the_whole_table() -> Result<()> {
        let (_dir, f) = sample()?;
        let whole = f.read_qtensor("token_embd.weight", &Device::Cpu)?.dequantize(&Device::Cpu)?;
        let ids = [7u32, 0, 39, 7, 21];
        let got = f.gather_rows("token_embd.weight", &ids, &Device::Cpu)?;
        assert_eq!(got.dims(), &[5, 64]);
        let want = whole.index_select(&Tensor::new(&ids[..], &Device::Cpu)?, 0)?;
        let diff = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert_eq!(diff, 0.0, "같은 바이트를 같은 방식으로 풀어야 합니다");
        Ok(())
    }

    #[test]
    fn gather_rejects_bad_requests() -> Result<()> {
        let (_dir, f) = sample()?;
        assert!(f.gather_rows("token_embd.weight", &[], &Device::Cpu).is_err());
        assert!(f.gather_rows("token_embd.weight", &[40], &Device::Cpu).is_err(), "행 수와 같은 번호는 범위 밖");
        assert!(f.gather_rows("blk.0.attn_norm.weight", &[0], &Device::Cpu).is_err(), "1차원은 행을 고를 수 없음");
        assert!(f.gather_rows("nope", &[0], &Device::Cpu).is_err());
        Ok(())
    }

    #[test]
    fn warming_changes_nothing_that_is_read() -> Result<()> {
        let (_dir, f) = sample()?;
        let names: Vec<String> = f.names_with_prefix("blk.");
        let before = f.read_dense("blk.1.ffn_up.weight", &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        f.warm_now(&names);
        f.warm_now(&names);
        f.warm_now(&["nope".to_string()]);
        assert_eq!(f.read_dense("blk.1.ffn_up.weight", &Device::Cpu, DType::F32)?.flatten_all()?.to_vec1::<f32>()?, before);
        assert_eq!(f.warmed.lock().unwrap().len(), 1, "붙어 있는 텐서는 한 범위로 합치고 한 번만 요청합니다");
        Ok(())
    }

    /// 빠른 경로의 결과는 양자화 텐서를 풀어 바꾸던 예전 경로와 비트까지 같아야 합니다.
    #[test]
    fn stored_floats_read_the_same_without_the_round_trip() -> Result<()> {
        let dir = TempDir::new("weights-dense");
        let path = dir.0.join("d.gguf");
        let v = Tensor::from_vec(det_normal(70_000 * 3, 8), (70_000, 3), &Device::Cpu)?;
        write_gguf(&path, &[("general.architecture", Value::String("zoro".into()))], &[("v".into(), v.clone(), GgmlDType::BF16), ("f".into(), v, GgmlDType::F32)])?;
        let f = WeightFile::open(&path)?;
        for (name, dtype) in [("v", DType::BF16), ("v", DType::F32), ("f", DType::F32), ("f", DType::BF16)] {
            let got = f.read_dense(name, &Device::Cpu, dtype)?;
            let old = f.read_qtensor(name, &Device::Cpu)?.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
            assert_eq!((got.dtype(), got.dims()), (old.dtype(), old.dims()), "{name} {dtype:?}");
            let bits = |t: &Tensor| -> Result<Vec<u32>> { Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?.iter().map(|x| x.to_bits()).collect()) };
            assert!(bits(&got)? == bits(&old)?, "{name} {dtype:?}");
        }
        Ok(())
    }

    #[test]
    fn dense_reads_convert_the_dtype() -> Result<()> {
        let (_dir, f) = sample()?;
        let a = f.read_dense("output_norm.weight", &Device::Cpu, DType::F32)?;
        let b = f.read_dense("output_norm.weight", &Device::Cpu, DType::BF16)?;
        assert_eq!((a.dtype(), b.dtype()), (DType::F32, DType::BF16));
        let err = (&a - &b.to_dtype(DType::F32)?)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(err < 0.05, "BF16 반올림 오차 정도여야 합니다: {err}");
        Ok(())
    }
}
