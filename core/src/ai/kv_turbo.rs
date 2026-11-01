//! KV 캐시 FP8 압축. logis-center 의 `models/common/kv_turbo.rs` · `accel_check.rs` 와 같은 정책입니다.
//!
//! 기본 형식은 벡터별 스케일 FP8(e4m3, 회전 없음)입니다. 벡터 하나가 `head_dim` + 8 바이트라서 F32 의 약 1/4 입니다.
//! GPU(CUDA · ROCm · Vulkan)에서 포크의 TurboQuant 커널로 압축 · 복원하고, CPU 는 원래대로 압축하지 않습니다.
//!
//! - `ZOROGRAVITY_KV_PRECISION=auto` (기본): 원본 KV 가 장치 여유 메모리에 들어가면 원본(F32), 아니면 FP8
//! - `exact`: 항상 원본, `compressed`: GPU 에서 항상 FP8
//!
//! 프리필은 캐시가 비어 있으므로 원본으로 계산합니다. 이후 새 토큰 하나를 읽을 때는 압축 행 위에서 바로 계산하고,
//! 여러 토큰을 읽을 때는 이력만 풀어서 새 토큰의 원본과 함께 계산합니다.
//! 압축 · 복원 · 어텐션은 처음 쓰기 전에 CPU 기준과 비교해 맞을 때만 씁니다 (`ai::verify`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::{bail, Result};
use candle_core::accel::{config, QuantFormat, RotationKind, TurboQuantOptions};
use candle_core::{Device, Tensor};
use candle_nn::flash_attn::FlashAttnConfig;
use candle_nn::turbo_quant::{attention_packed_blocks, KvBlockRef, TurboQuantizer};

use crate::ai::attention::{attend, AttnMode, StepMask};
use crate::ai::{accel, verify};

/// KV 정밀도 정책을 정하는 환경 변수: `auto`(기본) · `exact` · `compressed`
pub const PRECISION_ENV: &str = "ZOROGRAVITY_KV_PRECISION";
/// 사용자가 KV 압축 형식 · 비트를 직접 고르는 포크의 환경 변수들. 하나라도 있으면 포크의 설정을 그대로 따릅니다.
const FORMAT_ENV_VARS: [&str; 6] = [
    "CANDLE_TURBO_QUANT_FORMAT",
    "CANDLE_TURBO_QUANT_K_FORMAT",
    "CANDLE_TURBO_QUANT_V_FORMAT",
    "CANDLE_TURBO_QUANT_BITS",
    "CANDLE_TURBO_QUANT_K_BITS",
    "CANDLE_TURBO_QUANT_V_BITS",
];

/// GPU 로 압축한 행을 CPU 코덱으로 풀었을 때 CPU 압축과의 차이 (경계값 반올림 몇 개 허용)
const QUANT_TOL: f32 = 2e-2;
/// GPU 복원과 CPU 복원의 차이
const DEQUANT_TOL: f32 = 1e-3;
/// 압축 행 위의 어텐션과 CPU 기준값의 차이
const ATTN_TOL: f32 = 3e-2;
/// 실측 감시를 하는 최대 횟수 (층 단위)
const PROBE_LIMIT: usize = 64;
/// 실측 오차가 이 값을 넘으면 커널 오동작 수준으로 보고 압축을 끕니다. FP8 은 보통 4–6%, 커널 오류는 50% 이상입니다.
const PROBE_BROKEN_REL: f32 = 0.5;

/// 압축 · 복원의 자가 검증 항목
fn codec_op(head_dim: usize) -> String {
    format!("kv_fp8_pack_d{head_dim}")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvPrecision {
    Auto,
    Exact,
    Compressed,
}

impl KvPrecision {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "exact" | "full" => Ok(Self::Exact),
            "compressed" | "fp8" => Ok(Self::Compressed),
            other => bail!("{PRECISION_ENV}={other:?}: auto, exact, compressed 중 하나여야 합니다"),
        }
    }

    pub fn from_env() -> Result<Self> {
        match std::env::var(PRECISION_ENV) {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(Self::Auto),
        }
    }
}

/// 이번 생성의 KV 크기
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    /// KV 캐시를 쓰는 층 수
    pub layers: usize,
    /// 프롬프트 + 생성 토큰 수
    pub tokens: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
}

impl KvShape {
    /// 원본(F32) K · V 가 차지하는 바이트 수
    pub fn exact_bytes(&self) -> u64 {
        [self.layers, self.tokens, self.kv_heads, self.head_dim].iter().fold(2 * 4u64, |a, &x| a.saturating_mul(x as u64))
    }
}

/// 압축할지 정합니다. CPU 는 압축하지 않고, `auto` 는 원본이 들어가면 원본으로 둡니다.
/// 여유 메모리를 모르는 GPU 는 메모리를 아끼는 쪽을 고릅니다.
fn wants_compression(policy: KvPrecision, gpu: bool, exact_bytes: u64, free: Option<u64>, headroom: u64) -> bool {
    if !gpu {
        return false;
    }
    match policy {
        KvPrecision::Exact => false,
        KvPrecision::Compressed => true,
        KvPrecision::Auto => !free.is_some_and(|f| f >= exact_bytes.saturating_add(headroom)),
    }
}

/// 이 앱의 KV 압축 형식. 아무것도 지정하지 않으면 정확도 우선으로 K · V 모두 벡터별 FP8, 회전 없음입니다.
/// FP8 은 회전을 걸면 오히려 나빠집니다 (logis-center 측정). 포크의 형식 변수를 지정하면 그 설정을 따릅니다.
pub fn kv_quant_options() -> TurboQuantOptions {
    let is_set = |v: &str| std::env::var(v).is_ok_and(|x| !x.trim().is_empty());
    let mut o = config().turbo_quant;
    if !FORMAT_ENV_VARS.iter().any(|v| is_set(v)) {
        o.key_format = Some(QuantFormat::Fp8);
        o.value_format = Some(QuantFormat::Fp8);
        o.key_bits = 8;
        o.value_bits = 8;
        if !is_set("CANDLE_TURBO_QUANT_ROTATION") {
            o.rotation = RotationKind::None;
        }
    }
    o
}

/// 한 생성의 KV 를 압축 · 복원하는 코덱. 층들이 함께 씁니다.
pub struct KvCodec {
    kq: TurboQuantizer,
    vq: TurboQuantizer,
    /// 압축 행 위의 어텐션 커널이 자가 검증을 통과했는가. 아니면 항상 풀어서 계산합니다.
    fused: bool,
    /// 실측 감시에서 커널 오동작 수준의 오차를 보았는가
    broken: AtomicBool,
    probe: Mutex<(usize, f32, f32)>,
}

impl KvCodec {
    pub fn new(head_dim: usize, device: &Device) -> Result<Self> {
        let o = kv_quant_options();
        Ok(Self {
            kq: TurboQuantizer::from_options(&o, head_dim, true, device)?,
            vq: TurboQuantizer::from_options(&o, head_dim, false, device)?,
            fused: o.fused_attention,
            broken: AtomicBool::new(false),
            probe: Mutex::new((0, 0.0, 0.0)),
        })
    }

    fn with_fused(mut self, fused: bool) -> Self {
        self.fused = fused;
        self
    }

    /// 새 토큰 하나를 압축 행 위에서 바로 계산해도 되는가
    pub fn fused(&self) -> bool {
        self.fused
    }

    /// 실측 감시가 이 코덱을 쓰지 말라고 했는가
    pub fn broken(&self) -> bool {
        self.broken.load(Ordering::Relaxed)
    }

    /// 키 `[Hkv, T, D]` → 압축 행 `[Hkv, T, 워드]` (u32)
    pub fn pack_k(&self, k: &Tensor) -> Result<Tensor> {
        pack(&self.kq, k)
    }

    /// 값 `[Hkv, T, D]` → 압축 행 `[Hkv, T, 워드]` (u32)
    pub fn pack_v(&self, v: &Tensor) -> Result<Tensor> {
        pack(&self.vq, v)
    }

    pub fn unpack_k(&self, rows: &Tensor) -> Result<Tensor> {
        unpack(&self.kq, rows)
    }

    pub fn unpack_v(&self, rows: &Tensor) -> Result<Tensor> {
        unpack(&self.vq, rows)
    }

    /// 압축 행 구간들 `(k, v)` 위의 인과 주의. `q` `[H, n, D]` 는 구간을 이은 끝의 `n` 토큰입니다. 결과는 `[H, n, D]` (F32).
    pub fn attend_rows(&self, q: &Tensor, segments: &[(&Tensor, &Tensor)], scale: f64) -> Result<Tensor> {
        let held = segments
            .iter()
            .map(|(k, v)| Ok((k.unsqueeze(0)?, v.unsqueeze(0)?, k.dim(1)?)))
            .collect::<Result<Vec<_>>>()?;
        let blocks: Vec<KvBlockRef<'_>> = held.iter().map(|(k, v, rows)| KvBlockRef { k, v, rows: *rows }).collect();
        let cfg = FlashAttnConfig::new(scale as f32).causal();
        Ok(attention_packed_blocks(&q.unsqueeze(0)?, &blocks, &self.kq, &self.vq, &cfg)?.squeeze(0)?)
    }

    /// 실측 감시: 원본으로 계산한 프리필의 마지막 토큰을 압축 행으로도 계산해 실제 모델의 오차를 기록합니다.
    /// 오차가 커널 오동작 수준이면 이 장치의 압축을 끄고, 아직 비어 있는 층들은 원본으로 이어 갑니다.
    pub fn watch(&self, q: &Tensor, kt: &Tensor, v: &Tensor, exact: &Tensor, scale: f64) -> Result<()> {
        static WATCHED: AtomicUsize = AtomicUsize::new(0);
        let device = q.device();
        if !self.fused || !accel::is_gpu(device) || !verify::checks_enabled() || WATCHED.fetch_add(1, Ordering::Relaxed) >= PROBE_LIMIT {
            return Ok(());
        }
        let n = q.dim(1)?;
        let k_rows = self.pack_k(&kt.transpose(1, 2)?.contiguous()?)?;
        let v_rows = self.pack_v(v)?;
        let got = self.attend_rows(&q.narrow(1, n - 1, 1)?.contiguous()?, &[(&k_rows, &v_rows)], scale)?;
        let rel = verify::rel_err(&got, &exact.narrow(1, n - 1, 1)?.contiguous()?)?;
        let mut p = self.probe.lock().unwrap_or_else(|e| e.into_inner());
        p.0 += 1;
        p.1 += rel;
        p.2 = p.2.max(rel);
        if rel > PROBE_BROKEN_REL {
            eprintln!("[ZOROGRAVITY-KV] ❌ 실측 오차 {:.1}% > {:.0}% → 이 장치에서 KV 압축을 끕니다", rel * 100.0, PROBE_BROKEN_REL * 100.0);
            self.broken.store(true, Ordering::Relaxed);
            verify::revoke(device, &codec_op(self.kq.dim()));
        }
        Ok(())
    }
}

impl Drop for KvCodec {
    fn drop(&mut self) {
        let p = self.probe.lock().unwrap_or_else(|e| e.into_inner());
        if p.0 > 0 {
            eprintln!("[ZOROGRAVITY-KV] FP8 실측 오차: 층 {}개 평균 {:.2}% · 최대 {:.2}% (원본 KV 어텐션 대비)", p.0, p.1 / p.0 as f32 * 100.0, p.2 * 100.0);
        }
    }
}

fn pack(q: &TurboQuantizer, x: &Tensor) -> Result<Tensor> {
    let (h, t, d) = x.dims3()?;
    Ok(q.quantize_rows(&x.reshape((h * t, d))?)?.reshape((h, t, q.row_words()))?)
}

fn unpack(q: &TurboQuantizer, rows: &Tensor) -> Result<Tensor> {
    let (h, t, w) = rows.dims3()?;
    Ok(q.dequantize_rows(&rows.contiguous()?.reshape((h * t, w))?)?.reshape((h, t, q.dim()))?)
}

/// 행마다 크기를 0.05 ~ 6.4 배로 바꾼 입력으로 GPU 압축 · 복원을 CPU 코덱과 비교합니다. `(압축 오차, 복원 오차)`
fn codec_probe(device: &Device, d: usize) -> Result<(f32, f32)> {
    const ROWS: usize = 48;
    let cpu = Device::Cpu;
    let scale: Vec<f32> = (0..ROWS).map(|r| 0.05 * (1u32 << (r % 8)) as f32).collect();
    let x = Tensor::from_vec(verify::det_normal(ROWS * d, 0x7100 + d as u64), (1, ROWS, d), &cpu)?
        .broadcast_mul(&Tensor::from_vec(scale, (1, ROWS, 1), &cpu)?)?;
    let (on_gpu, on_cpu) = (KvCodec::new(d, device)?, KvCodec::new(d, &cpu)?);
    let (mut quant, mut dequant) = (0f32, 0f32);
    for keys in [true, false] {
        let packs = |c: &KvCodec, x: &Tensor| if keys { c.pack_k(x) } else { c.pack_v(x) };
        let unpacks = |c: &KvCodec, rows: &Tensor| if keys { c.unpack_k(rows) } else { c.unpack_v(rows) };
        let packed_gpu = packs(&on_gpu, &x.to_device(device)?)?;
        let want = unpacks(&on_cpu, &packs(&on_cpu, &x)?)?;
        let got = unpacks(&on_cpu, &packed_gpu.to_device(&cpu)?)?;
        quant = quant.max(verify::rel_err(&got, &want)?);
        dequant = dequant.max(verify::rel_err(&unpacks(&on_gpu, &packed_gpu)?.to_device(&cpu)?, &got)?);
    }
    Ok((quant, dequant))
}

/// 압축 행 위의 어텐션 `[실제 헤드 구성, 토큰 300개의 이력 + 새 토큰 1개]` 를 CPU 에서 복원한 K · V 의 어텐션과 비교합니다.
fn fused_probe(device: &Device, q_heads: usize, kv_heads: usize, d: usize) -> Result<f32> {
    const KEYS: usize = 300;
    let cpu = Device::Cpu;
    let scale = 1.0 / (d as f64).sqrt();
    let seed = 0x7A00 + d as u64;
    let k = Tensor::from_vec(verify::det_normal(kv_heads * KEYS * d, seed), (kv_heads, KEYS, d), &cpu)?;
    let v = Tensor::from_vec(verify::det_normal(kv_heads * KEYS * d, seed + 1), (kv_heads, KEYS, d), &cpu)?;
    let q = (Tensor::from_vec(verify::det_normal(q_heads * d, seed + 2), (q_heads, 1, d), &cpu)? * 2.0)?;
    let on_gpu = KvCodec::new(d, device)?;
    let on_cpu = KvCodec::new(d, &cpu)?;
    let (k_rows, v_rows) = (on_gpu.pack_k(&k.to_device(device)?)?, on_gpu.pack_v(&v.to_device(device)?)?);
    let got = on_gpu.attend_rows(&q.to_device(device)?, &[(&k_rows, &v_rows)], scale)?.to_device(&cpu)?;
    let (kd, vd) = (on_cpu.unpack_k(&on_cpu.pack_k(&k)?)?, on_cpu.unpack_v(&on_cpu.pack_v(&v)?)?);
    let want = attend_all(&q, &kd, &vd, scale)?;
    verify::rel_err(&got, &want)
}

/// 이력 전체를 보는 새 토큰 하나의 어텐션 (CPU 기준값)
fn attend_all(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor> {
    let n = k.dim(1)?;
    let last = |x: &Tensor| -> Result<Tensor> { Ok(x.narrow(1, n - 1, 1)?.contiguous()?) };
    let past = |x: &Tensor| -> Result<Tensor> { Ok(x.narrow(1, 0, n - 1)?.contiguous()?) };
    let (h, hkv) = (q.dim(0)?, k.dim(0)?);
    let kt = |x: &Tensor| -> Result<Tensor> { Ok(x.transpose(1, 2)?.contiguous()?) };
    let mask = StepMask::new(AttnMode::Causal, 1, h / hkv, q.device())?;
    attend(q, &kt(&last(k)?)?, &last(v)?, &[(&kt(&past(k)?)?, &past(v)?)], &mask, scale)
}

/// 이 장치에서 쓸 코덱. 자가 검증을 통과하지 못하면 None 이고, 호출자는 원본 KV 를 씁니다.
fn verified_codec(shape: &KvShape, device: &Device) -> Option<KvCodec> {
    let d = shape.head_dim;
    if !config().turbo_quant.enabled {
        return None;
    }
    // 허용 오차가 둘이라서 각각을 허용 오차로 나눈 값이 1 이하인지 봅니다.
    let packs = verify::trusted(device, &codec_op(d), 1.0, || {
        let (quant, dequant) = codec_probe(device, d)?;
        Ok((quant / QUANT_TOL).max(dequant / DEQUANT_TOL))
    });
    if !packs {
        return None;
    }
    let op = format!("kv_fp8_attn_{}_{}_d{d}", shape.q_heads, shape.kv_heads);
    let fused = verify::trusted(device, &op, ATTN_TOL, || fused_probe(device, shape.q_heads, shape.kv_heads, d));
    match KvCodec::new(d, device) {
        Ok(c) => {
            let fused = c.fused && fused;
            Some(c.with_fused(fused))
        }
        Err(e) => {
            eprintln!("[ZOROGRAVITY-KV] 코덱을 만들 수 없어 원본 KV 를 씁니다: {e}");
            None
        }
    }
}

/// 이번 생성의 KV 를 압축할 코덱을 정합니다. None 이면 원본(F32)으로 둡니다.
/// `free` 는 가중치를 올린 뒤의 장치 여유 메모리, `headroom` 은 남겨 둘 여유입니다.
pub fn plan(shape: &KvShape, device: &Device, free: Option<u64>, headroom: u64) -> Result<Option<Arc<KvCodec>>> {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    let policy = KvPrecision::from_env()?;
    let exact = shape.exact_bytes();
    if !wants_compression(policy, accel::is_gpu(device), exact, free, headroom) {
        return Ok(None);
    }
    let codec = verified_codec(shape, device).map(Arc::new);
    if codec.is_some() && !LOGGED.swap(true, Ordering::Relaxed) {
        let (mb, free_mb) = (exact >> 20, free.map_or("모름".to_string(), |f| format!("{} MB", f >> 20)));
        eprintln!("[ZOROGRAVITY-KV] 원본 KV {mb} MB 를 벡터별 FP8 로 압축합니다 (정책 {policy:?}, 장치 여유 {free_mb})");
    }
    Ok(codec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::verify::{det_normal, rel_err};

    fn t(data: Vec<f32>, shape: (usize, usize, usize)) -> Tensor {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    const GB: u64 = 1 << 30;

    #[test]
    fn parses_the_precision_policy() {
        assert_eq!(KvPrecision::parse("").unwrap(), KvPrecision::Auto);
        assert_eq!(KvPrecision::parse(" AUTO ").unwrap(), KvPrecision::Auto);
        assert_eq!(KvPrecision::parse("exact").unwrap(), KvPrecision::Exact);
        assert_eq!(KvPrecision::parse("FP8").unwrap(), KvPrecision::Compressed);
        assert!(KvPrecision::parse("half").is_err());
    }

    #[test]
    fn exact_bytes_count_keys_and_values_in_f32() {
        let s = KvShape { layers: 16, tokens: 1000, q_heads: 16, kv_heads: 8, head_dim: 128 };
        assert_eq!(s.exact_bytes(), 16 * 1000 * 8 * 128 * 2 * 4);
    }

    #[test]
    fn compression_follows_the_policy_and_the_free_memory() {
        let want = |p, gpu, exact, free| wants_compression(p, gpu, exact, free, GB);
        assert!(!want(KvPrecision::Compressed, false, GB, None), "CPU 는 압축하지 않습니다");
        assert!(!want(KvPrecision::Exact, true, 100 * GB, Some(0)));
        assert!(want(KvPrecision::Compressed, true, 0, Some(100 * GB)));
        assert!(!want(KvPrecision::Auto, true, GB, Some(2 * GB)), "여유를 남기고 들어가면 원본입니다");
        assert!(want(KvPrecision::Auto, true, GB, Some(2 * GB - 1)));
        assert!(want(KvPrecision::Auto, true, GB, None), "여유를 모르면 아끼는 쪽입니다");
    }

    #[test]
    fn the_default_format_is_per_vector_fp8_at_a_quarter_of_the_size() -> Result<()> {
        let c = KvCodec::new(128, &Device::Cpu)?;
        assert_eq!(c.kq.format(), QuantFormat::Fp8);
        assert_eq!(c.vq.format(), QuantFormat::Fp8);
        assert_eq!(c.kq.rotation(), RotationKind::None);
        let x = t(det_normal(3 * 5 * 128, 1), (3, 5, 128));
        let rows = c.pack_k(&x)?;
        assert_eq!(rows.dtype(), candle_core::DType::U32);
        assert_eq!(rows.dims()[..2], [3, 5]);
        assert!(rows.elem_count() * 4 * 3 < x.elem_count() * 4, "F32 의 1/3 보다 작아야 합니다");
        Ok(())
    }

    #[test]
    fn a_round_trip_keeps_vectors_of_any_size() -> Result<()> {
        let c = KvCodec::new(64, &Device::Cpu)?;
        for gain in [1e-3f32, 1.0, 1e3] {
            let x = t(det_normal(2 * 7 * 64, 2).into_iter().map(|v| v * gain).collect(), (2, 7, 64));
            let err = rel_err(&c.unpack_v(&c.pack_v(&x)?)?, &x)?;
            assert!(err < 0.05, "배율 {gain}: {err}");
        }
        Ok(())
    }

    #[test]
    fn attention_on_the_rows_matches_the_decoded_keys_and_values() -> Result<()> {
        // 헤드 묶음이 어텐션 쪽과 같은지 봅니다. GPU 커널을 검증할 때도 이 기준값과 비교합니다.
        let (h, hkv, n, d) = (8usize, 4usize, 40usize, 32usize);
        let c = KvCodec::new(d, &Device::Cpu)?;
        let k = t(det_normal(hkv * n * d, 3), (hkv, n, d));
        let v = t(det_normal(hkv * n * d, 4), (hkv, n, d));
        let q = t(det_normal(h * d, 5), (h, 1, d));
        let (kr, vr) = (c.pack_k(&k)?, c.pack_v(&v)?);
        let scale = 1.0 / (d as f64).sqrt();
        let got = c.attend_rows(&q, &[(&kr, &vr)], scale)?;
        let want = attend_all(&q, &c.unpack_k(&kr)?, &c.unpack_v(&vr)?, scale)?;
        assert_eq!(got.dims(), &[h, 1, d]);
        let err = rel_err(&got, &want)?;
        assert!(err < 1e-4, "{err}");
        // 행을 나눠 넣어도 같습니다.
        let piece = |x: &Tensor, s, l| x.narrow(1, s, l).unwrap().contiguous().unwrap();
        let (k1, v1, k2, v2) = (piece(&kr, 0, 25), piece(&vr, 0, 25), piece(&kr, 25, 15), piece(&vr, 25, 15));
        let parted = c.attend_rows(&q, &[(&k1, &v1), (&k2, &v2)], scale)?;
        assert!(rel_err(&parted, &got)? < 1e-5);
        Ok(())
    }

    #[test]
    fn a_cpu_plan_keeps_the_exact_cache() -> Result<()> {
        let s = KvShape { layers: 2, tokens: 10, q_heads: 4, kv_heads: 2, head_dim: 8 };
        assert!(plan(&s, &Device::Cpu, None, 0)?.is_none());
        Ok(())
    }
}
