//! 장치 선택과 dtype 정책.
//! 순서는 logis-center `utils/device_utils.rs` 와 같습니다: CUDA → ROCm → Metal → Vulkan → CPU.
//! Metal 을 Vulkan 앞에 둡니다. 맥의 Vulkan(MoltenVK)은 Metal 위에서 도는 변환 층이기 때문입니다.
//! 장치가 없거나 기능이 빠졌으면 다음 후보로 넘어가고, 열기에 실패한 이유는 한 줄로 남깁니다.

use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device};

/// 장치 선택 환경 변수: `auto`(기본) · `cuda` · `rocm` · `metal` · `vulkan` · `cpu`
pub const DEVICE_ENV: &str = "ZOROGRAVITY_DEVICE";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DevicePref {
    Auto,
    Cuda,
    Rocm,
    Metal,
    Vulkan,
    Cpu,
}

impl DevicePref {
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cuda" | "nvidia" => Ok(Self::Cuda),
            "rocm" | "hip" => Ok(Self::Rocm),
            "metal" => Ok(Self::Metal),
            "vulkan" => Ok(Self::Vulkan),
            "cpu" => Ok(Self::Cpu),
            other => bail!("{DEVICE_ENV}={other:?}: auto, cuda, rocm, metal, vulkan, cpu 중 하나여야 합니다"),
        }
    }

    pub fn from_env() -> Result<Self> {
        match std::env::var(DEVICE_ENV) {
            Ok(v) => Self::parse(&v),
            Err(_) => Ok(Self::Auto),
        }
    }
}

/// 환경 변수 `ZOROGRAVITY_DEVICE` 에 따라 장치를 엽니다.
pub fn select_device() -> Result<Device> {
    select_device_with(DevicePref::from_env()?)
}

pub fn select_device_with(pref: DevicePref) -> Result<Device> {
    match pref {
        DevicePref::Cpu => Ok(Device::Cpu),
        DevicePref::Cuda => open_cuda()?.ok_or_else(|| {
            anyhow!("CUDA 장치를 열 수 없습니다. cuda 기능으로 빌드했는지, NVIDIA 드라이버가 있는지 확인하세요 (docs/gpu-backends.md)")
        }),
        DevicePref::Rocm => open_rocm()?.ok_or_else(|| {
            anyhow!("ROCm 장치를 열 수 없습니다. rocm 기능으로 빌드했는지, HIP SDK 와 드라이버가 있는지 확인하세요 (docs/gpu-backends.md)")
        }),
        DevicePref::Metal => open_metal()?.ok_or_else(|| {
            anyhow!("Metal 장치를 열 수 없습니다. metal 기능으로 빌드했는지, Metal 을 지원하는 맥인지 확인하세요 (docs/gpu-backends.md)")
        }),
        DevicePref::Vulkan => open_vulkan()?
            .ok_or_else(|| anyhow!("Vulkan 장치가 없습니다. Vulkan 로더(vulkan-1.dll, libvulkan)와 GPU 드라이버를 확인하세요")),
        DevicePref::Auto => {
            match open_cuda() {
                Ok(Some(d)) => return Ok(d),
                Ok(None) => {}
                Err(e) => eprintln!("[ZOROGRAVITY-ACCEL] CUDA 장치 열기 실패 → ROCm 으로 넘어갑니다: {e}"),
            }
            match open_rocm() {
                Ok(Some(d)) => return Ok(d),
                Ok(None) => {}
                Err(e) => eprintln!("[ZOROGRAVITY-ACCEL] ROCm 장치 열기 실패 → Metal 로 넘어갑니다: {e}"),
            }
            match open_metal() {
                Ok(Some(d)) => return Ok(d),
                Ok(None) => {}
                Err(e) => eprintln!("[ZOROGRAVITY-ACCEL] Metal 장치 열기 실패 → Vulkan 으로 넘어갑니다: {e}"),
            }
            match open_vulkan() {
                Ok(Some(d)) => return Ok(d),
                Ok(None) => {}
                Err(e) => eprintln!("[ZOROGRAVITY-ACCEL] Vulkan 장치 열기 실패 → CPU 로 실행합니다: {e}"),
            }
            Ok(Device::Cpu)
        }
    }
}

/// 쓸 GPU 번호를 직접 정하는 환경 변수. 없으면 여유 VRAM 이 가장 많은 GPU 를 고릅니다 (Metal · Vulkan 은 0번).
pub const GPU_ENV: &str = "ZOROGRAVITY_GPU";

#[cfg_attr(not(any(test, feature = "cuda", feature = "rocm")), allow(dead_code))]
fn parse_gpu(s: &str) -> Result<usize> {
    s.trim().parse().map_err(|_| anyhow!("{GPU_ENV}={s:?}: 0 이상의 GPU 번호여야 합니다"))
}

#[cfg_attr(not(any(test, feature = "cuda", feature = "rocm")), allow(dead_code))]
fn gpu_override() -> Result<Option<usize>> {
    std::env::var(GPU_ENV).ok().filter(|v| !v.trim().is_empty()).map(|v| parse_gpu(&v)).transpose()
}

/// 여유 VRAM 이 가장 많은 GPU (logis-center `get_best_device_info` 와 같음). 같으면 앞 번호이고, 모두 재지 못했으면 0번입니다.
#[cfg_attr(not(any(test, feature = "cuda", feature = "rocm")), allow(dead_code))]
fn best_gpu(free: &[(usize, u64)]) -> usize {
    for (id, bytes) in free {
        eprintln!("[ZOROGRAVITY-ACCEL] GPU {id}: 여유 VRAM {:.2} GB", *bytes as f64 / 1e9);
    }
    let mut best = (0, 0);
    for &(id, bytes) in free {
        if bytes > best.1 {
            best = (id, bytes);
        }
    }
    if free.len() > 1 {
        eprintln!("[ZOROGRAVITY-ACCEL] GPU {} 을 씁니다", best.0);
    }
    best.0
}

#[cfg(feature = "cuda")]
fn open_cuda() -> Result<Option<Device>> {
    /// 살펴볼 GPU 수의 상한
    const MAX_GPUS: usize = 8;
    if let Some(id) = gpu_override()? {
        return Ok(Some(Device::new_cuda(id)?));
    }
    // 포크의 CUDA 는 장치 수를 알려 주지 않아서 열리는 데까지 엽니다. 고르지 않은 장치는 여기서 닫힙니다.
    let mut devices: Vec<Device> = (0..MAX_GPUS).map_while(|id| Device::new_cuda(id).ok()).collect();
    if devices.is_empty() {
        return Ok(Some(Device::new_cuda(0)?));
    }
    let free: Vec<(usize, u64)> = devices
        .iter()
        .enumerate()
        .map(|(id, d)| (id, d.mem_info().ok().flatten().map_or(0, |(free, _)| free as u64)))
        .collect();
    Ok(Some(devices.swap_remove(best_gpu(&free))))
}

#[cfg(not(feature = "cuda"))]
fn open_cuda() -> Result<Option<Device>> {
    Ok(None)
}

/// ROCm 버퍼 캐시 한도를 정하는 포크의 환경 변수
#[cfg(feature = "rocm")]
const ROCM_BUFFER_CACHE_ENV: &str = "CANDLE_ROCM_ALLOC_CACHE_MB";
/// 프리필의 임시 버퍼(한 장에 90 MB)만으로 포크의 기본 한도 256 MiB 가 차서, 이어지는 작은 버퍼가 캐시에 들어가지 못합니다.
/// 그러면 할당마다 앞선 GPU 작업을 기다려서 글쓰기가 토큰당 42 ms 까지 느려집니다 (1 GiB 에서 36 ms, 그 이상은 그대로).
/// 올라온 가중치를 내릴 때 함께 비워지고, 메모리가 모자라면 할당이 캐시를 먼저 돌려줍니다.
#[cfg(feature = "rocm")]
const ROCM_BUFFER_CACHE_MB: &str = "1024";

#[cfg(feature = "rocm")]
fn open_rocm() -> Result<Option<Device>> {
    let count = candle_rocm::device_count().unwrap_or(0);
    if count == 0 {
        return Ok(None);
    }
    if std::env::var_os(ROCM_BUFFER_CACHE_ENV).is_none() {
        std::env::set_var(ROCM_BUFFER_CACHE_ENV, ROCM_BUFFER_CACHE_MB);
    }
    let id = match gpu_override()? {
        Some(id) => id,
        None => {
            let free: Vec<(usize, u64)> = (0..count).map(|id| (id, candle_rocm::mem_info(id).map_or(0, |(free, _)| free as u64))).collect();
            best_gpu(&free)
        }
    };
    Ok(Some(Device::new_rocm(id)?))
}

#[cfg(not(feature = "rocm"))]
fn open_rocm() -> Result<Option<Device>> {
    Ok(None)
}

/// 포크의 `Device::new_metal` 은 장치가 없으면 오류를 돌려주지 않고 패닉합니다 (GPU 없는 가상 맥).
/// 그 경우를 장치가 없는 것으로 보아서 `auto` 가 다음 후보로 넘어가게 합니다.
#[cfg(feature = "metal")]
fn open_metal() -> Result<Option<Device>> {
    let id = gpu_override()?.unwrap_or(0);
    match std::panic::catch_unwind(|| Device::new_metal(id)) {
        Ok(device) => Ok(Some(device?)),
        Err(_) => Ok(None),
    }
}

#[cfg(not(feature = "metal"))]
fn open_metal() -> Result<Option<Device>> {
    Ok(None)
}

#[cfg(feature = "vulkan")]
fn open_vulkan() -> Result<Option<Device>> {
    if candle_core::vulkan_backend::device_count().unwrap_or(0) == 0 {
        return Ok(None);
    }
    Ok(Some(Device::new_vulkan(gpu_override()?.unwrap_or(0))?))
}

#[cfg(not(feature = "vulkan"))]
fn open_vulkan() -> Result<Option<Device>> {
    Ok(None)
}

/// 모델 가중치·어텐션 계산 dtype. logis-center 와 같습니다: ROCm 은 BF16, 그 밖은 F32.
/// Zoro-1 판단 헤드(jev_head)는 명세대로 항상 F32 입니다. (`ai::jev_head` 참고)
pub fn model_dtype(device: &Device) -> DType {
    match device {
        Device::Rocm(_) => DType::BF16,
        _ => DType::F32,
    }
}

/// 정상 종료 경로를 타게 하는 환경 변수 (`0`). 멈춤을 재현할 때만 씁니다.
pub const FAST_EXIT_ENV: &str = "ZOROGRAVITY_ROCM_FAST_EXIT";

#[cfg(windows)]
extern "system" {
    fn GetCurrentProcess() -> *mut std::ffi::c_void;
    fn TerminateProcess(process: *mut std::ffi::c_void, code: u32) -> i32;
}

/// ROCm 을 쓴 Windows 프로세스를 끝냅니다. `main` 의 마지막에 부릅니다. 그 밖의 장치에서는 아무것도 하지 않고 돌아옵니다.
///
/// `ExitProcess` 는 다른 스레드를 먼저 끝낸 뒤 DLL 을 내리는데, `rocblas.dll` 의 종료 처리가 `amdhip64` 를 부르고
/// `amdhip64` 는 이미 사라진 HIP 런타임 스레드를 기다리며 무한히 돕니다. 이렇게 멈춘 프로세스는 코어 하나를 계속 쓰고 `taskkill` 로도 끝나지 않습니다
/// (logis-center 측정 68번 중 2번, zorogravity 에서도 재현). GPU 작업을 끝내고 출력을 비운 뒤 DLL 해제를 건너뛰어 끝냅니다.
pub fn finish_process(device: &Device, code: u32) {
    use std::io::Write;
    if !matches!(device, Device::Rocm(_)) || std::env::var(FAST_EXIT_ENV).is_ok_and(|v| v.trim() == "0") {
        return;
    }
    let _ = device.synchronize_all();
    let _ = std::io::stdout().flush();
    let _ = std::io::stderr().flush();
    #[cfg(windows)]
    unsafe {
        TerminateProcess(GetCurrentProcess(), code);
    }
    #[cfg(not(windows))]
    let _ = code;
}

pub fn is_gpu(device: &Device) -> bool {
    matches!(device, Device::Cuda(_) | Device::Rocm(_) | Device::Metal(_) | Device::Vulkan(_))
}

pub fn backend_name(device: &Device) -> &'static str {
    match device {
        Device::Cuda(_) => "cuda",
        Device::Rocm(_) => "rocm",
        Device::Metal(_) => "metal",
        Device::Vulkan(_) => "vulkan",
        Device::Cpu => "cpu",
        _ => "other",
    }
}

/// 장치 시험(`device_` 로 시작하고 `#[ignore]` 가 붙은 시험)이 쓸 GPU. `ZOROGRAVITY_DEVICE` 로 고릅니다.
/// GPU 를 열지 못하면 실패합니다. 건너뛰고 통과하는 일이 없도록 CI 는 시험이 남긴 `[DEVICE-CHECK]` 줄의 수까지 확인합니다.
#[cfg(test)]
pub(crate) fn test_gpu() -> Device {
    let device = select_device().expect("장치를 열 수 없습니다");
    assert!(is_gpu(&device), "{DEVICE_ENV} 로 GPU 를 고르세요 (지금은 {})", backend_name(&device));
    device
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_device_preferences() {
        assert_eq!(DevicePref::parse("").unwrap(), DevicePref::Auto);
        assert_eq!(DevicePref::parse(" AUTO ").unwrap(), DevicePref::Auto);
        assert_eq!(DevicePref::parse("hip").unwrap(), DevicePref::Rocm);
        assert_eq!(DevicePref::parse("Metal").unwrap(), DevicePref::Metal);
        assert_eq!(DevicePref::parse("Vulkan").unwrap(), DevicePref::Vulkan);
        assert_eq!(DevicePref::parse("cpu").unwrap(), DevicePref::Cpu);
        assert_eq!(DevicePref::parse("CUDA").unwrap(), DevicePref::Cuda);
        assert!(DevicePref::parse("opencl").is_err());
    }

    #[test]
    fn the_gpu_with_the_most_free_memory_wins() {
        assert_eq!(best_gpu(&[(0, 2), (1, 8), (2, 4)]), 1);
        assert_eq!(best_gpu(&[(0, 8), (1, 8)]), 0, "같으면 앞 번호입니다");
        assert_eq!(best_gpu(&[(0, 0), (1, 0)]), 0, "재지 못했으면 0번입니다");
        assert_eq!(best_gpu(&[]), 0);
    }

    #[test]
    fn parses_a_gpu_number() {
        assert_eq!(parse_gpu(" 1 ").unwrap(), 1);
        assert!(parse_gpu("first").is_err());
    }

    #[test]
    fn cpu_preference_always_opens_cpu() {
        assert!(matches!(select_device_with(DevicePref::Cpu).unwrap(), Device::Cpu));
    }

    #[test]
    fn auto_never_fails_even_without_gpu() {
        // GPU 가 없는 CI 에서도 auto 는 CPU 로 떨어져야 합니다.
        assert!(select_device_with(DevicePref::Auto).is_ok());
    }

    #[test]
    fn model_dtype_follows_logis_policy() {
        assert_eq!(model_dtype(&Device::Cpu), DType::F32);
    }
}
