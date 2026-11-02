# GPU 백엔드 (Vulkan · ROCm/HIP · CUDA · Metal)

AI 엔진은 candle 입니다. AMD GPU 연동은 [Qntology/candle-rocm-vulkan](https://github.com/Qntology/candle-rocm-vulkan) 포크(candle 0.11.0 기반)를 씁니다.
포크는 `Cargo.toml` 의 `[patch.crates-io]` 로 `candle-core` · `candle-nn` 을 바꾸고, `candle-rocm` 크레이트로 ROCm 을 붙입니다.

지금은 GitHub 폴더의 **로컬 워킹트리**(`../candle-rocm-vulkan`, path 의존성)를 씁니다.
기준 커밋은 Qntology master 의 `2191c7ad898d866c4db5380ea1fc13a98f435e0c` 이고, 그 위에 Vulkan 셰이더 수정(4.1절)을 sungu-che 포크의 `claude/vulkan-finite-shader-identities` 브랜치(`5ab190e`)에 올렸습니다.
포크를 GitHub 에서 받아 쓰려면 `Cargo.toml` 과 `core/Cargo.toml` 의 path 줄을 git 줄로 되돌리세요.

[logis-center](https://github.com/Qntology/logis-center) 는 **최적화 방식**만 참고했습니다. 장치를 ROCm → Vulkan → CPU 순서로 열고, GPU 커널은 처음 쓰기 전에 CPU 기준값과 비교해 통과한 것만 씁니다. 모델 코드는 가져오지 않았습니다.

## 백엔드 선택

| 백엔드 | 기능 (`core` 의 feature) | 요구 사항 | 이 PC (RX 6600) |
| --- | --- | --- | --- |
| **Vulkan** (기본) | `vulkan` | 드라이버의 `vulkan-1.dll`. SDK 불필요 | ✅ 동작 확인 |
| **ROCm/HIP** | `rocm` | HIP SDK 7.2 (`hipcc`, `amdhip64`, `rocblas`), **MSVC 14.44 툴셋** | ✅ 동작 확인 (2절) |
| **CUDA** | `cuda` | NVIDIA 드라이버, CUDA Toolkit (`nvcc`), Windows 는 MSVC | ⚠️ 실행 미검증. 빌드는 CI 에서 확인 (2.1절) |
| **Metal** | `metal` | Apple Silicon 맥. 따로 설치할 것 없음 | 해당 없음 (2.2절) |

`ZOROGRAVITY_DEVICE` 로 고릅니다. `auto`(기본)는 CUDA → ROCm → Metal → Vulkan → CPU 순서로 처음 열리는 장치를 씁니다. 맥의 Vulkan(MoltenVK)은 Metal 위에서 도는 변환 층이라서 Metal 을 먼저 봅니다.

```bat
set ZOROGRAVITY_DEVICE=vulkan   :: auto | cuda | rocm | metal | vulkan | cpu
```

빌드부터 실행까지 한 번에 하려면 `run/` 의 스크립트를 쓰세요 ([README](../README.md) 의 "실행 스크립트").

## 1. Vulkan 빌드와 실행

```bat
cargo run -p zorogravity-core --example qmatmul_check -- %LOCALAPPDATA%\zorogravity\models\zoro
cargo run -p zorogravity-core --example softmax_check
```

## 2. ROCm 빌드

```bat
call scripts\rocm-env.bat
cargo build -p zorogravity-core --no-default-features --features rocm --example qmatmul_check
set ZOROGRAVITY_DEVICE=rocm
target\debug\examples\qmatmul_check.exe %LOCALAPPDATA%\zorogravity\models\zoro
```

`scripts\rocm-env.bat` 이 하는 일:

- **MSVC 14.44 툴셋**으로 `vcvars64.bat` 을 호출합니다. 기본 툴셋 14.51 은 clang 21 의 HIP 헤더와 충돌합니다.
  - 증상: `__device__ function 'isgreater' cannot overload __host__ __device__ function 'isgreater'` (kernels `unary.hip` 등 20개 이상)
  - 해결: Visual Studio 설치 프로그램에서 "MSVC v143 … (14.44)" 구성요소를 추가합니다. 이 PC 에는 14.44.35207 이 있어서 같은 설정으로 빌드가 통과했습니다.
- `HIP_PATH` 를 HIP SDK 7.2 로 잡고 `bin` 을 `PATH` 에 넣습니다. 런타임에 `rocblas.dll` 이 필요합니다.
- `CANDLE_ROCM_TRACK=legacy`, `HIP_ARCH=gfx1032` (RX 6600). 다른 GPU 는 `HIP_ARCH` 를 바꾸세요.
- `.bat` 파일은 ASCII 로만 씁니다. cmd 가 CP949 콘솔에서 한글이 든 줄을 잘못 읽어 명령처럼 실행하기 때문입니다.

## 2.1 CUDA 빌드

```bat
call scripts\cuda-env.bat
cargo build -p zorogravity-core --no-default-features --features cuda --example qmatmul_check
set ZOROGRAVITY_DEVICE=cuda
target\debug\examples\qmatmul_check.exe %LOCALAPPDATA%\zorogravity\models\zoro
```

- `candle-kernels` 가 빌드 중에 `nvcc` 로 커널을 컴파일합니다. 그래서 드라이버만으로는 빌드되지 않고 CUDA Toolkit 이 있어야 합니다.
- `scripts\cuda-env.bat` 은 `vcvars64.bat` 을 호출하고 `CUDA_PATH` 의 `bin` 을 `PATH` 에 넣습니다. CUDA 13 헤더를 위해 `NVCC_PREPEND_FLAGS=-Xcompiler /Zc:preprocessor` 를 줍니다 (logis-center 와 같습니다).
- 가중치 dtype 은 Vulkan 과 같이 F32 입니다. 비전 타워가 ROCm 의 BF16 보다 VRAM 을 두 배 씁니다.
- 양자화 곱은 `forward_exact` 를 씁니다. CUDA 의 기본 곱은 활성값을 8비트로 줄여서 ROCm · Vulkan 과 확률이 달라지기 때문입니다.
- **미검증**: 이 PC 에는 NVIDIA GPU 와 `nvcc` 가 없습니다. CI 의 `linux-cuda` · `windows-cuda` 가 실제 `nvcc` 로 커널까지 빌드하고 타입 검사를 통과하지만 ([ci.md](ci.md)) 실행은 해 보지 않았습니다. 판단 · 읽기 수치는 없습니다. 자가 검증(`verify`)이 장치 커널과 CPU 결과를 비교합니다. 자동화 기준 0.95 도 CUDA 에서 다시 재야 합니다.

## 2.2 Metal 빌드

```sh
cargo build -p zorogravity-core --release --no-default-features --features metal --example decide_probe
ZOROGRAVITY_DEVICE=metal target/release/examples/decide_probe
```

`run/macos/metal.command` 가 같은 일을 하고, 맥의 앱(`run/macos/app.command`)도 Metal 로 돕니다.

- 포크의 Metal 백엔드를 그대로 씁니다. `candle-metal-kernels` 는 logis-center 에서 검증한 0.11.0 으로 `Cargo.lock` 에 고정했습니다.
- **Apple Silicon 전용입니다.** `candle-metal-kernels` 의 행렬곱은 `simdgroup_matrix` 와 SIMD 폭 32 를 전제로 합니다. `macos-15-intel` 러너(Apple Paravirtualized Graphics, Metal 2)에서는 복사 · 전치 · 평균은 맞지만 행렬곱, softmax, silu, rms_norm, Q8_0 곱이 **오류 없이 틀린 값**을 냈습니다 (상대오차 0.9 이상). 그래서 `open_metal` 이 장치를 열 때 행렬곱 · softmax · silu 를 CPU 와 비교하고, 틀리면 Metal 을 없는 장치로 봅니다. `auto` 는 다음 후보로 넘어가고 `metal` 을 직접 고르면 오류입니다. 실제 Intel 맥의 AMD GPU 로는 재 보지 않았습니다.
- **dtype**: Vulkan · CUDA 와 같이 F32 입니다. 비전 타워의 BF16 가중치는 올릴 때 F32 로 풉니다.
- **메모리**: 통합 메모리라서 포크는 `recommendedMaxWorkingSetSize − currentAllocatedSize` 를 여유로 돌려줍니다. Metal 은 다 쓴 작업 버퍼를 동기화할 때 돌려주므로 `Residency` 는 올리기 전에 동기화합니다. 그러지 않으면 올리는 동안 돌아온 버퍼 때문에 여유가 덜 줄어든 것처럼 보여서 `WeightsLeftOnHost` 로 잘못 거부합니다. `device_loading_is_measured_and_releasing_returns_the_memory` 가 이 경우를 재현합니다.
- **장치가 없는 맥**: GPU 없는 가상 맥에서 포크의 `Device::new_metal` 은 오류 대신 패닉합니다. `open_metal` 이 이를 장치 없음으로 바꿔서 `auto` 는 다음 후보로 넘어갑니다.
- **KV FP8**: 포크에 Metal 용 TurboQuant 커널이 없어서 같은 형식을 CPU 코덱으로 압축합니다 (3.7절). 값은 같고 느립니다. 기본 `auto` 는 메모리가 모자랄 때만 압축합니다.
- **자기 추측**: 재 본 적이 없어서 끕니다 (`speculation_for`). `ZOROGRAVITY_ZORO_SPECULATION` 으로 켤 수 있습니다.

## 3. 실측 (RX 6600)

### 3.1 양자화 행렬곱 (`qmatmul_check`, zoro 의 실제 GGUF 가중치)

`blk.0.ffn_down.weight` [2560×9216]. 장치 시간은 입력이 이미 장치에 있을 때의 행렬곱 시간입니다. ROCm 1행은 3.6절의 커널로 0.180 → 0.078 ms 가 되었습니다 (아래 표는 그 전 값).

| 백엔드 | 형식 | 1행 (디코딩) 장치 / CPU | 64행 (프리필) 장치 / CPU |
| --- | --- | --- | --- |
| Vulkan | **Q8_0** | 0.160 ms / 3.58 ms | 2.15 ms / 152 ms |
| Vulkan | Q4_K | 2.19 ms / 2.47 ms | 144 ms / 198 ms |
| ROCm | Q8_0 | 0.180 ms / 2.45 ms | 2.18 ms / 152 ms |
| ROCm | Q4_K | 0.312 ms / 3.34 ms | 2.33 ms / 141 ms |

- **Vulkan 은 Q8_0 만 GPU 에서 돕니다.** Q4_K 는 CPU 로 계산됩니다 (CPU 와 결과 오차가 정확히 0, 시간도 같음). ROCm 은 둘 다 GPU 에서 돕니다.
- Q8_0 의 GPU 결과는 F32 로 푼 정확한 곱과 2.5e-7 ~ 1.8e-6 안에서 일치합니다. GPU↔CPU 의 0.5% 차이는 CPU 경로가 활성값을 양자화하기 때문에 생긴 CPU 쪽 오차입니다.
- 이 결과가 q8 을 기본으로 정한 근거의 하나입니다 ([model-quantization.md](model-quantization.md)).

### 3.2 softmax · reduce (`softmax_check`, 2048×1024)

| 실행 | 결과 |
| --- | --- |
| Vulkan 기본 | 장치 결과가 CPU 와 같음 (상대오차 0), 1979 µs (CPU 1521 µs). 네이티브 커널이 꺼져 있어 셰이더를 타지 않습니다 |
| Vulkan + `CANDLE_VULKAN_NATIVE=1` (수정 전) | 폴백 52회 (softmax) + 1회 (reduce max · min). 2360 µs |
| Vulkan + `CANDLE_VULKAN_NATIVE=1` (수정 후) | 폴백 0회. softmax 상대오차 4.8e-7, reduce max · min 상대오차 0. 1597 µs |

### 3.3 작은 연산은 GPU 에 올리지 않습니다

크기가 작은 행렬곱(0.2 MFLOP)은 GPU 호출 비용이 계산보다 커서 CPU 가 빨랐습니다 (Vulkan 1045 µs, ROCm 191 µs, CPU 약 16–23 µs). 판단 헤드(`jev_head`)가 이런 크기라서 CPU, F32 로 둡니다. 선택지 4개에 호출당 139 µs 입니다 ([jev-head.md](jev-head.md)). GPU 의 이득은 위 3.1 의 큰 양자화 행렬곱에서 나옵니다.

### 3.4 Vulkan 의 메모리 배치

비전 타워처럼 일반 가중치를 쓰는 모델은 Vulkan 이 크게 느립니다. 양자화 가중치는 VRAM 에 올라가지만 일반 가중치와 활성값은 시스템 메모리에 있어서입니다.

| 작업 (RX 6600) | ROCm | Vulkan |
| --- | --- | --- |
| SigLIP2 비전 (256 패치) | 0.13 s | 0.94 s |
| gocr 비전 (2628 패치) | 1.5 s | 17 s |
| 합성 서식 한 장 OCR (패치 18,829) | 20 s | 121 s |

**문서 읽기는 ROCm 에서 쓰세요.** 아래는 양자화 가중치까지 포함한 Vulkan 의 메모리 배치입니다.

독립 GPU(이 PC 의 RX 6600)에서 Vulkan 은 **양자화(Q8_0) 가중치만 VRAM 에 올립니다.** 일반 텐서(활성값 · F32 정규화 가중치 · BF16 비전 타워)는 포크의 `alloc_buffer` 가 호스트에서 보이는 메모리(시스템 RAM)에 잡습니다. `Device::mem_info()` 는 Q8_0 가중치의 장치 힙만 보여 주므로, 512 MiB 일반 텐서를 만들어도 여유가 변하지 않습니다 (`CANDLE_VULKAN_NATIVE=1` 이어도 같음).
일반 텐서의 할당량은 `Device::Vulkan(d).allocated_bytes()` 로 봅니다 (`Residency::tensor_buffer_bytes`). ROCm 은 모든 텐서가 장치 메모리에 있습니다. 자세한 측정은 [weight-residency.md](weight-residency.md).

### 3.5 백본 실측 (zoro q8, 프롬프트 149 토큰)

| 백엔드 | 판단 1회 | 비고 |
| --- | --- | --- |
| ROCm | 0.67 s | 델타 규칙 · RMSNorm 융합 커널이 자가 검증을 통과 (상대오차 3e-7 · 2e-7) |
| Vulkan | 1.5 ~ 1.7 s | 선형 주의의 델타 규칙은 입력을 CPU 로 내려 CPU 커널로 계산 (층당 약 14 ms). `CANDLE_VULKAN_NATIVE=1` 로도 시간은 거의 같음 (1.5 s) |
| CPU | 약 100 s | 폴백용 |

세 백엔드의 다음 토큰 확률은 같습니다 (`" Paris"` 37.3 / 37.3 / 38.9%). 자세한 내용은 [zoro-backbone.md](zoro-backbone.md).

### 3.6 ROCm: 입력 행이 적은 Q8_0 행렬곱과 버퍼 캐시

토큰을 하나씩 읽을 때는 입력이 1 ~ 8행인 행렬곱이 시간 대부분입니다. 여기서 두 가지 병목을 찾아 고쳤습니다.

**다중 행 커널 (포크 `qmm_q8_0_*`).** 예전 커널(`qmv`)은 입력 행마다 가중치 표 전체를 다시 훑고, 원소마다 양자화를 풀고, 출력 원소마다 작업 그룹 동기화를 했습니다. 시간이 행 수에 비례해서 8행이 1행의 4배였습니다. 새 커널은 출력 원소 하나를 작업 항목 하나가 맡아 가중치 행을 한 번만 읽고 입력 행 전부와 곱합니다. 포크 시험이 1 ~ 22행 모두 역양자화 기준값과 맞는지, 행 수가 달라도 같은 행이 비트까지 같은지 확인합니다.

| 입력 행 | 1 | 2 | 4 | 8 | 16 | 20 | 24 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 예전 (ms) | 92 | 144 | 220 | 370 | 355 | - | - |
| 새 커널 (ms) | 55 | 61 | 81 | 120 | 209 | 239 | 1586 |

zoro 32층 순전파 하나의 시간입니다 (층마다 기다려서 재서 실제보다 깁니다). 21행부터는 예전처럼 역양자화 + GEMM 으로 갑니다. 24행부터 커널이 크게 느려지기 때문입니다.

역양자화도 블록 단위로 바꿔 보았지만 4배 느려져서 되돌렸습니다 (355 → 1234 ms). 원소 단위 커널은 쓰기가 한 캐시 라인으로 묶입니다.

**버퍼 캐시.** 포크는 해제한 버퍼를 재사용합니다. 기본 한도 256 MiB 를 프리필의 임시 버퍼(역양자화 타일 한 장에 90 MB)가 채우면, 이어지는 작은 버퍼가 캐시에 들어가지 못해 할당마다 앞선 GPU 작업을 기다립니다. zorogravity 는 시작할 때 한도를 1 GiB 로 잡습니다 (`CANDLE_ROCM_ALLOC_CACHE_MB`).

| 한도 | 256 MiB | 512 MiB | 1 GiB | 2 GiB |
| --- | --- | --- | --- | --- |
| 토큰당 | 39.4 ms | 37.4 ms | 35.7 ms | 35.6 ms |

프리필도 1.2 s 에서 0.96 s 로 줄었습니다. 가중치를 내릴 때 캐시도 함께 비워지고, 메모리가 모자라면 할당이 캐시를 먼저 돌려줍니다.

두 가지 모두 ROCm 만의 변경입니다. 커널은 입력 20행까지만 쓰므로 판단(프리필)의 확률은 그대로입니다.

### 3.7 KV 캐시 FP8

**ROCm · Vulkan · CUDA 모두 가능합니다.** 포크가 세 백엔드 모두에 FP8 압축 · 복원 · 압축 행 어텐션 커널(`tq_quantize` · `tq_dequantize` · `tq_attention`)을 갖고 있습니다 (Vulkan 은 GLSL 포팅). Metal 에는 이 커널이 없어서 포크의 CPU 코덱으로 압축합니다. `ai::kv_turbo` 가 logis-center 의 `kv_turbo.rs` · `accel_check.rs` 와 같은 정책으로 씁니다. 스케일 없는 `to_dtype(F8E4M3)` 캐스팅은 쓰지 않습니다. 448 을 넘으면 포화하고 작은 값은 사라지기 때문입니다.

| `ZOROGRAVITY_KV_PRECISION` | 동작 |
| --- | --- |
| `auto` (기본) | 원본(F32) KV + `ZOROGRAVITY_VRAM_HEADROOM_MB`(640 MB) 가 장치 여유 메모리에 들어가면 원본, 아니면 FP8 |
| `exact` | 항상 원본 |
| `compressed` (`fp8`) | GPU 에서 항상 FP8 |

- **형식**: K · V 모두 벡터별 스케일 FP8(e4m3), 회전 없음. 토큰 · 헤드 하나가 `head_dim` + 8 바이트라서 F32 의 약 1/4 입니다. `CANDLE_TURBO_QUANT_*` 로 형식을 지정하면 포크의 설정을 따릅니다.
- **적용 범위**: gocr 언어 디코더의 KV 캐시와 zoro 전체 주의 층의 KV 캐시입니다. 선형 주의의 순환 상태와 판단(프리필 한 번)은 그대로입니다. CPU 는 압축하지 않습니다.
- **계산 경로** (`turbo_kv_step` 과 같음): 이력이 없는 프리필은 원본으로, 새 토큰 하나는 압축 행 위에서 바로, 여러 토큰(추측 · 블록 초안)은 이력만 풀어서 새 토큰의 원본과 함께 계산합니다.
- **자가 검증**: GPU 압축 · 복원을 CPU 코덱과 비교하고(`kv_fp8_pack_d*`), 압축 행 어텐션을 CPU 기준과 비교합니다(`kv_fp8_attn_*`). 압축이 틀리면 원본 KV 를 쓰고, 어텐션만 틀리면 압축은 유지하고 풀어서 계산합니다.
- **실측 감시**: 프리필의 마지막 토큰을 압축 행으로도 계산해 오차를 `[ZOROGRAVITY-KV] FP8 실측 오차` 로 남깁니다. 50% 를 넘으면 그 장치의 압축을 끕니다.

**FP8 은 근사입니다.** logis-center 측정에서 짧은 JSON 추출은 원본과 다음 토큰이 100% 같았지만 수천 토큰의 긴 서술에서는 4% 안팎의 스텝에서 1순위 토큰이 바뀌었습니다. 그래서 기본은 메모리가 모자랄 때만 압축합니다. 압축 중에는 추측의 양에 따라 글이 달라질 수 있고(읽는 묶음에 따라 새 토큰이 원본이거나 FP8 입니다), 공문서 초안의 재현성이 중요하면 `exact` 를 쓰세요.

**미검증**: CPU 코덱 단위 테스트와, CI 의 lavapipe(소프트웨어 Vulkan)에서 합성 입력으로 한 장치 시험만 통과했습니다. 실제 GPU · 실제 모델의 속도, 메모리, 오차, 글 변화는 재지 않았습니다. Vulkan 은 일반 텐서가 시스템 메모리에 있어서 `auto` 가 보는 장치 여유와 KV 가 실제로 쓰는 메모리가 다릅니다 (3.4절).

## 4. 알려진 문제

### 4.1 Vulkan 셰이더의 무한대 리터럴 (수정 완료, sungu-che 포크에 올림)

- **증상**: 네이티브 GPU 커널이 켜졌을 때 (`CANDLE_VULKAN_NATIVE=1`) softmax 와 reduce max · min 이 호출마다 CPU 로 되돌아갑니다. `CANDLE_VULKAN_DEBUG=1` 이면 `Float literal is infinite` 오류가 보이고, 디버그를 끄면 조용히 느려집니다 (호출당 약 0.8 ms 추가).
- **기본 경로는 영향이 없습니다.** 이 PC 의 RX 6600 처럼 독립 GPU 에서는 네이티브 커널이 기본으로 꺼져 있어 셰이더를 타지 않습니다.
- **원인**: `candle-core/src/vulkan_backend/shaders.rs` 의 `-1.0 / 0.0` (reduce max 항등원, softmax 최댓값 초기값)과 `1.0 / 0.0` (reduce min 항등원). naga 가 이를 무한대 리터럴로 보고 셰이더 파싱을 거부합니다. 오류 위치(바이트 1089–1099)가 `-1.0 / 0.0` 열 개 글자와 정확히 맞았습니다.
- **수정**: 유한한 항등원 `-3.0e38` (max) 과 `3.0e38` (min) 으로 바꿨습니다. 유한한 입력에서는 결과가 같습니다.
- **회귀 테스트**: 포크의 `softmax_and_reduce_shaders_compile` (`candle-core/tests/vulkan_tests.rs`). 수정을 빼면 `softmax shader fell back to the CPU` 로 실패하고, 수정을 넣으면 통과합니다. 포크의 Vulkan 테스트 20개가 모두 통과합니다.
- **커밋**: sungu-che/candle-rocm-vulkan 의 `claude/vulkan-finite-shader-identities` 브랜치, 커밋 `5ab190e`. 풀 리퀘스트는 만들지 않았습니다.
- **logis-center**: 고정 커밋 `60ec257e` 에도 같은 셰이더 문장이 있습니다. 그 프로젝트가 네이티브 Vulkan 커널을 켠 적이 없다면 이 문제를 볼 일이 없었을 것입니다 (확인하지는 않았습니다).

### 4.3 Vulkan 은 GPU 메모리가 모자라면 조용히 호스트에 둡니다

`QVulkanStorage::from_bytes` 는 Q8_0 가중치를 GPU 에 올리다 실패하면(`Err(_) => {}`) 오류 없이 호스트 메모리에 둡니다. 그 가중치를 쓰는 층만 CPU 경로로 계산되어, 같은 GPU 를 다른 프로그램이 쓸 때 판단 확률이 실행마다 달라졌습니다 (재현과 수치는 [weight-residency.md](weight-residency.md)).
zorogravity 는 포크를 고치지 않고 올리기 전후의 장치 여유 메모리를 검사해서 거부합니다 (`DeviceMemoryShort`, `WeightsLeftOnHost`). 포크 쪽 개선안은 장치 상주 여부를 알려 주는 `QTensor` 접근자나, 실패를 오류로 돌려주는 엄격 모드입니다 (아직 안 함).
같은 이유로 포크의 기본 실행 모드(`ExecMode::Auto`)는 "연산 크기와 가중치 재사용에 따라 연산마다 경로를 고르고, 같은 행도 배치가 다르면 다른 비트가 나올 수 있다"고 밝혀 둡니다. 우리 판단 경로는 한 프로세스 안에서 반복하면 비트 단위로 같고, GPU 에 모두 올라간 상태에서는 프로세스 사이에도 같았습니다 (8번 반복).

### 4.2 자가 검증이 잡지 못한 것

`ai::verify::trusted` 는 **결과의 정확성**만 봅니다. 4.1처럼 결과는 맞지만 느린 경로(CPU 폴백)는 통과로 기록됩니다. 속도 회귀는 `qmatmul_check` 와 `softmax_check` 의 지연 시간으로 따로 봐야 합니다.

### 4.4 ROCm 프로세스가 끝나지 않습니다 (Windows)

일을 다 끝낸 프로세스가 가끔 종료 단계에서 멈춥니다. `ExitProcess` 가 다른 스레드를 먼저 끝낸 뒤 DLL 을 내리는데, `rocblas.dll` 의 종료 처리가 `amdhip64` 를 부르고 `amdhip64` 는 이미 사라진 HIP 스레드를 기다리며 무한히 돕니다. 코어 하나를 계속 쓰고 `taskkill` 로도 끝나지 않으며, 출력 파이프도 닫히지 않습니다 (logis-center 측정 68번 중 2번, zorogravity 에서도 두 번 재현).

- **대응**: `main` 끝에서 `accel::finish_process(&device, 0)` 을 부릅니다. GPU 작업을 끝내고 출력을 비운 뒤 DLL 해제를 건너뛰어 끝냅니다. 예제는 모두 이렇게 합니다. 앱에 붙일 때도 종료 경로에 넣어야 합니다.
- `cargo test` 는 표준 종료를 타므로 ROCm 으로 돌리면 같은 일이 생길 수 있습니다. 결과는 이미 출력되어 있습니다.
- 이미 멈춘 프로세스는 마지막 스레드를 `TerminateThread` 로 끝내면 사라집니다.

## 5. 환경 변수 요약

| 변수 | 기본값 | 뜻 |
| --- | --- | --- |
| `ZOROGRAVITY_DEVICE` | `auto` | `auto` · `cuda` · `rocm` · `metal` · `vulkan` · `cpu` |
| `ZOROGRAVITY_GPU` | 자동 | GPU 번호. 없으면 CUDA · ROCm 은 여유 VRAM 이 가장 많은 GPU, Metal · Vulkan 은 0번 (logis-center 와 같음) |
| `ZOROGRAVITY_ACCEL_CHECK` | 켜짐 | `off` 이면 커널 자가 검증을 건너뜁니다 (진단용으로만) |
| `CANDLE_VULKAN_NATIVE` | 자동 (독립 GPU 는 꺼짐) | `1` 이면 Vulkan 네이티브 element-wise · softmax · reduce 커널을 강제로 씁니다 |
| `CANDLE_VULKAN_DEBUG` | 꺼짐 | `1` 이면 커널이 CPU 로 되돌아간 이유를 출력합니다 |
| `CANDLE_VULKAN_*` | 포크 기본 | 포크의 기타 Vulkan 튜닝 변수 (`…_MIN_ELEMS`, `…_MIN_FLOPS` 등). 포크 README 참고 |
| `CANDLE_ROCM_TRACK` | `auto` | `legacy` (HIP SDK 7.x) 또는 `core` (ROCm 10.x) |
| `CANDLE_ROCM_ALLOC_CACHE_MB` | 1024 (zorogravity 가 정함, 포크 기본은 256) | 해제한 버퍼를 재사용하는 캐시의 한도 |
| `ZOROGRAVITY_ZORO_SPECULATION` | 장치별 기본값 | 글쓰기의 추측 `마스크:탐색:폭`. `0:0:0` 이면 한 토큰씩 읽습니다 |
| `ZOROGRAVITY_WARM` | 켜짐 | `0` 이면 가중치 파일을 배경에서 미리 읽지 않습니다 |
| `ZOROGRAVITY_KV_PRECISION` | `auto` | KV 캐시 정밀도 `auto` · `exact` · `compressed` (3.7절) |
| `CANDLE_TURBO_QUANT` | `on` | `off` 이면 KV 를 압축하지 않습니다 |
| `CANDLE_TURBO_QUANT_*` | 벡터별 FP8 | 포크의 KV 압축 형식 · 비트 · 회전 변수. 포크 README 참고 |
| `HIP_ARCH` | 자동 감지 | GPU 타깃. RX 6600 은 `gfx1032` |
| `CUDA_PATH` | 설치 프로그램이 설정 | CUDA Toolkit 위치 (`nvcc`, `cublas`) |
