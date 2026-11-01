# zorogravity — GovGraph-ZOROGRAVITY 민원 심사 엔진

공무원 민원 심사(5-Gate)를 Zoro-1 판단 헤드로 돕는 로컬 엔진입니다. 원본 명세는 `zorogravity.txt` 입니다.

- **AI 엔진**: [candle](https://github.com/huggingface/candle) (Rust). AMD GPU 는 [candle-rocm-vulkan](https://github.com/Qntology/candle-rocm-vulkan) 포크로 ROCm/HIP 과 Vulkan 에 연결합니다.
- **모델**: 앱 데이터 폴더의 `models/siglip2` (크롭), `models/gocr` (문서 인식), `models/zoro` (판단). 모두 GGUF 로 바꿔 쓰고, 지금은 **q8** 입니다 (비전 가중치는 BF16). 폴더와 내려받기는 아래 "모델 폴더와 다운로드"에 있습니다. 다른 모델(Qwen3.5 등)은 쓰지 않습니다. 명세의 `indra-ocr` 은 라이선스 문제로 쓰지 않습니다.
- **문서 읽기**: 기울기 보정 → 잉크 분석으로 조각 나누기 → SigLIP2 로 제목 · 표 · 서명 · 안내문 판정 → gocr 비전 → gocr 언어. [docs/document-crop.md](docs/document-crop.md)
- **zoro 백본**: 텍스트 순전파를 직접 구현했습니다 (`core/src/ai/zoro`). 실제 가중치로 "The capital of France is" → `" Paris"` 를 확인했습니다. [docs/zoro-backbone.md](docs/zoro-backbone.md)
- **KV 캐시 FP8**: 원본 KV 가 장치 메모리에 들어가지 않으면 벡터별 FP8 로 압축합니다 (F32 의 1/4). ROCm · Vulkan · CUDA 모두 포크의 커널을 쓰고, 정책은 logis-center 와 같습니다. [docs/gpu-backends.md](docs/gpu-backends.md) 3.7절
- **가중치 관리**: 모델은 **한 번에 하나**, 모델 안에서도 **지금 단계가 쓰는 묶음만** 올리고 단계가 바뀌면 내립니다 (크롭은 SigLIP2 비전 타워만, OCR 은 비전 → 비전을 내림 → 언어, zoro 는 텍스트 블록만). [docs/weight-residency.md](docs/weight-residency.md)
- **최적화 방식**: [logis-center](https://github.com/Qntology/logis-center) 는 방식만 참고했습니다. 장치 선택 순서(ROCm → Vulkan → CPU), 커널 자가 검증, 양자화 가중치의 GPU 실행, 쓰는 타워만 올리고 즉시 반환하는 단계 전환입니다. 모델 코드는 가져오지 않았습니다.
- **자동화 모드**: 기본은 보조 모드입니다. 자동화 모드는 승인과 법령 근거가 있어야 켜지고, 불리한 처분은 항상 공무원이 확정합니다. 설계 전체는 [docs/automation-mode.md](docs/automation-mode.md).

## 현재 상태

| 영역 | 상태 |
| --- | --- |
| 5-Gate 규칙 엔진 (Gate 0~5) · 엔드투엔드 (명세 사례) | 구현 · 단위 테스트 267개 + 엔드투엔드 23개 통과 (모델 파일이 필요한 점검은 `--ignored`, `real_models` 14개) |
| 자동화 정책 · 감사 로그 · 비식별 해시 | 구현 |
| 관계망 멀티홉 · 동명이인 판별 (사건 범위) | 구현 |
| 판단 모델 인터페이스 (`DecisionModel`) | 구현. 게이트가 식별 정보 없는 사실과 선택지로 질문을 만듭니다 |
| `jev_head` (질의·키 투영, 0-Generation 포인터) | 구현 (`JevHead`, `PointerModel`). 형식은 조합 1,680개를 재서 골랐습니다 (선택지는 닫는따옴표, 스케일 1/32). [docs/jev-head.md](docs/jev-head.md). **합성 · 명세 사례로만 검증** |
| **zoro 백본** (판단 · 글쓰기) | **구현** (`ai::zoro`, 직접 작성). 합성 모델은 스칼라 기준 구현과 일치. KV 캐시 · 순환 상태 · 출력 투영으로 글을 씁니다. 자기 추측 디코딩으로 토큰당 30 ms ([docs/zoro-backbone.md](docs/zoro-backbone.md)). 비전 입력과 블록 디퓨전 생성은 없음 |
| **모델이 쓰는 초안** (민원 요약 · 소명 정규화 · 심사 의견) | **구현** (`drafting`, `factcheck`). 숫자 · 날짜를 원문과 대조합니다. 초안은 처분을 바꾸지 않습니다. [docs/automation-mode.md](docs/automation-mode.md) 10절 |
| **가중치 상주 관리** (모델 하나 · 단계별 묶음 · 스트리밍 · VRAM 반환) | **구현** (`ai::residency`). 실제 GPU 에서 올리고 내릴 때 VRAM 이 돌아오는 것을 측정 |
| 문서 인식 (gocr) | **구현** (`ai::gocr`, 직접 작성). 비전 인코더(같은 격자 묶음) · 언어 디코더 · KV 캐시 · 블록 디퓨전 초안 + 자기회귀 검증 · 신뢰도 조기 종료 |
| 문서 크롭 (SigLIP2) | **구현** (`ai::siglip2`, 직접 작성). 패치 경쟁(NMS)과 크롭 NMS 포함. 합성 서식으로 평가했고 **실제 스캔으로는 학습 · 평가 전**. [docs/document-crop.md](docs/document-crop.md) |
| 문서 묶음 읽기 | **구현** (`ai::document`). 픽셀 계산은 병렬, 모델은 단계마다 한 번만 올립니다. 읽은 글을 항목별 사실로 바꾸는 단계는 미구현 |
| 모델 양자화 (GGUF Q4 · Q8, 비전 BF16) | 구현 · gocr · zoro 모두 q8 선택. [docs/model-quantization.md](docs/model-quantization.md) |
| 장치 선택 · 커널 자가 검증 (ROCm · Vulkan · CPU) | 구현 · 실측은 아래 |
| HWP 문서 생성 (rhwp) | 구현 (`tools/hwp-builder`, 내부결재용 · 교부용, 금지어 · 확정 수치 검사). 심사 흐름과 앱 편집기에는 아직 연결 안 됨 |
| 모델 다운로드 · 삭제 (AppData, 진행률 표시) | 구현 (`tools/model-settings`, 앱 설정 화면) |
| KV 캐시 FP8 (ROCm · Vulkan · CUDA) | 구현 (`ai::kv_turbo`). CPU 단위 테스트만 통과했고 **GPU 에서는 아직 돌려 보지 않았습니다** |
| Tauri 앱 (`app`) | 두 창 틀과 모델 설정만 구현. 심사 화면은 미구현 |
| 영속 저장 (SQLite) | 미구현 (메모리) |
| 사건 간 관계망 누적 | 미구현 (법무 검토 필요) |

## 실행 스크립트

백엔드마다 빌드부터 실행까지 한 번에 합니다. Windows 는 더블클릭, 맥은 `.command` 더블클릭, 리눅스는 터미널에서 실행합니다.

| OS | CUDA | ROCm | Vulkan |
| --- | --- | --- | --- |
| Windows | `run\windows\cuda.bat` | `run\windows\rocm.bat` | `run\windows\vulkan.bat` |
| Linux | `run/linux/cuda.sh` | `run/linux/rocm.sh` | `run/linux/vulkan.sh` |
| macOS | 지원 안 됨 | 지원 안 됨 | `run/macos/vulkan.command` |

앱은 `run\windows\app.bat` · `run/linux/app.sh` · `run/macos/app.command` 로 실행합니다.

```bat
:: 기본은 판단 경로 전체를 도는 decide_probe 입니다. 예제 이름과 인자를 주면 그 예제를 돌립니다.
run\windows\vulkan.bat
run\windows\rocm.bat read_page 이미지.jpg
```

- 결과물은 `target-cuda` · `target-rocm` · `target` 의 `release\examples` 에 생깁니다. 아직 앱 본체가 없어서 실행 대상은 `core/examples` 의 예제입니다.
- 공통으로 이 저장소 옆에 `candle-rocm-vulkan` 이 있어야 하고, 모델은 앱 데이터 폴더에 받아 두어야 합니다 (아래 "모델 폴더와 다운로드").
- **CUDA** 는 CUDA Toolkit(`nvcc`)이, **ROCm** 은 HIP SDK 가, **Vulkan** 은 GPU 드라이버가 있어야 합니다. 맥의 Vulkan 은 `brew install vulkan-loader molten-vk` 로 설치합니다.
- 맥의 `cuda.command` · `rocm.command` 는 안내만 하고 끝납니다. macOS 용 ROCm 은 없고 CUDA 는 10.2 를 끝으로 지원이 끊겼습니다.
- 이 PC 에서 돌려 본 것은 Windows 의 Vulkan · ROCm 입니다. CUDA, 리눅스, 맥은 **실행해 보지 않았습니다**. 특히 맥의 Vulkan(MoltenVK)은 이 프로젝트에서 한 번도 측정한 적이 없습니다.

## 모델 폴더와 다운로드

모델은 저장소가 아니라 OS 의 앱 데이터 폴더에 둡니다 (logis-center 와 같은 방식). 예제와 테스트도 이 폴더를 씁니다.

| OS | 위치 |
| --- | --- |
| Windows | `%LOCALAPPDATA%\zorogravity\models` (`C:\Users\<사용자>\AppData\Local\zorogravity\models`) |
| macOS | `~/Library/Application Support/zorogravity/models` |
| Linux | `$XDG_DATA_HOME/zorogravity/models` (없으면 `~/.local/share/zorogravity/models`) |

`ZOROGRAVITY_MODELS_DIR` 로 폴더를 바꿀 수 있습니다.

앱을 켜면 없는 모델을 자동으로 내려받고, 공무원 창의 설정 화면에 진행률을 표시합니다. 삭제도 이 화면에서 합니다. 예제만 돌릴 때는 앱으로 모델을 먼저 받아 두어야 합니다.

| 모델 | 내려받는 곳 | 크기 |
| --- | --- | --- |
| `zoro` | [Qntology/zoro-1-gguf](https://huggingface.co/Qntology/zoro-1-gguf) | 5.2 GB |
| `gocr` | [Qntology/GravityOCR-KO-gguf](https://huggingface.co/Qntology/GravityOCR-KO-gguf) | 1.6 GB |
| `siglip2` | 없음. 직접 변환해서 `models/siglip2` 에 넣습니다 ([docs/document-crop.md](docs/document-crop.md)) | - |

- `.tmp` 로 받아서 크기가 맞을 때만 이름을 바꾸므로 받다 만 파일이 모델로 열리지 않습니다. 끊겨도 다시 누르면 이어 받습니다.
- `ZOROGRAVITY_MODEL_MIRROR` 에 같은 경로 구조의 미러 주소를 줄 수 있습니다.

## 빌드와 실행

```bat
:: CPU + Vulkan (기본). 드라이버의 vulkan-1.dll 만 있으면 됩니다. HIP SDK 는 필요 없습니다.
:: 로컬 candle 포크를 씁니다. GitHub 폴더에 candle-rocm-vulkan 이 있어야 합니다 (Cargo.toml 의 path).
cargo test --workspace

:: 실제 모델 파일(models/)과 GPU 로 하는 점검 (오래 걸림, 병렬 금지)
cargo test -p zorogravity-core --test real_models -- --ignored --nocapture --test-threads 1

:: 점검 예제
cargo run -p zorogravity-core --example zoro_probe -- %LOCALAPPDATA%\zorogravity\models\zoro "The capital of France is" 6   :: 다음 토큰
cargo run -p zorogravity-core --example decide_probe                                             :: 판단 경로 전체
cargo run -p zorogravity-core --example residency_check                                          :: 올리고 내리기 + VRAM 반환
cargo run -p zorogravity-core --example qmatmul_check -- %LOCALAPPDATA%\zorogravity\models\zoro                             :: 양자화 행렬곱
cargo run -p zorogravity-core --example read_page -- 이미지.jpg                                   :: 페이지 끝까지 읽기

:: ROCm/HIP (HIP SDK 7.2 필요). MSVC 14.44 툴셋으로 빌드하고, 기본 빌드 캐시와 섞이지 않게 별도 폴더를 씁니다.
call scripts\rocm-env.bat
set CARGO_TARGET_DIR=target-rocm
cargo build -p zorogravity-core --no-default-features --features rocm --example zoro_probe --example decide_probe
set ZOROGRAVITY_DEVICE=rocm
target-rocm\debug\examples\decide_probe.exe

:: 모델 변환 (models 폴더의 safetensors → GGUF)
cargo build -p zorogravity-gguf
target\debug\zorogravity-gguf.exe auto %LOCALAPPDATA%\zorogravity\models\zoro
```

장치 선택은 `ZOROGRAVITY_DEVICE=auto|cuda|rocm|vulkan|cpu` 입니다. 기본 `auto` 는 CUDA → ROCm → Vulkan → CPU 순서로 처음 열리는 장치를 씁니다.
메모리 관련 환경 변수(`ZOROGRAVITY_VRAM_BUDGET_MB`, `ZOROGRAVITY_ZORO_STREAM_BLOCKS`, `ZOROGRAVITY_KV_PRECISION` 등)는 [docs/weight-residency.md](docs/weight-residency.md) 에 있습니다.

판단 모델을 엔진에 꽂는 방법:

```rust
let residency = Residency::shared(device.clone(), Budget::auto(&device));   // OCR 과 공유해야 "한 번에 한 모델"이 지켜집니다
let model = zorogravity_core::ai::zoro::open_decision_model(&model_dir("zoro"), residency)?;
let engine = Engine::new(salt, AutomationPolicy::default(), Some(Box::new(model)));
```

## 실측 (AMD Radeon RX 6600 8 GB, zoro q8)

**판단 1회** (Gate 3 질문, 프롬프트 149 토큰, 가중치가 올라온 뒤 호출 한 번):

| 백엔드 | 첫 호출 (가중치 적재 포함) | 이후 호출 |
| --- | --- | --- |
| ROCm | 3.1 s | **0.62 s** |
| Vulkan | 3.7 s | 1.3 s |
| CPU (Ryzen 5 5600X) | 약 100 s | 약 100 s |

**가중치 사용량**: zoro 판단은 텍스트 블록 3631 MB 만 올립니다. 비전 타워(636 MB BF16)와 임베딩 표(643 MB)는 올리지 않고 임베딩은 필요한 행만 파일에서 읽습니다. 예산이 빠듯하면 블록 8개씩 4구간으로 스트리밍해서 최대 908 MB 만 상주시킵니다 (판단 결과는 같음).

**글쓰기** (ROCm): 토큰당 30 ms, 고충민원 요약 건당 2.5 ~ 2.9 s, 소명 정규화 건당 0.8 s. 같은 글이 나오고 부서 · 소명 정확도도 그대로입니다. 처음에는 토큰당 78 ms 였습니다. 이득의 대부분은 Q8_0 다중 행 커널과 버퍼 캐시이고 자기 추측은 36 → 30 ms 입니다 ([docs/zoro-backbone.md](docs/zoro-backbone.md)).

**전환 시간** (zoro → SigLIP2 비전 → gocr 비전 → gocr 언어 한 바퀴, ROCm): 1.9 s. 프로세스 첫 바퀴도 같고, 파일이 OS 캐시에 없으면 2.1 s 입니다 (처음에는 2.3 · 4.4 · 5.9 s). [docs/weight-residency.md](docs/weight-residency.md)

**모델 전환** (gocr 비전 → gocr 언어 → zoro → gocr 비전 → 모두 내림): 두 모델이 함께 올라온 적이 없고, 내린 VRAM 이 전부 돌아옵니다 (Vulkan 7146 → 3300 → 7146 MB). 표는 [docs/weight-residency.md](docs/weight-residency.md).

**문서 읽기** (합성 서식 64장, ROCm): 8장씩 묶어 읽으면 페이지당 8.7 s, 한 장씩이면 9.8 s. 크롭 면적은 페이지의 27% 이고 값 · 서명 · 제목을 놓치지 않습니다 (처음 보는 서식 종류 기준). 값을 글자 그대로 읽은 비율은 85.9%, 흉내 낸 손글씨를 빼면 91.7% 입니다. Vulkan 은 같은 일이 6배 느려서 문서 읽기는 ROCm 에서 쓰세요. 실제 스캔으로는 아직 재지 않았습니다. [docs/document-crop.md](docs/document-crop.md)

**양자화 행렬곱** (zoro q8 의 실제 가중치 [2560×9216]):

| 백엔드 | 1행 (디코딩) 장치 / CPU | 64행 (프리필) 장치 / CPU |
| --- | --- | --- |
| Vulkan, Q8_0 | 0.160 ms / 3.58 ms | 2.15 ms / 152 ms |
| ROCm, Q8_0 | 0.078 ms / 2.45 ms | 2.19 ms / 152 ms |

- Q8_0 의 GPU 결과는 F32 로 푼 정확한 곱과 1e-6 안에서 일치합니다.
- Vulkan 은 **Q8_0 만 GPU 에서 돕니다.** Q4_K 는 CPU 로 계산됩니다. 이것도 q8 을 고른 이유입니다.
- `jev_head` 는 CPU 에서 선택지 4개에 139 µs 입니다 (명세 목표 15 ms).

표와 원인 분석은 [docs/gpu-backends.md](docs/gpu-backends.md) 에 있습니다.

## 디렉토리

```
zorogravity/
├── Cargo.toml              워크스페이스 · 로컬 candle 포크 패치
├── core/                   zorogravity-core: 엔진 코어 (Tauri 없이 테스트 가능)
│   ├── src/
│   │   ├── ai/             accel(장치 선택) · verify(커널 자가 검증) · weights(GGUF 에서 필요한 텐서만 읽기)
│   │   │                   residency(가중치 상주 관리) · ocr(비전→언어 단계 전환) · jev_head · pointer(프로토콜)
│   │   │                   attention(공용 주의 연산) · kv_turbo(KV 캐시 FP8) · speculative(자기 추측 디코딩) · tokenizer · gguf_model · document(문서 묶음 읽기)
│   │   │                   zoro/(백본 · 글쓰기) · gocr/(문서 인식) · siglip2/(크롭: 타워, 잉크 분석, 패치 경쟁, 조각 계획)
│   │   ├── state_machine/  gate0_router · gate1_formal · gate2_consistency
│   │   │                   gate3_substantive · gate4_5_finalize · pipeline
│   │   ├── decision.rs     판단 질문(사실·선택지)과 모델 인터페이스
│   │   ├── drafting.rs     모델이 쓰는 초안 (민원 요약 · 소명 정규화 · 심사 의견)
│   │   ├── factcheck.rs    글의 숫자 · 날짜를 원문과 대조
│   │   ├── policy.rs       운영 모드 · 자동화 정책
│   │   ├── verdict.rs      처분 코드 · 판정
│   │   ├── case.rs         민원 입력 모델 (원문은 드롭 시 0 으로 덮음)
│   │   ├── graph.rs        관계망 질의
│   │   ├── store.rs        4대 테이블 (메모리 구현)
│   │   ├── audit.rs        감사 로그 (추가만 가능)
│   │   └── privacy.rs      솔트 해시 · 누출 검사
│   ├── examples/           zoro_probe · decide_probe · residency_check · weight_groups · pressure_probe
│   │                       qmatmul_check · softmax_check · jev_head_probe · gocr_probe · siglip2_probe
│   │                       read_page · read_eval · crop_eval · crop_view · layout_bench · siglip2_dump
│   │                       automation_eval · jev_format_search · write_eval
│   └── tests/              governance.rs (명세 사례 엔드투엔드) · real_models.rs (실제 모델 점검, --ignored)
├── tools/gguf-convert/     zorogravity-gguf: safetensors → GGUF (Q4_K · Q8_0, 비전 BF16), 정밀도 점검, jev_head 변환
├── tools/hwp-builder/      zorogravity-hwp: rhwp 로 내부결재용 · 민원인 교부용 HWP 생성
├── tools/model-settings/   zorogravity-models: 모델 폴더(AppData) · 다운로드 · 삭제
├── app/                    Tauri + React 앱: 키오스크 · 공무원 콘솔 창, 모델 설정 화면
├── run/                    백엔드별 실행 스크립트 (windows · linux · macos)
├── scripts/                ROCm · CUDA 빌드 환경, 실행 스크립트 공용 부분, 평가 · 학습 스크립트
└── docs/
    ├── automation-mode.md     자동화 모드 설계 · 법령 검토 항목 · 실제 모델 결과
    ├── weight-residency.md    가중치 상주 관리 · 단계 전환 · 실측
    ├── zoro-backbone.md       zoro 백본 구조 · 검증 · 한계
    ├── gpu-backends.md        GPU 백엔드 · 실측 · 알려진 문제
    ├── model-quantization.md  모델 양자화 · 정밀도 점검 · q8 선택 근거
    ├── document-crop.md       SigLIP2 크롭 · 패치 경쟁 · 묶음 읽기 · 실측
    └── jev-head.md            jev_head 분석 · 프로토콜 · 스케일 보정 · 형식 탐색
```

## 명세와 다른 점

- 명세의 Python `ai-engine` 과 Tauri `src-tauri` 대신, AI 와 규칙을 Rust `core` 에 모았습니다. Tauri 는 나중에 `core` 를 감싸는 얇은 층으로 붙입니다.
- 명세의 거짓 소명 행동 페널티(+8.2%)는 구현하지 않았습니다. 이유는 [docs/automation-mode.md](docs/automation-mode.md) 5절에 있습니다.
- 민원 분류는 접수 양식 코드가 규칙이고, 모델은 의견만 냅니다.
- 사건 간 관계망 누적과 HWP 바이너리 생성은 아직 없습니다.

## 알려진 문제

1. **`jev_head` 의 학습 형식은 재서 고른 것입니다.** 파일에 형식이 없어서 조합 1,680개를 재 봤고, 선택지 위치(닫는따옴표)와 스케일(1/32)만 근거가 뚜렷했습니다. 합성 · 명세 사례로만 검증했습니다 ([docs/jev-head.md](docs/jev-head.md) 6절).
2. **자동화 비율은 합성 사례의 값입니다.** 기준 0.95 에서 깨끗한 건 67 ~ 78% 가 자동 확정되고 함정은 통과하지 않았지만, 사례가 62건이고 모두 합성입니다. 모델은 관계 · 기간 · 용도의 안전망이 아니므로 그 판단은 규칙이 합니다 ([docs/automation-mode.md](docs/automation-mode.md) 9절). 자동화 모드는 기본으로 꺼져 있습니다.
3. **Vulkan 속도.** 독립 GPU 의 Vulkan 에는 선형 주의 커널이 없어 CPU 커널을 쓰고, 판단 1회가 1.3 s 입니다 (ROCm 은 0.62 s). 접두 캐시나 Vulkan 커널로 줄일 수 있습니다 (아직 안 함).
4. **GPU 메모리가 모자라면 포크가 조용히 호스트에 둡니다.** 같은 GPU 를 다른 프로그램이 쓰면 일부 층이 CPU 경로로 계산되어 판단 확률이 실행마다 달라졌습니다 (실측과 재현은 [docs/weight-residency.md](docs/weight-residency.md)). zorogravity 는 올리기 전후에 장치 여유 메모리를 검사해 오류로 거부합니다 (`DeviceMemoryShort`). 포크 자체는 고치지 않았습니다.
5. **Vulkan 의 메모리 배치.** 독립 GPU 에서는 Q8_0 가중치만 VRAM 에 올라가고 일반 텐서(비전 타워 포함)는 시스템 RAM 에 잡힙니다. `mem_info` 만 보면 비전을 올려도 변화가 없어 보이므로 `tensor_buffer_bytes` 로 같이 봅니다. [docs/weight-residency.md](docs/weight-residency.md).
6. **Vulkan 무한대 리터럴 (수정 완료).** 네이티브 Vulkan 커널을 켜면 softmax 와 reduce 셰이더의 `-1.0 / 0.0` 을 naga 가 거부해서 CPU 로 되돌아갑니다. 유한한 항등원으로 고쳐 sungu-che/candle-rocm-vulkan 의 `claude/vulkan-finite-shader-identities` 브랜치에 올렸습니다. 기본 설정에는 영향이 없습니다. [docs/gpu-backends.md](docs/gpu-backends.md) 4.1절.
7. **ROCm 커널 빌드.** 기본 MSVC 14.51 의 STL 이 clang 21 의 HIP 헤더와 충돌합니다. `scripts/rocm-env.bat` 이 14.44 툴셋을 씁니다.

## 라이선스

Apache-2.0 ([LICENSE](LICENSE)).
