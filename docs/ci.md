# CI (cross-verify)

`.github/workflows/cross-verify.yml` 이 Linux · macOS · Windows 에서 같은 코드를 검증합니다. 구조는 logis-center 의 `cross-verify` 와 같습니다. 푸시마다, 그리고 포크에서 `main` 으로 오는 PR 마다 돕니다.

## 작업별 범위

호스팅 러너에는 AMD · NVIDIA GPU 가 없어서 실제로 돌릴 수 있는 것만 돌립니다. 건너뛴 단계는 통과가 아닙니다. 작업마다 마지막에 무엇을 확인했는지 `scope` 알림으로 남깁니다.

| 작업 | 러너 | 확인하는 것 | 돌리지 않는 것 |
| --- | --- | --- | --- |
| `mac-cpu` | macos-15 | 빌드, 시험 전체, 앱 빌드 | |
| `mac-metal` | macos-15 | 위와 같음 + **실제 Metal 장치**에서 장치 시험 | 실제 모델 |
| `mac-vulkan` | macos-15 | 위와 같음 + MoltenVK 에서 장치 시험 | 실제 모델 |
| `linux-cpu` | ubuntu-latest | 빌드, 시험 전체, 앱 빌드 | |
| `linux-vulkan` | ubuntu-latest | 위와 같음 + lavapipe(CPU 위의 소프트웨어 Vulkan)에서 장치 시험 | 실제 GPU |
| `windows-cpu` | windows-latest | 빌드, 시험 전체, 앱 빌드 | |
| `windows-vulkan` | windows-latest | 빌드, 앱 빌드 | 시험 (소프트웨어 Vulkan 드라이버가 없음) |
| `linux-cuda` · `windows-cuda` | ubuntu · windows | 타입 검사 + 실제 `nvcc` 커널 빌드 | 링크, 모든 시험, GPU 위의 커널 |
| `linux-rocm` · `windows-rocm` | ubuntu · windows | 타입 검사 | 링크, 모든 시험 (`hipcc` 없음, 커널 이미지는 비어 있음) |
| `frontend` | ubuntu-latest | `npm ci`, `tsc`, `vite build` | |

앱은 시험이 없어서 링크까지만 확인합니다. CI 의 앱 빌드는 빈 `dist` 를 넣으므로 프론트엔드는 `frontend` 작업이 따로 봅니다.

## 시험

- **시험 전체**: `cargo test --workspace` 에서 앱을 뺀 것입니다. `--ignored` 시험은 돌리지 않습니다. 실제 모델 파일(`real_models`)은 CI 에 없습니다.
- **장치 시험**: 이름이 `device_` 로 시작하는 `#[ignore]` 시험 6개입니다. 합성 모델을 그 장치에서 돌려 스칼라 기준 구현이나 CPU 결과와 비교합니다.

| 시험 | 보는 것 |
| --- | --- |
| `device_text_matches_the_scalar_reference` | zoro 순전파, 이어 읽기, 글쓰기, q8 가중치 |
| `device_vision_matches_the_cpu` | gocr 비전 인코더 |
| `device_decoder_matches_the_cpu` | gocr 언어 디코더와 KV 캐시 |
| `device_towers_match_the_cpu` | siglip2 비전 · 텍스트 타워 |
| `device_fp8_matches_the_cpu_codec` | KV FP8 압축 · 복원 · 압축 행 어텐션 |
| `device_loading_is_measured_and_releasing_returns_the_memory` | 올리고 내릴 때의 장치 메모리 |

직접 돌리려면 장치를 고르고 `--ignored` 로 실행합니다. GPU 가 열리지 않으면 건너뛰지 않고 실패합니다.

```sh
ZOROGRAVITY_DEVICE=metal cargo test -p zorogravity-core --lib --no-default-features --features metal -- device_ --ignored --nocapture --test-threads 1
```

시험은 끝날 때 `[DEVICE-CHECK]` 줄을 남기고, CI 는 그 줄의 수를 셉니다. 장치 시험을 더하면 워크플로의 `DEVICE_CHECKS` 도 올리세요.

## 대상 고르기

바뀐 것만 다시 돌릴 때 머리 커밋 메시지에 한 줄로 씁니다. `workflow_dispatch` 의 `targets` 도 같은 목록을 받습니다.

```text
[targets: mac-metal,linux-vulkan]
```

`[skip ci]` 는 건너뜁니다. 브랜치에 새로 푸시하면 진행 중인 실행은 취소되고, `main` 은 커밋마다 끝까지 돕니다.

## rhwp

`tools/hwp-builder` 가 쓰는 rhwp 는 저장소가 git 3.6 GB, 체크아웃 9 GB 입니다. cargo 는 워크스페이스를 해석할 때 git 의존성을 받으므로 `-p` 로 골라도 피할 수 없습니다.

그래서 `.github/scripts/use-rhwp-path.sh` 가 `tools/hwp-builder/Cargo.toml` 이 고정한 커밋에서 빌드에 필요한 경로만(139 MB) 받아 작업 안에서만 path 의존성으로 바꿉니다. `Cargo.lock` 의 rhwp 출처 줄도 지우므로 나머지 의존성은 `--locked` 로 그대로 검사합니다. rhwp 를 올린 뒤 빌드가 파일을 찾지 못하면 스크립트의 경로 목록에 더하세요.

## 도구와 캐시

- Rust 는 워크플로의 `RUST_VERSION` 으로 고정합니다. 모든 OS 가 같은 컴파일러를 씁니다.
- 빌드 캐시는 작업이 실패해도 저장합니다. 계속 실패하는 작업이 매번 처음부터 빌드하지 않게 하기 위해서입니다.
- 포크는 `Cargo.lock` 이 고정한 커밋을 받습니다. 저장소 옆에 `candle-rocm-vulkan` 이 없어도 됩니다.

## 결과 읽기

작업 로그는 인증 없이 받을 수 없지만 어노테이션은 공개 저장소에서 인증 없이 읽을 수 있습니다. 그래서 필요한 것을 어노테이션에 남깁니다.

- `log`: 실패한 단계의 오류와 로그 끝부분
- `suite result`: 시험 수
- `device (...)`: 장치 시험 결과와 커널 자가 검증 결과
- `scope ...`: 그 작업이 확인한 범위
- `env ...`: 러너, 컴파일러, SDK, 빌드 입력

`The job was not acquired by Runner of type hosted` 는 코드와 무관한 러너 배정 실패입니다. 다시 돌리면 됩니다.
