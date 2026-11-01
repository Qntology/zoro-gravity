# 모델 양자화 (GGUF Q4 · Q8, 비전 BF16)

`models/` 폴더의 두 모델을 GGUF 로 바꿉니다. 변환과 점검은 `tools/gguf-convert` 의 `zorogravity-gguf` 가 합니다. 모델 파일은 저장소에 올리지 않습니다 (`.gitignore` 의 `models/`).
zorogravity 는 `gocr` · `zoro` · `siglip2` 의 가중치를 씁니다. 다른 모델(Qwen3.5 등)을 쓰지 않고, logis-center 의 모델 코드도 가져오지 않았습니다. logis-center 는 장치 선택, 커널 자가 검증, 양자화 GPU 실행 같은 **최적화 방식**만 참고했습니다.

## 대상

| 폴더 | 모델 | 구조 | 원본 |
| --- | --- | --- | --- |
| `models/gocr` | gocr (GravityOCR) | 텍스트 16층 (hidden 1536, 주의 헤드 16 / KV 8) + 비전 24층, 블록 디퓨전 | safetensors BF16 1개 파일 (2.2 GB) |
| `models/zoro` | zoro (Zoro-1) | 텍스트 32층 (선형 주의 24 + 전체 주의 8, hidden 2560) + 비전 24층, `jev_head.pt` | safetensors BF16 17 샤드 (8.5 GB) |

각 `config.json` 의 `model_type` 은 가중치가 따르는 구조 이름으로 `glm_ocr`, `qwen3_5` 라고 적혀 있습니다. 변환기는 그 값을 보고 텐서 이름을 매핑할 뿐이고, GGUF 의 구조 이름은 모델 이름(`gocr`, `zoro`)으로 붙입니다. 원본이 선언한 이름은 출처로만 `zorogravity.source_architecture` 에 남습니다.

## 변환 규칙

- **선형 가중치**: q4 는 Q4_K (열 수가 256 의 배수일 때), q8 은 Q8_0 (열 수가 32 의 배수일 때). 행이 64 보다 적은 행렬은 F32.
- **비전 타워** (`model.visual.*`): BF16 그대로. 두 버전 모두 같습니다. 텍스트 GGUF 에 넣지 않고 `mmproj-*-bf16.gguf` 로 따로 씁니다 (아래 절).
- **정규화·게이트·컨볼루션**: F32. 선형 주의의 `A_log`, `dt_bias`, `conv1d`, `in_proj_a/b` 를 포함합니다.
- **텐서 이름**: llama.cpp 의 블록 이름 관례를 씁니다 (`blk.N.attn_q.weight`, `token_embd.weight`, `ssm_a` 등).
- **차원 4개 제한**: GGUF 는 차원이 4개까지입니다. 5차원 비전 패치 임베딩 (1024×3×2×14×14) 은 시간축과 채널을 합쳐 1024×6×14×14 로 저장하고, 원래 모양은 `zorogravity.reshaped_tensors` 에 적습니다. 메모리 배치는 그대로입니다.
- **메타데이터**: 구조 값 (`zoro.*` 또는 `gocr.*`) 과 원본 config.json 전문 (`zorogravity.source_config_json`) 을 넣습니다. `block_diffusion.json` 이 있으면 `zorogravity.block_diffusion.size` · `mask_id` 로 옮깁니다. 초안은 언어 가중치만 쓰므로 텍스트 GGUF 에 넣고, `verify` 가 마스크 토큰이 임베딩 표 안에 있는지 확인합니다.
- **선형 주의의 V 헤드 순서**: 원본 순서를 그대로 둡니다 (포크의 `GdnHeadOrder::Grouped`). 재배열(`Tiled`)하지 않았습니다.
- **GGUF 밖**: `tokenizer.json` 은 그대로 둡니다. `jev_head` 는 F32 safetensors 로 따로 둡니다.
- **연결 가중치**: zoro 는 `tie_word_embeddings` 이므로 `output.weight` 를 따로 두지 않습니다 (`token_embd` 를 공유합니다). gocr 는 `lm_head` 를 `output.weight` 로 둡니다.

## 정밀도 점검 (프록시)

양자화된 선형층마다 두 값을 잽니다.

- **가중치 상대오차** ‖Ŵ − W‖ / ‖W‖
- **출력 상대오차** ‖X Ŵᵀ − X Wᵀ‖ / ‖X Wᵀ‖. X 는 고정 시드의 가우시안 입력 4행입니다.

기준은 **출력 상대오차의 평균 5% 이하, 최대 10% 이하**입니다. 이 값은 측정하기 전에 정했습니다. 통과하면 q4, 아니면 q8 을 씁니다. `zorogravity-gguf auto` 가 q4 를 먼저 만들고 점검한 뒤, 실패하면 q8 로 전환하고 `active.json` 에 기록합니다.

**한계**: 이 점검은 디코딩 품질을 직접 재지 않습니다. 로짓, OCR 글자 정확도, 판단 확률은 아직 못 봅니다. 모델 순전파가 생기면 그 결과로 기준을 다시 정해야 합니다. 토큰 임베딩은 행 조회라서 출력 오차를 재지 않고 가중치 오차만 봅니다.

## 결과

| 모델 | 버전 | 파일 크기 | 출력 상대오차 평균 / 최대 | 판정 |
| --- | --- | --- | --- | --- |
| gocr | q4 (Q4_K) | 1.25 GB | 7.48% / 8.05% | ❌ 평균이 5% 를 넘음 |
| gocr | q8 (Q8_0) | 1.58 GB | 0.54% / 0.57% | ✅ 통과 |
| zoro | q4 (Q4_K) | 2.38 GB | 7.524% / 8.159% | ❌ 평균이 5% 를 넘음 |
| zoro | q8 (Q8_0) | 4.48 GB | 0.545% / 0.604% | ✅ 통과 |
| zoro | mmproj (BF16) | 0.67 GB | 비전 297개 모두 원본과 값이 같음 | ✅ 통과 |
| siglip2 | q8 (Q8_0, 비전 BF16) | 1.73 GB | 0.544% / 0.590% (텍스트 타워) | ✅ 통과 |

siglip2 는 hidden 1152 가 Q4_K 블록(256)의 배수가 아니라서 q4 를 만들지 않습니다. 비전 타워는 BF16 이라 양자화 오차가 없고, 텍스트 타워는 앵커 문구를 새로 인코딩할 때만 씁니다.

zoro 의 q4 · q8 크기는 비전 타워를 뺀 값입니다. 빼기 전에는 3.05 GB · 5.15 GB 였습니다.

**선택: 두 모델 모두 q8.** 기록은 `models/<모델>/active.json` 에 있고, 엔진은 `ai::gguf_model::load_active` 로 읽습니다.

두 모델의 q4 오차는 층마다 7.4–8.2% 로 고르게 나옵니다. 특정 층의 매핑 오류였다면 한두 층만 크게 튀었을 것이고, 모든 텐서 이름과 모양이 원본과 맞았습니다. 그래서 이 결과는 4비트 양자화의 일반적인 오차로 봅니다.

## q8 을 쓰는 이유

1. **정밀도**: q4 는 위 기준을 넘습니다. 숫자를 정확히 읽어야 하는 OCR (계좌번호 등) 과 판단 헤드는 오차에 민감합니다.
2. **GPU 에서 실제로 도는 형식**: 선택된 GGUF 의 실제 가중치로 RX 6600 에서 재 봤습니다 (`core/examples/qmatmul_check.rs`, zoro `blk.0.ffn_down.weight` [2560×9216]).

| 백엔드 | 형식 | 1행 (디코딩) 장치 / CPU | 64행 (프리필) 장치 / CPU |
| --- | --- | --- | --- |
| Vulkan | **Q8_0** | 0.160 ms / 3.58 ms (22배) | 2.15 ms / 152 ms (71배) |
| Vulkan | Q4_K | 2.19 ms / 2.47 ms (**GPU 이득 없음**) | 144 ms / 198 ms |
| ROCm | Q8_0 | 0.180 ms / 2.45 ms (14배) | 2.18 ms / 152 ms (70배) |
| ROCm | Q4_K | 0.312 ms / 3.34 ms (11배) | 2.33 ms / 141 ms (61배) |

   - Vulkan 은 Q8_0 가중치만 VRAM 에 올려 GPU 커널로 곱합니다. Q4_K 는 CPU 에서 계산됩니다 (오차가 정확히 0 이고 시간이 CPU 와 같음). ROCm 은 둘 다 GPU 에서 돕니다.
   - Q8_0 의 GPU 결과는 가중치를 F32 로 푼 정확한 곱과 2.5e-7 ~ 1.8e-6 안에서 일치합니다. CPU 결과가 정확한 곱에서 0.5% 벗어나는데, CPU 경로가 활성값도 양자화하기 때문입니다. 그래서 GPU↔CPU 오차 0.5% 는 GPU 의 오차가 아닙니다.
3. **VRAM**: 파일 크기가 커집니다. zoro q8 4.48 GB, gocr q8 0.72 GB 입니다 (비전 mmproj 는 각각 0.67 GB · 0.87 GB 로 따로). RX 6600 (8 GB) 에 둘을 한꺼번에 올리면 빠듯해서, **모델은 한 번에 하나, 단계마다 필요한 묶음만** 올립니다 ([weight-residency.md](weight-residency.md)). zoro 판단은 텍스트 블록 3.6 GB 만 올립니다 (비전 · 임베딩 표 · 출력 투영은 올리지 않음).

q4 로 바꿀 수 있는 조건은 두 가지입니다. 디코딩 품질 평가로 q4 가 충분하다고 확인될 것, 그리고 Vulkan 에서 Q4_K 의 GPU 커널이 생기거나 ROCm 을 쓸 것. 바꿀 때는 `active.json` 의 `active` 와 `file` 을 고치면 됩니다. 기준을 바꾸려면 `tools/gguf-convert/src/main.rs` 의 `ACT_MEAN_LIMIT` 를 고치세요.

## 비전 타워 분리 (mmproj)

`zorogravity-gguf auto` 는 세 모델 모두 비전 타워를 텍스트 GGUF 에 넣지 않고 따로 씁니다.

| 파일 | 내용 | 크기 | 분리 전 |
| --- | --- | --- | --- |
| `zoro-q4.gguf` | 텍스트 426 텐서 | 2.38 GB | 3.05 GB |
| `zoro-q8.gguf` | 텍스트 426 텐서 | 4.48 GB | 5.15 GB |
| `mmproj-zoro-bf16.gguf` | 비전 타워 297 텐서, BF16 | 0.67 GB | - |
| `gocr-q4.gguf` | 텍스트 163 텐서 | 0.38 GB | 1.25 GB |
| `gocr-q8.gguf` | 텍스트 163 텐서 | 0.72 GB | 1.58 GB |
| `mmproj-gocr-bf16.gguf` | 비전 타워 347 텐서, BF16 | 0.87 GB | - |
| `siglip2-q8.gguf` | 텍스트 타워 440 텐서 | 0.88 GB | 1.73 GB |
| `mmproj-siglip2-bf16.gguf` | 비전 타워 448 텐서, BF16 | 0.86 GB | - |

- 비전 타워는 q4 · q8 에서 같아서 파일 하나를 함께 씁니다.
- `active.json` 의 `mmproj` 가 이 파일을 가리킵니다. 엔진은 매핑만 하고 비전 단계 전에는 읽지 않습니다. zoro 는 판단과 글쓰기에 이미지를 쓰지 않으므로 지금은 이 파일의 텐서를 읽는 일이 없습니다.
- 파일이 없어도 판단과 글쓰기는 돕니다. 비전이 들어 있는 예전 GGUF 도 그대로 읽습니다.
- `verify` 는 mmproj 를 함께 열어 텐서 수를 맞춰 보고, 비전 텐서가 원본과 값이 같은지 텐서마다 비교합니다. siglip2 는 원본이 F32 라서 BF16 으로 반올림한 값과 비교합니다.
- 텐서 이름은 `model.visual.*` 를 유지합니다. 이름은 mmproj 이지만 llama.cpp 의 `clip` 형식(`v.blk.N.*`)은 아니어서 llama.cpp 로는 읽지 못합니다. 메타데이터의 `general.type` 이 `mmproj` 입니다.
- 비전 타워는 저장 형식 그대로 매핑에서 장치로 바로 올립니다 ([weight-residency.md](weight-residency.md) 전환 비용).
- Windows 는 매핑 중인 파일을 덮어쓰지 못합니다. 엔진을 끈 뒤에 변환하세요.

## 재현

```bat
cargo build -p zorogravity-gguf
target\debug\zorogravity-gguf.exe auto models\gocr
target\debug\zorogravity-gguf.exe auto models\zoro
target\debug\zorogravity-gguf.exe auto models\siglip2
target\debug\zorogravity-gguf.exe head models\zoro\jev_head.pt models\zoro\jev_head.safetensors
```

만들어지는 파일 (모두 `models/` 안):

- `gocr/gocr-q4.gguf`, `gocr-q8.gguf`, `mmproj-gocr-bf16.gguf`, `precision-q4.json`, `precision-q8.json`, `active.json`
- `zoro/zoro-q4.gguf`, `zoro-q8.gguf`, `mmproj-zoro-bf16.gguf`, `precision-q4.json`, `precision-q8.json`, `active.json`
- `zoro/jev_head.safetensors` (F32)
- `siglip2/siglip2-q8.gguf`, `mmproj-siglip2-bf16.gguf`, `precision-q8.json`, `active.json`, `crop_probe.bin` (시각 앵커, [document-crop.md](document-crop.md))

`jev_head` 의 구조와 반영 내용은 [jev-head.md](jev-head.md) 에 있습니다.
