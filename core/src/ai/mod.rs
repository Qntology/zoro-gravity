//! candle 기반 AI 엔진.
//!
//! - `accel`: 장치 선택(ROCm → Vulkan → CPU) 과 dtype 정책
//! - `verify`: GPU 커널 자가 검증 (CPU 기준값과 비교해 틀리면 그 경로만 끔)
//! - `jev_head`: Zoro-1 판단 헤드 (질의·키 투영, 0-Generation 포인터 판단)
//! - `pointer`: 판단 질문을 프롬프트로 만들고 백본 은닉 상태를 `jev_head` 로 점수화
//! - `gguf_model`: 선택된 양자화 모델(`active.json` → GGUF) 열기
//! - `weights` · `residency`: GGUF 에서 필요한 텐서만 읽기, 모델 하나 · 단계별 묶음만 올리기
//! - `attention` · `tokenizer`: 모델들이 함께 쓰는 주의 연산과 토크나이저
//! - `kv_turbo`: KV 캐시 FP8 압축 (장치 메모리가 모자랄 때)
//! - `speculative`: 자기 추측 디코딩 (같은 가중치의 블록 초안 + 자기회귀 검증)
//! - `zoro`: 판단 백본, `gocr`: 문서 인식, `siglip2`: 읽을 곳 고르기
//! - `ocr`: 비전 → 언어 단계 전환, `document`: 문서 한 장 읽기

pub mod accel;
pub mod attention;
pub mod document;
pub mod gguf_model;
pub mod gocr;
pub mod jev_head;
pub mod kv_turbo;
pub mod ocr;
pub mod pointer;
pub mod residency;
pub mod runtime;
pub mod siglip2;
pub mod speculative;
pub mod tokenizer;
pub mod trace;
pub mod verify;
pub mod weights;
pub mod zoro;

/// 모델 폴더 위치 (`%LOCALAPPDATA%\zorogravity\models` 등). `zorogravity-models` 크레이트입니다.
pub use zorogravity_models as model_store;
