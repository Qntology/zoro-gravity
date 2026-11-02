//! GovGraph-ZOROGRAVITY 코어.
//!
//! - `ai`: candle 기반 엔진 (장치 선택, 커널 자가 검증, Zoro-1 판단 헤드)
//! - `state_machine`: 5-Gate 민원 심사 파이프라인
//! - `policy`: 운영 모드와 자동화 정책 (기본은 보조 모드)
//! - `drafting`, `factcheck`: 모델이 쓰는 초안과 숫자 대조
//! - `verdict`, `case`, `graph`, `store`, `audit`, `privacy`: 도메인 모델과 저장·감사·비식별

pub mod ai;
pub mod audit;
pub mod case;
pub mod decision;
pub mod drafting;
pub mod evidence;
pub mod factcheck;
pub mod extract;
pub mod forms;
pub mod graph;
pub mod intake;
pub mod lattice;
pub mod policy;
pub mod privacy;
pub mod state_machine;
pub mod store;
pub mod verdict;
