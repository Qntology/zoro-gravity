//! 5-Gate 상태 머신. 게이트마다 순수 함수로 결과를 내고, 파이프라인이 순서와 정지 조건을 맡습니다.
//!
//! - `gate0_router`: 5대 민원 분류와 필수 서류 스키마
//! - `gate1_formal`: 형식심사 (서류, 전자서명, 위임장 유효기간)
//! - `gate2_consistency`: 서류 간 정합성 (계좌 원장 우위, 차액, 거짓 소명 차단, 증빙 없는 수정 차단)
//! - `gate3_substantive`: 관계망 실질 심사 (동명이인, 가까운 연결, 고위험 승인 통제, 제3자 열람 기준)
//! - `gate4_5_finalize`: 양식 선택, 민원인 문서 누출 차단, 확정 수치 복원, 보존 행
//! - `pipeline`: 위 게이트를 순서대로 실행하고 확정 경로를 정하는 `Engine`

pub mod gate0_router;
pub mod gate1_formal;
pub mod gate2_consistency;
pub mod gate3_substantive;
pub mod gate4_5_finalize;
pub mod pipeline;

pub use pipeline::{CaseOutcome, Engine, OfficerDecision};
