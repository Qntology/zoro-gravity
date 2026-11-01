//! 운영 모드와 자동화 정책 (명세의 "Zoro-1 자동화 모드 활성화 여부").
//!
//! - 기본은 보조 모드(Assist)입니다. Zoro-1 은 근거와 의견을 보여 주고, 처분은 공무원이 확정합니다.
//! - 자동화 모드(Auto)는 승인자 사번과 법령 근거 조항을 적어야 켤 수 있고, 켜고 끈 기록은 감사 로그에 남습니다.
//! - 불리한 처분(반려·위반 판정)과 절차 처분(보완요구·진행 차단)은 어떤 모드에서도 자동 확정하지 않습니다.
//!   이 규칙에는 설정 스위치가 없습니다.
//! - 자동 확정은 (1) 규칙 검사가 모두 통과하고 (2) Zoro-1 확신도가 기준 이상일 때만 허용합니다.
//!   확정된 건의 일정 비율은 사후 샘플 감사 대상으로 표시합니다.

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::audit::{Actor, AuditCategory, AuditEvent};
use crate::verdict::{Judgement, OutcomeClass};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OperatingMode {
    #[default]
    Assist,
    Auto,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutomationApproval {
    pub approver_id: String,
    pub legal_basis: String,
    pub approved_at: DateTime<Utc>,
}

#[derive(Debug, Error, PartialEq)]
pub enum PolicyError {
    #[error("승인자 사번이 없으면 자동화 모드를 켤 수 없습니다")]
    MissingApprover,
    #[error("법령 근거 조항이 없으면 자동화 모드를 켤 수 없습니다")]
    MissingLegalBasis,
    #[error("신뢰도 기준은 0.5 이상 1.0 이하여야 합니다 (받은 값: {0})")]
    BadThreshold(f32),
    #[error("사후 샘플 비율은 0 이상 1 이하여야 합니다 (받은 값: {0})")]
    BadSampleRate(f32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Disposition {
    /// 자동 확정. `post_audit` 이면 사후 샘플 감사 대상입니다.
    AutoFinalize { post_audit: bool },
    /// 공무원이 확인해야 합니다. 사유를 함께 남깁니다.
    PendingOfficer { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutomationPolicy {
    mode: OperatingMode,
    approval: Option<AutomationApproval>,
    min_confidence: f32,
    post_audit_rate: f32,
}

impl Default for AutomationPolicy {
    fn default() -> Self {
        Self {
            mode: OperatingMode::Assist,
            approval: None,
            min_confidence: 0.95,
            post_audit_rate: 0.10,
        }
    }
}

impl AutomationPolicy {
    pub fn mode(&self) -> OperatingMode {
        self.mode
    }

    pub fn approval(&self) -> Option<&AutomationApproval> {
        self.approval.as_ref()
    }

    pub fn min_confidence(&self) -> f32 {
        self.min_confidence
    }

    pub fn post_audit_rate(&self) -> f32 {
        self.post_audit_rate
    }

    pub fn with_min_confidence(mut self, v: f32) -> Result<Self, PolicyError> {
        if !(0.5..=1.0).contains(&v) {
            return Err(PolicyError::BadThreshold(v));
        }
        self.min_confidence = v;
        Ok(self)
    }

    pub fn with_post_audit_rate(mut self, v: f32) -> Result<Self, PolicyError> {
        if !(0.0..=1.0).contains(&v) {
            return Err(PolicyError::BadSampleRate(v));
        }
        self.post_audit_rate = v;
        Ok(self)
    }

    /// 자동화 모드를 켭니다. 승인자 사번과 법령 근거가 모두 있어야 하며, 켜진 기록을 감사 이벤트로 돌려줍니다.
    pub fn enable_auto(
        &mut self,
        approver_id: &str,
        legal_basis: &str,
        at: DateTime<Utc>,
    ) -> Result<AuditEvent, PolicyError> {
        let approver_id = approver_id.trim();
        let legal_basis = legal_basis.trim();
        if approver_id.is_empty() {
            return Err(PolicyError::MissingApprover);
        }
        if legal_basis.is_empty() {
            return Err(PolicyError::MissingLegalBasis);
        }
        self.mode = OperatingMode::Auto;
        self.approval = Some(AutomationApproval {
            approver_id: approver_id.to_string(),
            legal_basis: legal_basis.to_string(),
            approved_at: at,
        });
        Ok(AuditEvent::new(
            at,
            Actor::OfficerConsole,
            AuditCategory::AutomationChanged,
            "-",
            Some(approver_id),
            format!("자동화 모드 ON · 법령 근거: {legal_basis}"),
        ))
    }

    /// 자동화 모드를 끕니다. 보조 모드로 돌아갑니다.
    pub fn disable_auto(&mut self, actor_id: &str, at: DateTime<Utc>) -> AuditEvent {
        self.mode = OperatingMode::Assist;
        self.approval = None;
        AuditEvent::new(
            at,
            Actor::OfficerConsole,
            AuditCategory::AutomationChanged,
            "-",
            Some(actor_id),
            "자동화 모드 OFF (보조 모드)",
        )
    }

    /// 판정을 자동 확정할 수 있는지 정합니다.
    /// `gates_clear` 는 Gate 1~4 의 규칙 검사가 모두 통과했는지입니다.
    pub fn disposition(&self, judgement: &Judgement, gates_clear: bool, case_id: &str) -> Disposition {
        if self.mode == OperatingMode::Assist {
            return pending("보조 모드: 공무원이 확정합니다");
        }
        match judgement.verdict.class() {
            OutcomeClass::Favorable => {}
            OutcomeClass::Adverse => return pending("불리한 처분은 자동 확정하지 않습니다"),
            OutcomeClass::Procedural => return pending("절차 처분(보완·진행 차단)은 공무원이 확인합니다"),
        }
        if !gates_clear {
            return pending("규칙 검사에 미확인 항목이 있습니다");
        }
        match judgement.confidence {
            None => pending("Zoro-1 확신도가 없습니다 (모델 미로드이거나 모델이 다른 답을 냈습니다)"),
            Some(p) if p < self.min_confidence => pending(format!(
                "Zoro-1 확신도 {p:.3} 이 기준 {:.2} 미만입니다",
                self.min_confidence
            )),
            Some(_) => Disposition::AutoFinalize {
                post_audit: sampled(case_id, self.post_audit_rate),
            },
        }
    }
}

fn pending(reason: impl Into<String>) -> Disposition {
    Disposition::PendingOfficer { reason: reason.into() }
}

/// 사건 ID 해시로 고르는 결정적 샘플링. 같은 사건은 언제나 같은 결과가 나옵니다.
pub fn sampled(case_id: &str, rate: f32) -> bool {
    if rate <= 0.0 {
        return false;
    }
    if rate >= 1.0 {
        return true;
    }
    let h = Sha256::digest(case_id.as_bytes());
    let x = u32::from_be_bytes([h[0], h[1], h[2], h[3]]) as f64 / u32::MAX as f64;
    x < rate as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verdict::Verdict;
    use chrono::TimeZone;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 10, 31, 9, 0, 0).unwrap()
    }

    fn favorable(conf: Option<f32>) -> Judgement {
        Judgement { verdict: Verdict::G3EligibleNormal, confidence: conf }
    }

    #[test]
    fn assist_mode_never_auto_finalizes() {
        let p = AutomationPolicy::default();
        assert_eq!(p.mode(), OperatingMode::Assist);
        assert!(matches!(
            p.disposition(&favorable(Some(0.999)), true, "case-1"),
            Disposition::PendingOfficer { .. }
        ));
    }

    #[test]
    fn enabling_auto_requires_approver_and_legal_basis() {
        let mut p = AutomationPolicy::default();
        assert_eq!(p.enable_auto("  ", "법령 X조", now()).unwrap_err(), PolicyError::MissingApprover);
        assert_eq!(p.enable_auto("o-17", " ", now()).unwrap_err(), PolicyError::MissingLegalBasis);
        assert_eq!(p.mode(), OperatingMode::Assist, "실패한 승인은 모드를 바꾸지 않습니다");

        let ev = p.enable_auto("o-17", "지침 제2조", now()).unwrap();
        assert_eq!(ev.category, AuditCategory::AutomationChanged);
        assert!(ev.detail.contains("지침 제2조"));
        assert_eq!(p.mode(), OperatingMode::Auto);
        assert_eq!(p.approval().map(|a| a.approver_id.as_str()), Some("o-17"));
    }

    #[test]
    fn adverse_is_never_auto_even_with_full_confidence() {
        let mut p = AutomationPolicy::default();
        p.enable_auto("o-17", "지침 제2조", now()).unwrap();
        let adverse = Judgement { verdict: Verdict::G3ViolationHighRisk, confidence: Some(1.0) };
        assert!(matches!(
            p.disposition(&adverse, true, "case-2"),
            Disposition::PendingOfficer { .. }
        ));
    }

    #[test]
    fn procedural_outcomes_stay_with_the_officer() {
        let mut p = AutomationPolicy::default();
        p.enable_auto("o-17", "지침 제2조", now()).unwrap();
        let j = Judgement { verdict: Verdict::G1InterceptForceHold, confidence: Some(0.99) };
        assert!(matches!(p.disposition(&j, true, "case-3"), Disposition::PendingOfficer { .. }));
    }

    #[test]
    fn favorable_needs_clear_gates_and_confidence() {
        let mut p = AutomationPolicy::default();
        p.enable_auto("o-17", "지침 제2조", now()).unwrap();
        assert!(matches!(p.disposition(&favorable(None), true, "c"), Disposition::PendingOfficer { .. }));
        assert!(matches!(p.disposition(&favorable(Some(0.90)), true, "c"), Disposition::PendingOfficer { .. }));
        assert!(matches!(p.disposition(&favorable(Some(0.99)), false, "c"), Disposition::PendingOfficer { .. }));
        assert!(matches!(
            p.disposition(&favorable(Some(0.99)), true, "c"),
            Disposition::AutoFinalize { .. }
        ));
    }

    #[test]
    fn disabling_returns_to_assist() {
        let mut p = AutomationPolicy::default();
        p.enable_auto("o-17", "지침 제2조", now()).unwrap();
        let ev = p.disable_auto("o-17", now());
        assert_eq!(p.mode(), OperatingMode::Assist);
        assert!(p.approval().is_none());
        assert!(ev.detail.contains("OFF"));
    }

    #[test]
    fn sampling_is_deterministic_and_roughly_at_rate() {
        assert!(!sampled("case-x", 0.0));
        assert!(sampled("case-x", 1.0));
        assert_eq!(sampled("case-x", 0.5), sampled("case-x", 0.5));
        let hits = (0..2000).filter(|i| sampled(&format!("case-{i}"), 0.10)).count();
        assert!((120..=280).contains(&hits), "hits {hits}");
    }

    #[test]
    fn rejects_out_of_range_settings() {
        assert!(AutomationPolicy::default().with_min_confidence(0.3).is_err());
        assert!(AutomationPolicy::default().with_post_audit_rate(1.5).is_err());
        assert!(AutomationPolicy::default().with_min_confidence(0.9).is_ok());
    }
}
