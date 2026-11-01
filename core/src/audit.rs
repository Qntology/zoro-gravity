//! 감사 로그 (명세 Table 4 `tb_audit_log`). 추가만 가능하고, 고치거나 지우는 API 가 없습니다.
//!
//! 기록에는 주민등록번호·실명 원문을 넣지 않습니다. 사건 ID 와 해시, 판정 코드만 씁니다.

use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Actor {
    CitizenKiosk,
    OfficerConsole,
    ZoroGuardrail,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditCategory {
    /// 태블릿 전자서명으로 원본 서명 누락을 보완함
    SignSupplemented,
    /// 민원인의 모순된 소명을 기각함
    ContradictionBlocked,
    /// 고위험 건에 대해 제3자 실명 열람을 허용함 (사유 기록)
    PrivacyUnmaskGranted,
    /// 저위험 건에 대한 제3자 실명 열람 요청을 거부함
    PrivacyUnmaskDenied,
    /// 공무원의 위법 가능 행위(강제 진입·무증빙 수정·무지성 승인 등)를 차단함
    IllegalActionIntercepted,
    /// 확정 수치를 원본 값으로 되돌림 (계좌번호·차액 등)
    FactLocked,
    /// 민원인 교부용 문서에서 내부 기밀 용어를 차단함
    LeakBlocked,
    /// 처분 후보가 정해짐
    DecisionRecorded,
    /// 자동화 모드를 켜거나 끔
    AutomationChanged,
    /// 자동 확정됨
    AutoFinalized,
    /// 공무원이 처분을 확정함
    OfficerConfirmed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditEvent {
    pub at: DateTime<Utc>,
    pub actor: Actor,
    pub category: AuditCategory,
    pub case_id: String,
    pub officer_id: Option<String>,
    pub detail: String,
}

impl AuditEvent {
    pub fn new(
        at: DateTime<Utc>,
        actor: Actor,
        category: AuditCategory,
        case_id: &str,
        officer_id: Option<&str>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            at,
            actor,
            category,
            case_id: case_id.to_string(),
            officer_id: officer_id.map(str::to_string),
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Default)]
pub struct AuditLog {
    events: Vec<AuditEvent>,
}

impl AuditLog {
    pub fn append(&mut self, event: AuditEvent) {
        self.events.push(event);
    }

    pub fn events(&self) -> &[AuditEvent] {
        &self.events
    }

    pub fn len(&self) -> usize {
        self.events.len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_only_grows() {
        let mut log = AuditLog::default();
        assert!(log.is_empty());
        let now = Utc::now();
        log.append(AuditEvent::new(now, Actor::System, AuditCategory::DecisionRecorded, "c1", None, "x"));
        log.append(AuditEvent::new(now, Actor::OfficerConsole, AuditCategory::OfficerConfirmed, "c1", Some("o1"), "y"));
        assert_eq!(log.len(), 2);
        assert_eq!(log.events()[1].officer_id.as_deref(), Some("o1"));
    }
}
