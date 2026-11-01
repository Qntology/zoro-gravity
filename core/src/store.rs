//! 4대 논리 테이블의 메모리 구현 (명세 Part 1 §4). 디스크 저장(SQLite 등)은 다음 단계입니다.
//!
//! - `tb_active_session`: 진행 중인 민원. 확정되면 원문을 0 으로 덮고 지웁니다.
//! - `tb_official_archive`: 확정 문서. 보존 기간 동안 남깁니다.
//! - `tb_audit_log`: 추가만 가능한 감사 로그.
//! - `tb_graph_entities`: 이번 버전은 사건 안에서만 쓰는 관계망입니다. 사건 간 누적은 법무 검토 후에 넣습니다.

use std::collections::HashMap;

use anyhow::{Result, bail};
use chrono::NaiveDate;

use crate::audit::{AuditEvent, AuditLog};
use crate::case::CaseFile;
use crate::drafting::GrievanceSummary;
use crate::verdict::Judgement;

#[derive(Debug, Clone, PartialEq)]
pub struct ArchiveRow {
    pub archive_id: String,
    pub case_id: String,
    /// 확정 처분 코드 (예: `G3_ELIGIBLE_NORMAL`, `SUPPLEMENT_7DAYS`)
    pub final_code: String,
    /// 내부 결재용 문서 (HWP 바이너리로 바꾸기 전 단계의 본문)
    pub officer_doc: String,
    /// 민원인 교부용 문서
    pub citizen_doc: String,
    /// 고위험 건을 공무원이 강제 승인했을 때 상급 감사실 보고 플래그
    pub special_audit_flag: bool,
    pub retention_until: NaiveDate,
}

/// 확정 전까지 세션이 들고 있는 것: 원문 민원과 처분 후보
pub struct ActiveSession {
    pub case: CaseFile,
    /// 규칙 판정 (처분 후보). 절차 정지 상태면 None
    pub judgement: Option<Judgement>,
    /// 공무원에게 보이는 정지·확인 사유
    pub reason: String,
    /// 고충민원 요약 초안 (공무원이 확인합니다)
    pub draft: Option<GrievanceSummary>,
}

#[derive(Default)]
pub struct Store {
    active: HashMap<String, ActiveSession>,
    archive: Vec<ArchiveRow>,
    audit: AuditLog,
}

impl Store {
    pub fn open(&mut self, session: ActiveSession) -> Result<()> {
        let id = session.case.case_id.clone();
        if self.active.contains_key(&id) {
            bail!("이미 진행 중인 민원입니다: {id}");
        }
        self.active.insert(id, session);
        Ok(())
    }

    pub fn active(&self, case_id: &str) -> Option<&ActiveSession> {
        self.active.get(case_id)
    }

    pub fn active_mut(&mut self, case_id: &str) -> Option<&mut ActiveSession> {
        self.active.get_mut(case_id)
    }

    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    pub fn record(&mut self, event: AuditEvent) {
        self.audit.append(event);
    }

    pub fn audit(&self) -> &AuditLog {
        &self.audit
    }

    pub fn archive(&self) -> &[ArchiveRow] {
        &self.archive
    }

    /// Gate 5: 아카이브 적재 → 감사 기록 → 활성 세션 삭제를 한 번에 처리합니다.
    /// 실패할 수 있는 검사는 모두 먼저 하고, 상태는 마지막에만 바꿉니다.
    pub fn finalize(&mut self, case_id: &str, row: ArchiveRow, events: Vec<AuditEvent>) -> Result<()> {
        if !self.active.contains_key(case_id) {
            bail!("활성 세션이 없습니다: {case_id}");
        }
        if self.archive.iter().any(|r| r.archive_id == row.archive_id) {
            bail!("중복 문서번호입니다: {}", row.archive_id);
        }
        self.archive.push(row);
        for e in events {
            self.audit.append(e);
        }
        // 세션을 꺼내 바로 버립니다. CaseFile 의 Drop 이 원문을 0 으로 덮습니다.
        drop(self.active.remove(case_id));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::{Actor, AuditCategory};
    use crate::case::{Category, Person};
    use crate::verdict::Verdict;
    use chrono::Utc;

    fn case(id: &str) -> CaseFile {
        CaseFile {
            case_id: id.into(),
            category: Category::Welfare,
            docs: vec![],
            applicant: Person { name: "김철수".into(), birth_prefix: "990101".into(), identifier: "9901011234567".into() },
            landlord: None,
            lease: None,
            bank: None,
            signature: None,
            poa_written_on: None,
            today: NaiveDate::from_ymd_opt(2026, 10, 31).unwrap(),
            excuse: None,
            relations: vec![],
            supplement_count: 0,
            grievance_text: None,
        }
    }

    fn session(id: &str) -> ActiveSession {
        ActiveSession {
            case: case(id),
            judgement: Some(Judgement::by_rule(Verdict::G3EligibleNormal)),
            reason: String::new(),
            draft: None,
        }
    }

    fn row(archive_id: &str, case_id: &str) -> ArchiveRow {
        ArchiveRow {
            archive_id: archive_id.into(),
            case_id: case_id.into(),
            final_code: "G3_ELIGIBLE_NORMAL".into(),
            officer_doc: "내부".into(),
            citizen_doc: "민원인".into(),
            special_audit_flag: false,
            retention_until: NaiveDate::from_ymd_opt(2031, 10, 31).unwrap(),
        }
    }

    #[test]
    fn finalize_archives_logs_and_drops_the_session() -> Result<()> {
        let mut s = Store::default();
        s.open(session("c1"))?;
        assert!(s.open(session("c1")).is_err(), "같은 사건을 두 번 열 수 없습니다");
        let ev = AuditEvent::new(Utc::now(), Actor::System, AuditCategory::AutoFinalized, "c1", None, "ok");
        s.finalize("c1", row("역삼-2026-0001호", "c1"), vec![ev])?;
        assert_eq!(s.active_count(), 0);
        assert_eq!(s.archive().len(), 1);
        assert_eq!(s.audit().len(), 1);
        Ok(())
    }

    #[test]
    fn finalize_is_all_or_nothing() -> Result<()> {
        let mut s = Store::default();
        s.open(session("c1"))?;
        s.finalize("c1", row("A-1", "c1"), vec![])?;
        s.open(session("c2"))?;
        assert!(s.finalize("c2", row("A-1", "c2"), vec![]).is_err(), "중복 문서번호는 실패해야 합니다");
        assert_eq!(s.active_count(), 1, "실패하면 세션이 그대로 남아야 합니다");
        assert_eq!(s.archive().len(), 1);
        assert!(s.finalize("missing", row("A-2", "missing"), vec![]).is_err());
        Ok(())
    }
}
