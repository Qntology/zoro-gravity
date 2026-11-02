//! 파이프라인: Gate 0 → 5 를 차례로 돌리고, 멈춰야 하는 곳에서 멈춥니다.
//!
//! 확정 경로는 두 가지뿐입니다.
//! 1. 자동화 정책이 허용한 자동 확정 (Auto 모드 + 유리한 처분 + 규칙 통과 + 확신도 기준 이상)
//! 2. 공무원의 명시적 결정 (`confirm`)
//!
//! 키오스크 재시도(서명 등)는 처분이 아닙니다. 공무원 확인 대기는 세션으로 남습니다.

use anyhow::{Result, bail};
use chrono::{DateTime, NaiveDate, Utc};

use super::gate0_router;
use super::gate1_formal::{self, FormalReport, Gate1};
use super::gate2_consistency::{self, Consistency, ReceiptReview};
use super::gate3_substantive::{self, Justification, OfficerOutcome, Substantive};
use super::gate4_5_finalize::{self as g45, Form};
use crate::audit::{Actor, AuditCategory, AuditEvent};
use crate::case::{CaseFile, CaseSource, Category, Excuse, Identity, LeaseFacts};
use crate::decision::DecisionModel;
use crate::drafting::{self, GrievanceSummary, TextWriter};
use crate::policy::{AutomationPolicy, Disposition};
use crate::privacy;
use crate::store::{ActiveSession, Store};
use crate::verdict::{Judgement, OutcomeClass, Verdict};

/// 공무원이 보류된 건에 내리는 결정
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OfficerDecision {
    /// 후보 처분을 그대로 확정합니다.
    Confirm,
    /// 반려로 확정합니다.
    Reject,
    /// 고위험 판정을 정상 승인으로 바꿉니다. 입증이 있어야 합니다.
    ApproveHighRisk(Justification),
    /// 누락 서류에 대해 보완요구(7일)를 발급합니다.
    IssueSupplement,
    /// 누락 서류를 둔 채 Gate 2 로 강제 진입합니다. 가드레일이 막습니다.
    ForcePassToGate2,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CaseOutcome {
    /// 키오스크에서 다시 받습니다. 처분이 아닙니다. (민원은 세션에 남지 않습니다)
    KioskRetry { verdict: Verdict, reason: String },
    /// 공무원 확인을 기다립니다.
    OfficerReview { judgement: Option<Judgement>, reason: String },
    /// 자동화 정책으로 자동 확정되었습니다 (Gate 4·5 완료).
    AutoFinalized { archive_id: String, post_audit: bool },
    /// 공무원 결정으로 확정되었습니다 (Gate 4·5 완료).
    Finalized { archive_id: String },
    /// 가드레일이 행위를 막았습니다. 세션은 그대로 남습니다.
    Blocked { verdict: Verdict, reason: String },
}

/// 확정에 필요한 값을 한데 모읍니다.
struct Closing {
    form: Form,
    final_code: String,
    notice: Option<Verdict>,
    actor: Actor,
    officer_id: Option<String>,
    special_audit_flag: bool,
    post_audit: bool,
    confidence: Option<f32>,
}

pub struct Engine {
    pub policy: AutomationPolicy,
    /// Zoro-1 판단 모델 (백본 + jev_head). 없으면 규칙만으로 동작하고, 자동 확정은 불가능합니다.
    pub model: Option<Box<dyn DecisionModel>>,
    /// 초안을 쓰는 모델. 없으면 초안 없이 규칙과 템플릿만 씁니다. 초안은 처분을 바꾸지 않습니다.
    pub writer: Option<Box<dyn TextWriter>>,
    /// 문서번호 앞부분 (기관 코드)
    pub office_code: String,
    salt: Vec<u8>,
    pub store: Store,
}

impl Engine {
    pub fn new(salt: &[u8], policy: AutomationPolicy, model: Option<Box<dyn DecisionModel>>) -> Self {
        Self {
            policy,
            model,
            writer: None,
            office_code: "ZOROGRAVITY".into(),
            salt: salt.to_vec(),
            store: Store::default(),
        }
    }

    pub fn with_writer(mut self, writer: Box<dyn TextWriter>) -> Self {
        self.writer = Some(writer);
        self
    }

    /// 자동화 모드를 켭니다. 승인자와 법령 근거는 감사 로그에 남습니다.
    pub fn enable_automation(&mut self, approver_id: &str, legal_basis: &str, now: DateTime<Utc>) -> Result<()> {
        let event = self.policy.enable_auto(approver_id, legal_basis, now)?;
        self.store.record(event);
        Ok(())
    }

    pub fn disable_automation(&mut self, actor_id: &str, now: DateTime<Utc>) {
        let event = self.policy.disable_auto(actor_id, now);
        self.store.record(event);
    }

    /// 노드 식별자의 솔트 해시. 관계망을 사건에 붙일 때 같은 솔트를 씁니다.
    pub fn node_hash(&self, id: &str) -> String {
        privacy::node_hash(&self.salt, id)
    }

    /// 민원 한 건을 Gate 0 부터 돌립니다. 모델이 없으면 규칙만 씁니다.
    ///
    /// 키오스크 접수 건은 모델이 실패해도 멈추지 않고 규칙으로 이어 갑니다. 민원인이 낸 내용을 잃으면 안 되기 때문입니다.
    pub fn run(&mut self, case: CaseFile, now: DateTime<Utc>) -> Result<CaseOutcome> {
        let case_id = case.case_id.clone();
        let model = self.model.as_deref();
        let kiosk = case.source == CaseSource::KioskForm;

        // Gate 0: 분류 (접수 양식이 정합니다)
        record(
            &mut self.store,
            now,
            Actor::System,
            AuditCategory::DecisionRecorded,
            &case_id,
            None,
            format!("분류: {}", gate0_router::verdict_of(case.category).code()),
        );
        if case.category == Category::Grievance {
            let draft = self.summarize(&case, now);
            let reason = match &draft {
                Some(d) if d.verified => format!("고충민원 요약 초안이 준비되었습니다. 담당 부서 후보: {}", d.department.as_deref().unwrap_or("미정")),
                Some(d) => format!("고충민원 요약 초안에 확인할 내용이 {}건 있습니다. 원문과 대조하세요", d.problems.len()),
                None => "고충민원은 요약과 부서 배정 후 공무원이 처리합니다".to_string(),
            };
            return self.hold_with_draft(case, None, reason, draft, now);
        }

        // Gate 1: 형식심사
        let report = match gate1_formal::evaluate(&case) {
            Gate1::SignatureRetry { verdict, reason } => {
                record(&mut self.store, now, Actor::CitizenKiosk, AuditCategory::DecisionRecorded, &case_id, None, reason.clone());
                return Ok(CaseOutcome::KioskRetry { verdict, reason });
            }
            Gate1::Reject { verdict, reason } => {
                return self.hold(case, Some(Judgement::by_rule(verdict)), reason, now);
            }
            Gate1::Hold { report } => {
                return self.hold(case, None, missing_reason(&report), now);
            }
            Gate1::Pass(report) => {
                if !report.duplicates.is_empty() {
                    record(
                        &mut self.store,
                        now,
                        Actor::System,
                        AuditCategory::DecisionRecorded,
                        &case_id,
                        None,
                        format!("중복 스캔 {}건을 한 장으로 합침", report.duplicates.len()),
                    );
                }
                report
            }
        };

        // 창구에서 확인할 일. 키오스크 접수 건만 생깁니다. 하나라도 있으면 자동 확정하지 않습니다.
        let mut notes = case.intake.pending.clone();
        if kiosk {
            if !report.missing.is_empty() {
                notes.push(format!("확인되지 않은 서류: {}", report.missing.join(", ")));
            }
            if !report.unexpected.is_empty() {
                notes.push(format!("이 민원에 필요 없는 서류: {}", report.unexpected.join(", ")));
            }
        }
        // 서류나 대장으로 확인되지 않은 값은 정정 · 잠금 · 민원인 재요청을 하지 않습니다.
        let unverified = !notes.is_empty();
        let soft = unverified || case.intake.counter_done;

        // Gate 2: 정합성 교차대조
        if let Some(bank) = &case.bank {
            if gate2_consistency::lock_account(bank).overwrote_handwritten {
                if unverified {
                    notes.push("입금받을 계좌와 통장 계좌가 다릅니다".to_string());
                } else {
                    record(
                        &mut self.store,
                        now,
                        Actor::ZoroGuardrail,
                        AuditCategory::FactLocked,
                        &case_id,
                        None,
                        "계좌번호를 인쇄 원장 값으로 고정 (수기 기재와 불일치)",
                    );
                }
            }
        }
        if let Some(lease) = &case.lease {
            // 자유 서술 소명은 표준 소명으로 바꿔 규칙에 넘깁니다. 바꾸지 못하면 그대로 공무원에게 갑니다.
            let normalized = match (self.writer.as_deref(), case.excuse.as_ref()) {
                (Some(writer), Some(Excuse::FreeText(text))) => drafting::normalize_excuse(writer, text).ok().flatten(),
                _ => None,
            };
            if normalized.is_some() {
                record(&mut self.store, now, Actor::System, AuditCategory::DecisionRecorded, &case_id, None, "자유 서술 소명을 표준 소명으로 정규화");
            }
            match gate2_consistency::check_rent(lease, normalized.as_ref().or(case.excuse.as_ref())) {
                Consistency::Consistent { note: None } => {}
                // 정정 문구는 확인된 값에서만 씁니다. 확인 전에는 차이가 있다는 사실만 남깁니다.
                Consistency::Consistent { note: Some(_) } if unverified => notes.push("신청 금액과 계약 월세가 다릅니다".to_string()),
                Consistency::Consistent { note: Some(note) } => {
                    record(&mut self.store, now, Actor::ZoroGuardrail, AuditCategory::FactLocked, &case_id, None, note);
                }
                Consistency::ClaimContradicted { reason, .. } if soft => notes.push(reason),
                Consistency::ClaimContradicted { verdict, reason } => {
                    record(&mut self.store, now, Actor::ZoroGuardrail, AuditCategory::ContradictionBlocked, &case_id, None, reason.clone());
                    return Ok(CaseOutcome::KioskRetry { verdict, reason });
                }
                // 키오스크 접수는 증빙을 창구에서 봅니다. 민원인에게 스캔을 다시 요구하지 않습니다.
                Consistency::EvidenceRequested { reason, .. } if kiosk => notes.push(reason),
                Consistency::EvidenceRequested { verdict, reason } => {
                    return Ok(CaseOutcome::KioskRetry { verdict, reason });
                }
                Consistency::OfficerReview { reason } if kiosk => notes.push(reason),
                Consistency::OfficerReview { reason } => {
                    return self.hold(case, None, reason, now);
                }
            }
        }
        let review = case.intake.subsidy.as_ref().map(gate2_consistency::review_receipts);
        if let Some(r) = &review {
            if !r.duplicates.is_empty() {
                notes.push(format!("중복 영수증 {}건을 한 장으로 합침", r.duplicates.len()));
            }
            if !r.barred.is_empty() {
                notes.push(format!("비인정 품목 {}건 제외", r.barred.len()));
            }
        }

        // Gate 3: 실질 요건 (관계망)
        // 식별 번호가 없으면 (관계망 미등록) 역할과 접수 번호로 구분한 값을 씁니다. 빈 값끼리 같아지면 같은 사람으로 잘못 봅니다.
        let identity = |role: &str, id: &str| {
            if id.is_empty() {
                privacy::node_hash(&self.salt, &format!("no-identifier:{role}:{case_id}"))
            } else {
                privacy::node_hash(&self.salt, id)
            }
        };
        let applicant_hash = identity("applicant", &case.applicant.identifier);
        let landlord = case.landlord.as_ref().map(|p| (identity("landlord", &p.identifier), p.name.clone()));
        let landlord_ref = landlord.as_ref().map(|(h, n)| (h.as_str(), n.as_str()));
        let judge = |model: Option<&dyn DecisionModel>| -> Result<Substantive> {
            match (case.intake.subsidy.as_ref(), case.intake.permit.as_ref(), case.intake.cert.as_ref()) {
                (Some(s), _, _) => gate3_substantive::evaluate_subsidy(&case.relations, s, review.as_ref().unwrap_or(&ReceiptReview::default()), model),
                (_, Some(p), _) => gate3_substantive::evaluate_permit(p, case.today, model),
                (_, _, Some(c)) => {
                    let days = case.poa_written_on.map(|d| gate1_formal::poa_elapsed_days(d, case.today)).unwrap_or(0);
                    gate3_substantive::evaluate_cert(days, c, model)
                }
                _ => gate3_substantive::evaluate(&case.relations, &applicant_hash, landlord_ref, model),
            }
        };
        let sub = match judge(model) {
            Ok(s) => s,
            Err(e) if kiosk => {
                record(&mut self.store, now, Actor::System, AuditCategory::DecisionRecorded, &case_id, None, format!("Gate 3 모델 실패, 규칙으로 진행: {e}"));
                judge(None)?
            }
            Err(e) => return Err(e),
        };
        if sub.homonym {
            record(
                &mut self.store,
                now,
                Actor::ZoroGuardrail,
                AuditCategory::DecisionRecorded,
                &case_id,
                None,
                "같은 이름이 연결망에 있으나 다른 사람으로 판별되었습니다 (동명이인)",
            );
        }
        if let Some(path) = &sub.close_path {
            record(
                &mut self.store,
                now,
                Actor::ZoroGuardrail,
                AuditCategory::DecisionRecorded,
                &case_id,
                None,
                format!("가까운 연결 {}홉 발견", path.len().saturating_sub(1)),
            );
        }
        if kiosk {
            notes.extend(verdict_reason(&case, &sub, review.as_ref()));
            if case.intake.identity == Identity::Declared {
                notes.push("본인 확인 필요 (신분증)".to_string());
            }
        }
        let judgement = sub.judgement;

        // Gate 1~3 규칙 검사를 모두 통과한 뒤에만 자동화 정책을 봅니다. Gate 4·5 는 처분이 정해진 뒤 실행합니다.
        // 창구에서 확인할 일이 남은 건은 규칙 검사가 끝나지 않은 것으로 봅니다 (자동 확정 금지).
        match self.policy.disposition(&judgement, notes.is_empty(), &case_id) {
            Disposition::AutoFinalize { post_audit } => {
                let final_code = judgement.verdict.code().to_string();
                let confidence = judgement.confidence;
                let verdict = judgement.verdict;
                self.store.open(ActiveSession {
                    case,
                    received_at: now,
                    judgement: Some(judgement),
                    reason: String::new(),
                    draft: None,
                })?;
                self.finish(
                    &case_id,
                    Closing {
                        form: Form::ApprovalNotice,
                        final_code,
                        notice: Some(verdict),
                        actor: Actor::ZoroGuardrail,
                        officer_id: None,
                        special_audit_flag: false,
                        post_audit,
                        confidence,
                    },
                    now,
                )
            }
            Disposition::PendingOfficer { reason } => {
                let reason = if notes.is_empty() { reason } else { notes.join(" · ") };
                self.hold(case, Some(judgement), reason, now)
            }
        }
    }

    /// 공무원이 보류된 건을 결정합니다. 확정이면 Gate 4·5 까지 실행합니다.
    pub fn confirm(
        &mut self,
        case_id: &str,
        officer_id: &str,
        decision: OfficerDecision,
        now: DateTime<Utc>,
    ) -> Result<CaseOutcome> {
        let (candidate, report, source) = {
            let Some(session) = self.store.active(case_id) else {
                bail!("진행 중인 민원이 아닙니다: {case_id}");
            };
            (session.judgement.clone(), gate1_formal::check_documents(&session.case), session.case.source)
        };
        let officer = Some(officer_id.to_string());

        match decision {
            OfficerDecision::Confirm | OfficerDecision::Reject => {
                // 양식 입력 건은 창구에서 원본을 모두 확인해야 확정할 수 있습니다. 확정이 막혀도 세션은 남습니다.
                if decision == OfficerDecision::Confirm && source == CaseSource::KioskForm && !report.missing.is_empty() {
                    record(
                        &mut self.store,
                        now,
                        Actor::ZoroGuardrail,
                        AuditCategory::IllegalActionIntercepted,
                        case_id,
                        Some(officer_id),
                        format!("누락 서류({})를 둔 채 확정 시도 → 차단", report.missing.join(", ")),
                    );
                    return Ok(CaseOutcome::Blocked {
                        verdict: Verdict::G1InterceptForceHold,
                        reason: format!("확인되지 않은 서류가 있습니다: {}. 원본을 확인하거나 보완요구(7일)를 발급하세요", report.missing.join(", ")),
                    });
                }
                let Some(j) = candidate else {
                    return Ok(no_candidate());
                };
                if decision == OfficerDecision::Confirm && j.verdict.class() == OutcomeClass::Procedural {
                    return Ok(CaseOutcome::OfficerReview {
                        judgement: Some(j),
                        reason: "증빙이 모자란 건은 확정할 수 없습니다. 인정 금액을 줄이거나 보완요구를 발급하세요".into(),
                    });
                }
                let (form, final_code) = if decision == OfficerDecision::Reject {
                    (Form::RejectNotice, format!("OFFICER_REJECT:{}", j.verdict.code()))
                } else if j.verdict.class() == OutcomeClass::Favorable {
                    (Form::ApprovalNotice, j.verdict.code().to_string())
                } else {
                    (Form::RejectNotice, j.verdict.code().to_string())
                };
                self.finish(
                    case_id,
                    Closing {
                        form,
                        final_code,
                        notice: Some(j.verdict),
                        actor: Actor::OfficerConsole,
                        officer_id: officer,
                        special_audit_flag: false,
                        post_audit: false,
                        confidence: j.confidence,
                    },
                    now,
                )
            }
            OfficerDecision::ApproveHighRisk(justification) => {
                let Some(j) = candidate else {
                    return Ok(no_candidate());
                };
                match gate3_substantive::officer_approve(j.verdict, Some(&justification)) {
                    Ok(OfficerOutcome::Reverted) => self.finish(
                        case_id,
                        Closing {
                            form: Form::RejectNotice,
                            final_code: format!("REVERTED_TO_REJECT:{}", j.verdict.code()),
                            notice: Some(j.verdict),
                            actor: Actor::OfficerConsole,
                            officer_id: officer,
                            special_audit_flag: false,
                            post_audit: false,
                            confidence: j.confidence,
                        },
                        now,
                    ),
                    Ok(OfficerOutcome::Approved { special_audit_flag }) => {
                        if special_audit_flag {
                            record(
                                &mut self.store,
                                now,
                                Actor::OfficerConsole,
                                AuditCategory::DecisionRecorded,
                                case_id,
                                Some(officer_id),
                                "고위험 강제 승인: 상급 감사실 직보 플래그",
                            );
                        }
                        self.finish(
                            case_id,
                            Closing {
                                form: Form::ApprovalNotice,
                                final_code: format!("OVERRIDE_APPROVED:{}", j.verdict.code()),
                                notice: None,
                                actor: Actor::OfficerConsole,
                                officer_id: officer,
                                special_audit_flag,
                                post_audit: false,
                                confidence: j.confidence,
                            },
                            now,
                        )
                    }
                    Err(verdict) => {
                        record(
                            &mut self.store,
                            now,
                            Actor::ZoroGuardrail,
                            AuditCategory::IllegalActionIntercepted,
                            case_id,
                            Some(officer_id),
                            format!("고위험 판정({})에 입증 없이 정상 승인 시도 → 차단", j.verdict.code()),
                        );
                        Ok(CaseOutcome::Blocked {
                            verdict,
                            reason: "고위험 판정은 입증 없이 승인할 수 없습니다. 증빙 스캔, 팀장 공동 결재, 조례 조항 중 하나가 필요합니다".into(),
                        })
                    }
                }
            }
            OfficerDecision::IssueSupplement => {
                let short_of_receipts = candidate.as_ref().is_some_and(|j| j.verdict == Verdict::G3HoldSimpleReceiptMissing);
                if report.is_complete() && !short_of_receipts {
                    return Ok(CaseOutcome::OfficerReview {
                        judgement: candidate,
                        reason: "누락된 서류가 없어 보완요구 대상이 아닙니다".into(),
                    });
                }
                let confidence = candidate.and_then(|j| j.confidence);
                self.finish(
                    case_id,
                    Closing {
                        form: Form::SupplementRequest,
                        final_code: "SUPPLEMENT_7DAYS".into(),
                        notice: None,
                        actor: Actor::OfficerConsole,
                        officer_id: officer,
                        special_audit_flag: false,
                        post_audit: false,
                        confidence,
                    },
                    now,
                )
            }
            OfficerDecision::ForcePassToGate2 => match gate1_formal::force_pass(&report) {
                Ok(()) => Ok(CaseOutcome::OfficerReview {
                    judgement: candidate,
                    reason: "서류가 완전하여 강제 진입이 필요하지 않습니다".into(),
                }),
                Err(verdict) => {
                    record(
                        &mut self.store,
                        now,
                        Actor::ZoroGuardrail,
                        AuditCategory::IllegalActionIntercepted,
                        case_id,
                        Some(officer_id),
                        format!("누락 서류({})를 둔 채 Gate 2 강제 진입 시도 → 차단", report.missing.join(", ")),
                    );
                    Ok(CaseOutcome::Blocked {
                        verdict,
                        reason: "누락된 서류가 있어 다음 단계로 넘어갈 수 없습니다. 보완요구(7일)를 발급하세요".into(),
                    })
                }
            },
        }
    }

    /// 제3자 실명 열람 요청 (명세 Case 3-C). 모델 위험도가 70% 이상일 때만 허용합니다.
    /// 허용과 거부는 모두 감사 로그에 남기고, 열람 대상의 이름은 로그에 쓰지 않습니다.
    pub fn request_unmask(&mut self, case_id: &str, officer_id: &str, now: DateTime<Utc>) -> Result<bool> {
        let risk = {
            let Some(session) = self.store.active(case_id) else {
                bail!("진행 중인 민원이 아닙니다: {case_id}");
            };
            session.judgement.as_ref().and_then(|j| j.confidence).unwrap_or(0.0)
        };
        if gate3_substantive::unmask_allowed(risk) {
            record(
                &mut self.store,
                now,
                Actor::ZoroGuardrail,
                AuditCategory::PrivacyUnmaskGranted,
                case_id,
                Some(officer_id),
                format!("위험도 {:.1}%: 제3자 실명 열람 허용", risk * 100.0),
            );
            Ok(true)
        } else {
            record(
                &mut self.store,
                now,
                Actor::ZoroGuardrail,
                AuditCategory::PrivacyUnmaskDenied,
                case_id,
                Some(officer_id),
                format!(
                    "위험도 {:.1}%: 기준 {:.0}% 미만이라 열람 거부",
                    risk * 100.0,
                    privacy::UNMASK_RISK_THRESHOLD * 100.0
                ),
            );
            Ok(false)
        }
    }

    /// 고충민원 본문을 요약합니다. 모델이 없거나 실패해도 민원은 멈추지 않습니다. 감사 로그에는 내용을 남기지 않습니다.
    fn summarize(&mut self, case: &CaseFile, now: DateTime<Utc>) -> Option<GrievanceSummary> {
        let (writer, text) = (self.writer.as_deref()?, case.grievance_text.as_deref()?);
        let result = drafting::summarize_grievance(writer, text);
        let note = match &result {
            Ok(d) if d.verified => "고충민원 요약 초안 작성: 숫자 대조 통과".to_string(),
            Ok(d) => format!("고충민원 요약 초안 작성: 확인 필요 {}건", d.problems.len()),
            Err(e) => format!("고충민원 요약 초안 실패: {e}"),
        };
        record(&mut self.store, now, Actor::System, AuditCategory::DecisionRecorded, &case.case_id, None, note);
        result.ok()
    }

    /// 내부결재 문서의 심사 의견 초안. 숫자가 어긋나면 넣지 않습니다.
    fn opinion(&self, facts: &[String]) -> Option<String> {
        drafting::review_opinion(self.writer.as_deref()?, facts).ok().flatten()
    }

    /// 공무원 확인을 기다리는 세션으로 남깁니다.
    fn hold(&mut self, case: CaseFile, judgement: Option<Judgement>, reason: String, now: DateTime<Utc>) -> Result<CaseOutcome> {
        self.hold_with_draft(case, judgement, reason, None, now)
    }

    fn hold_with_draft(&mut self, case: CaseFile, judgement: Option<Judgement>, reason: String, draft: Option<GrievanceSummary>, now: DateTime<Utc>) -> Result<CaseOutcome> {
        let case_id = case.case_id.clone();
        record(
            &mut self.store,
            now,
            Actor::System,
            AuditCategory::DecisionRecorded,
            &case_id,
            None,
            format!("공무원 확인 대기: {reason}"),
        );
        let received_at = case.intake.received_at.unwrap_or(now);
        self.store.open(ActiveSession {
            case,
            received_at,
            judgement: judgement.clone(),
            reason: reason.clone(),
            draft,
        })?;
        Ok(CaseOutcome::OfficerReview { judgement, reason })
    }

    /// Gate 4 (양식·문서 규칙) 와 Gate 5 (확정 기록) 를 실행합니다.
    fn finish(&mut self, case_id: &str, closing: Closing, now: DateTime<Utc>) -> Result<CaseOutcome> {
        let Closing { form, final_code, notice, actor, officer_id, special_audit_flag, post_audit, confidence } = closing;
        let officer = officer_id.as_deref();

        // 세션에서 필요한 값만 꺼냅니다.
        let (category, today, supplement_count, report, third_party, lease, reason) = {
            let Some(session) = self.store.active(case_id) else {
                bail!("진행 중인 민원이 아닙니다: {case_id}");
            };
            let case = &session.case;
            (
                case.category,
                case.today,
                case.supplement_count,
                gate1_formal::check_documents(case),
                case.landlord.iter().map(|p| p.name.clone()).collect::<Vec<String>>(),
                case.lease.clone(),
                session.reason.clone(),
            )
        };

        // Gate 4-A: 간단한 서류가 빠졌고 보완 기회가 없었다면 반려 대신 보완요구로 바꿉니다.
        let choice = g45::select_form(form, supplement_count, !report.missing.is_empty());
        let (form, notice, final_code) = if choice.overridden {
            record(
                &mut self.store,
                now,
                Actor::ZoroGuardrail,
                AuditCategory::IllegalActionIntercepted,
                case_id,
                officer,
                "보완 기회 없이 반려 양식을 고르려 함 → 보완요구 양식으로 전환",
            );
            (Form::SupplementRequest, None, "SUPPLEMENT_7DAYS".to_string())
        } else if form == Form::SupplementRequest {
            (form, None, "SUPPLEMENT_7DAYS".to_string())
        } else {
            (form, notice, final_code)
        };

        // Gate 4-B: 민원인 교부용 문서에 내부 용어나 제3자 실명이 있으면 막습니다.
        let citizen_doc = citizen_notice(case_id, category, form, notice, today);
        if let Err((verdict, leaks)) = g45::check_citizen_text(&citizen_doc, &third_party) {
            record(
                &mut self.store,
                now,
                Actor::ZoroGuardrail,
                AuditCategory::LeakBlocked,
                case_id,
                officer,
                format!("민원인 문서에서 내부 용어 {}건 차단", leaks.len()),
            );
            return Ok(CaseOutcome::Blocked {
                verdict,
                reason: "민원인 교부용 문서에 내부 기밀이나 제3자 정보가 섞여 발급을 막았습니다".into(),
            });
        }

        // Gate 5: 아카이브 적재 → 감사 기록 → 활성 세션 삭제
        let confidence_text = confidence
            .map(|c| format!("{:.1}%", c * 100.0))
            .unwrap_or_else(|| "없음(규칙·공무원 판단)".to_string());
        let mut officer_doc = format!(
            "내부결재 · 사건 {case_id} · 처분 {final_code} · Zoro-1 확신도 {confidence_text} · 처리 {actor:?}"
        );
        if let Some(opinion) = self.opinion(&opinion_facts(lease.as_ref(), &final_code, &reason)) {
            officer_doc.push_str(&format!("\n심사 의견 초안 (모델 작성, 공무원 확인 전): {opinion}"));
        }
        let archive_id = format!("{}-{}-{:04}호", self.office_code, now.format("%Y"), self.store.archive().len() + 1);
        let row = g45::build_archive_row(
            &archive_id,
            case_id,
            &final_code,
            officer_doc,
            citizen_doc,
            special_audit_flag,
            today,
        );
        let (category, label) = if actor == Actor::ZoroGuardrail {
            (AuditCategory::AutoFinalized, "자동화 모드 자동 확정")
        } else {
            (AuditCategory::OfficerConfirmed, "공무원 확정")
        };
        let event = AuditEvent::new(
            now,
            actor,
            category,
            case_id,
            officer,
            format!("{label}: {final_code}"),
        );
        self.store.finalize(case_id, row, vec![event])?;

        if actor == Actor::ZoroGuardrail {
            Ok(CaseOutcome::AutoFinalized { archive_id, post_audit })
        } else {
            Ok(CaseOutcome::Finalized { archive_id })
        }
    }
}

fn record(
    store: &mut Store,
    at: DateTime<Utc>,
    actor: Actor,
    category: AuditCategory,
    case_id: &str,
    officer_id: Option<&str>,
    detail: impl Into<String>,
) {
    store.record(AuditEvent::new(at, actor, category, case_id, officer_id, detail));
}

/// 심사 의견에 넘기는 사실. 이름과 식별 번호는 넣지 않습니다.
fn opinion_facts(lease: Option<&LeaseFacts>, final_code: &str, reason: &str) -> Vec<String> {
    let mut facts = vec![format!("처분: {}", drafting::disposition(final_code))];
    if !reason.is_empty() {
        facts.push(format!("판단 사유: {reason}"));
    }
    if let Some(lease) = lease {
        facts.push(format!("계약 월세: {}원", lease.contract_rent));
        facts.push(format!("최근 이체액: {}원", lease.latest_transfer));
    }
    facts
}

/// 공무원이 왜 이 건을 봐야 하는지 한 줄로 적습니다. 공무원 화면에만 나갑니다.
fn verdict_reason(case: &CaseFile, sub: &Substantive, review: Option<&ReceiptReview>) -> Option<String> {
    let hops = sub.close_path.as_ref().map(|p| p.len().saturating_sub(1));
    match sub.judgement.verdict {
        Verdict::G3ViolationHighRisk => match case.intake.permit.as_ref() {
            Some(p) => Some(match (&p.building_violation, &p.building_use) {
                (Some(v), _) => format!("위반건축물: {v}"),
                (None, Some(u)) => format!("건축물 용도가 맞지 않습니다: {u}"),
                _ => "영업신고 요건을 확인해 주세요".to_string(),
            }),
            None => Some(format!("임대인과 가까운 연결이 관계망에 있습니다 ({}홉)", hops.unwrap_or(0))),
        },
        Verdict::G3ViolationRelatedVendorSplit => {
            let mut parts = Vec::new();
            if let Some(h) = hops {
                parts.push(format!("거래처 특수관계 {h}홉"));
            }
            if let Some(r) = review.filter(|r| !r.splits.is_empty()) {
                parts.push(format!("분할 결제: {}", r.splits.join(", ")));
            }
            Some(parts.join(", "))
        }
        Verdict::G3HoldSimpleReceiptMissing => review.map(|r| format!("증빙이 {}원 모자랍니다", r.gap)),
        _ => None,
    }
}

fn no_candidate() -> CaseOutcome {
    CaseOutcome::OfficerReview {
        judgement: None,
        reason: "확정할 처분 후보가 없습니다. 보완요구 등 다른 결정을 고르세요".into(),
    }
}

fn missing_reason(report: &FormalReport) -> String {
    let join = |v: &[String]| if v.is_empty() { "없음".to_string() } else { v.join(", ") };
    format!(
        "서류 보완 필요 — 누락: {}, 불일치: {}",
        join(&report.missing),
        join(&report.unexpected)
    )
}

/// 민원인 교부용 본문. 내부 용어와 제3자 정보를 넣지 않습니다.
fn citizen_notice(case_id: &str, category: Category, form: Form, verdict: Option<Verdict>, today: NaiveDate) -> String {
    let approved = match category {
        Category::Subsidy => "심사 결과 정산을 승인합니다.",
        Category::Permit => "심사 결과 영업신고를 수리합니다.",
        Category::Cert => "심사 결과 발급 신청을 승인합니다.",
        _ => "심사 결과 지급 요건을 충족하여 승인합니다.",
    };
    let body = match (form, verdict) {
        (Form::SupplementRequest, _) => format!(
            "누락된 서류를 {}까지 제출해 주세요.",
            gate1_formal::supplement_deadline(today)
        ),
        (Form::ApprovalNotice, _) => approved.to_string(),
        (Form::RejectNotice, Some(Verdict::G1ExpiredMandatoryReject)) => {
            "위임장 작성일이 법정 유효기간을 넘어 재작성이 필요합니다.".to_string()
        }
        (Form::RejectNotice, Some(Verdict::G3ViolationHighRisk)) => {
            "관련 지침 제2조에 따라 지급 대상에서 제외됩니다. 이의가 있으면 처분일로부터 90일 이내에 이의신청 또는 행정심판을 청구할 수 있습니다.".to_string()
        }
        (Form::RejectNotice, _) => {
            "심사 결과 지급이 제외됩니다. 이의가 있으면 처분일로부터 90일 이내에 이의신청 또는 행정심판을 청구할 수 있습니다.".to_string()
        }
    };
    format!("심사 결과 안내 (사건번호 {case_id})\n{body}")
}
