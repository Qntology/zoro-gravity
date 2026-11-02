//! Gate 3: 실질 요건 — 관계망 탐색, 동명이인 분리, 고위험 판정의 승인 통제 (명세 Part 4).
//!
//! 불리한 처분은 관계망에서 나온 증거(가까운 연결 경로)가 있을 때만 성립합니다.
//! 모델 확률이 높다는 것만으로는 불리한 처분이 되지 않습니다.

use anyhow::Result;
use chrono::NaiveDate;

use super::gate2_consistency::ReceiptReview;
use crate::case::{CertFacts, PermitFacts, Relation, SubsidyFacts};
use crate::decision::{DecisionModel, DecisionTask, Fact};
use crate::graph::{close_link_path, related_party_path, shares_name_with_other_node};
use crate::privacy;
use crate::verdict::{Judgement, Verdict, judge_by_criteria};

/// 멀티홉 탐색 한도 (명세: 2~3 홉)
pub const MAX_HOPS: usize = 3;

/// 이 게이트의 선택지
pub const OPTIONS: [Verdict; 3] = [
    Verdict::G3EligibleNormal,
    Verdict::G3EligibleHomonymClear,
    Verdict::G3ViolationHighRisk,
];

/// 유리한 판정을 자동 확정하기 전에 따로 묻는 기준 (적용 규칙의 조건들)
pub const CRITERIA: [&str; 3] = ["2촌 이내 친족 여부", "같은 세대 여부", "같은 계좌 여부"];

#[derive(Debug, Clone, PartialEq)]
pub struct Substantive {
    pub judgement: Judgement,
    /// 가까운 연결이 발견된 경로 (노드 해시)
    pub close_path: Option<Vec<String>>,
    /// 같은 이름이 연결망에 있지만 다른 사람으로 판별됨
    pub homonym: bool,
}

/// `landlord` 는 (해시, 이름) 입니다. 임대인이 없는 민원은 관계 검사 대상이 아닙니다.
pub fn evaluate(
    relations: &[Relation],
    applicant_hash: &str,
    landlord: Option<(&str, &str)>,
    model: Option<&dyn DecisionModel>,
) -> Result<Substantive> {
    let Some((landlord_hash, landlord_name)) = landlord else {
        return Ok(Substantive {
            judgement: Judgement::by_rule(Verdict::G3EligibleNormal),
            close_path: None,
            homonym: false,
        });
    };
    let close_path = close_link_path(relations, applicant_hash, landlord_hash, MAX_HOPS);
    let homonym = close_path.is_none() && shares_name_with_other_node(relations, landlord_hash, landlord_name);
    let rule = if close_path.is_some() {
        Verdict::G3ViolationHighRisk
    } else if homonym {
        Verdict::G3EligibleHomonymClear
    } else {
        Verdict::G3EligibleNormal
    };
    // 모델에는 관계망 결과를 사실로만 줍니다. 이름·주민번호 같은 식별 정보는 넣지 않습니다.
    let hops = close_path.as_ref().map(|p| p.len().saturating_sub(1));
    let task = DecisionTask {
        title: "임차 요건 심사".into(),
        facts: vec![
            Fact::new("적용 규칙", "임차인과 임대인이 2촌 이내 친족이거나 같은 세대·같은 계좌이면 지급 제외"),
            Fact::new(
                "가까운 연결 경로",
                match hops {
                    Some(h) => format!("있음 ({h}홉)"),
                    None => "없음".to_string(),
                },
            ),
            Fact::new(
                "동명이인 후보",
                // 사람을 가른 근거를 그대로 적습니다. 대장 조회가 성명과 생년월일로 한 사람을 정하기 때문입니다.
                if homonym { "예 (생년월일이 달라 다른 사람으로 확인)" } else { "아니오" },
            ),
        ],
        options: OPTIONS.to_vec(),
    };
    let judgement = judge_by_criteria(rule, &task, (Verdict::G3EligibleNormal, Verdict::G3ViolationHighRisk), &CRITERIA, model)?;
    Ok(Substantive { judgement, close_path, homonym })
}

/// 보조금 집행 심사의 선택지 (명세 Part 6 §1.2)
pub const SUBSIDY_OPTIONS: [Verdict; 3] = [Verdict::G3EligibleNormal, Verdict::G3HoldSimpleReceiptMissing, Verdict::G3ViolationRelatedVendorSplit];
pub const SUBSIDY_CRITERIA: [&str; 2] = ["거래처 특수관계 여부", "입찰 회피 분할 결제 여부"];
/// 영업신고 · 위임장 심사의 선택지
pub const REVIEW_OPTIONS: [Verdict; 2] = [Verdict::G3EligibleNormal, Verdict::G3ViolationHighRisk];
pub const PERMIT_CRITERIA: [&str; 3] = ["위반건축물 여부", "건축물 용도 적합 여부", "필증 유효 여부"];
pub const CERT_CRITERIA: [&str; 2] = ["유효기간 여부", "위임자 서명 여부"];

/// 지원금 정산: 관계망 경로와 분할 결제는 규칙이 정하고, 모델은 유리한 판정에 확신도만 줍니다.
pub fn evaluate_subsidy(relations: &[Relation], facts: &SubsidyFacts, review: &ReceiptReview, model: Option<&dyn DecisionModel>) -> Result<Substantive> {
    let close_path = facts
        .org_node
        .as_deref()
        .and_then(|org| facts.vendor_nodes.iter().find_map(|(_, vendor)| related_party_path(relations, org, vendor, MAX_HOPS)));
    let rule = if close_path.is_some() || !review.splits.is_empty() {
        Verdict::G3ViolationRelatedVendorSplit
    } else if review.gap > 0 {
        Verdict::G3HoldSimpleReceiptMissing
    } else {
        Verdict::G3EligibleNormal
    };
    let task = DecisionTask {
        title: "보조금 집행 심사".into(),
        facts: vec![
            Fact::new("적용 규칙", "특수관계 거래처 집행과 50만 원 입찰 회피 분할 결제는 인정하지 않음"),
            Fact::new("거래처 관계", close_path.as_ref().map_or("없음".to_string(), |p| format!("있음 ({}홉)", p.len().saturating_sub(1)))),
            Fact::new("분할 결제", if review.splits.is_empty() { "없음" } else { "같은 날 같은 거래처 연속 결제" }),
            Fact::new(
                "영수증 대조",
                if review.gap > 0 { format!("신청액보다 증빙 합계가 {}원 적음", review.gap) } else { "신청액과 증빙 합계 일치".to_string() },
            ),
        ],
        options: SUBSIDY_OPTIONS.to_vec(),
    };
    let judgement = judge_by_criteria(rule, &task, (Verdict::G3EligibleNormal, Verdict::G3ViolationRelatedVendorSplit), &SUBSIDY_CRITERIA, model)?;
    Ok(Substantive { judgement, close_path, homonym: false })
}

/// 일반음식점이 들어갈 수 있는 건축물 용도인지 봅니다.
pub fn use_allows_restaurant(usage: &str) -> bool {
    usage.contains("근린생활시설")
}

/// 영업신고: 위반건축물과 용도는 규칙이 정합니다. 필증 만료는 서류 누락으로 다룹니다 (`evidence.rs`).
pub fn evaluate_permit(facts: &PermitFacts, today: NaiveDate, model: Option<&dyn DecisionModel>) -> Result<Substantive> {
    let bad_use = facts.building_use.as_deref().is_some_and(|u| !use_allows_restaurant(u));
    let rule = if facts.building_violation.is_some() || bad_use { Verdict::G3ViolationHighRisk } else { Verdict::G3EligibleNormal };
    let valid = |e: Option<NaiveDate>| match e {
        Some(d) if d >= today => "유효".to_string(),
        Some(d) => format!("{d} 만료"),
        None => "미확인".to_string(),
    };
    let task = DecisionTask {
        title: "영업신고 수리 심사".into(),
        facts: vec![
            Fact::new("적용 규칙", "위반건축물이거나 용도가 맞지 않거나 필증이 만료되면 수리 불가"),
            Fact::new("위반건축물 표기", facts.building_violation.as_deref().unwrap_or("없음")),
            Fact::new("건축물 용도", facts.building_use.as_deref().unwrap_or("미확인")),
            Fact::new("위생교육필증", valid(facts.hygiene_expires)),
            Fact::new("소방안전필증", valid(facts.fire_expires)),
        ],
        options: REVIEW_OPTIONS.to_vec(),
    };
    let judgement = judge_by_criteria(rule, &task, (Verdict::G3EligibleNormal, Verdict::G3ViolationHighRisk), &PERMIT_CRITERIA, model)?;
    Ok(Substantive { judgement, close_path: None, homonym: false })
}

/// 증명 · 대리: 유효기간 초과는 Gate 1 이 막으므로 여기까지 온 위임장은 기간 안입니다. 모델은 확신도만 줍니다.
pub fn evaluate_cert(elapsed_days: i64, facts: &CertFacts, model: Option<&dyn DecisionModel>) -> Result<Substantive> {
    let signed = match facts.principal_signed {
        Some(true) => "있음",
        Some(false) => "없음",
        None => "미확인",
    };
    let task = DecisionTask {
        title: "위임장 유효성 심사".into(),
        facts: vec![
            Fact::new("적용 규칙", "위임장은 작성일로부터 6개월 이내여야 하고 위임자 자필 서명이나 도장이 있어야 함"),
            Fact::new("작성일 경과", format!("{elapsed_days}일")),
            Fact::new("위임자 서명", signed),
        ],
        options: REVIEW_OPTIONS.to_vec(),
    };
    let judgement = judge_by_criteria(Verdict::G3EligibleNormal, &task, (Verdict::G3EligibleNormal, Verdict::G3ViolationHighRisk), &CERT_CRITERIA, model)?;
    Ok(Substantive { judgement, close_path: None, homonym: false })
}

/// 고위험 판정 건을 공무원이 `정상 승인`으로 바꾸려 할 때 내미는 입증 (명세 Case 3-D)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Justification {
    /// 클릭 실수: 반려 처분으로 되돌립니다.
    CancelMistake,
    /// 동명이인·DB 오류 입증: 원본 증빙 스캔이 1장 이상 있어야 합니다.
    ProveFalsePositive { evidence_scans: u32 },
    /// 조례 예외 적용: 팀장급 공동 결재와 조항 번호가 있어야 합니다. 상급 감사실 보고 플래그가 붙습니다.
    ApplyLawException { team_lead_cosign: bool, rule_ref: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OfficerOutcome {
    /// 반려 처분이 유지됩니다 (실수 되돌림)
    Reverted,
    /// 정상 승인으로 확정합니다
    Approved { special_audit_flag: bool },
}

/// 공무원의 승인 시도를 검사합니다. 고위험 판정이 아니면 그대로 승인입니다.
pub fn officer_approve(verdict: Verdict, justification: Option<&Justification>) -> Result<OfficerOutcome, Verdict> {
    if !matches!(verdict, Verdict::G3ViolationHighRisk | Verdict::G3ViolationRelatedVendorSplit) {
        return Ok(OfficerOutcome::Approved { special_audit_flag: false });
    }
    match justification {
        None => Err(Verdict::G3InterceptRubberStamp),
        Some(Justification::CancelMistake) => Ok(OfficerOutcome::Reverted),
        Some(Justification::ProveFalsePositive { evidence_scans }) if *evidence_scans > 0 => {
            Ok(OfficerOutcome::Approved { special_audit_flag: false })
        }
        Some(Justification::ProveFalsePositive { .. }) => Err(Verdict::G3InterceptRubberStamp),
        Some(Justification::ApplyLawException { team_lead_cosign: true, rule_ref })
            if !rule_ref.trim().is_empty() =>
        {
            Ok(OfficerOutcome::Approved { special_audit_flag: true })
        }
        Some(Justification::ApplyLawException { .. }) => Err(Verdict::G3InterceptRubberStamp),
    }
}

/// 제3자 실명 열람 허용 여부 (명세 Case 3-C). 허용과 거부 기록은 호출자가 감사 로그에 남깁니다.
pub fn unmask_allowed(risk: f32) -> bool {
    privacy::unmask_allowed(risk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::case::{Link, Node};

    fn node(hash: &str, name: &str) -> Node {
        Node { hash: hash.into(), name: name.into() }
    }

    #[test]
    fn homonym_is_cleared_not_flagged() -> Result<()> {
        // 명세 Case 3-A: 임대인과 모친 이름이 같지만 다른 사람
        let rels = [Relation {
            a: node("applicant", "김철수"),
            b: node("mother", "박영희"),
            link: Link::Family { degree: 1 },
        }];
        let s = evaluate(&rels, "applicant", Some(("landlord", "박영희")), None)?;
        assert_eq!(s.judgement.verdict, Verdict::G3EligibleHomonymClear);
        assert!(s.homonym);
        assert!(s.close_path.is_none());
        Ok(())
    }

    #[test]
    fn close_family_landlord_is_a_violation_with_a_path() -> Result<()> {
        // 명세 Case 3-B: 임대인이 모친 본인
        let rels = [Relation {
            a: node("applicant", "김철수"),
            b: node("mother", "박영희"),
            link: Link::Household { year: 2024 },
        }];
        let s = evaluate(&rels, "applicant", Some(("mother", "박영희")), None)?;
        assert_eq!(s.judgement.verdict, Verdict::G3ViolationHighRisk);
        assert_eq!(s.close_path.as_ref().map(Vec::len), Some(2));
        Ok(())
    }

    /// 모델이 받은 질문을 기록하는 시험용 모델
    #[derive(Default)]
    struct Recorder(std::sync::Mutex<Option<DecisionTask>>);

    impl DecisionModel for Recorder {
        fn probabilities(&self, task: &DecisionTask) -> Result<Vec<f32>> {
            *self.0.lock().unwrap() = Some(task.clone());
            let n = task.options.len();
            Ok(vec![1.0 / n as f32; n])
        }
    }

    #[test]
    fn the_model_never_sees_names_or_identifiers() -> Result<()> {
        let rels = [Relation {
            a: node("hash-applicant-xyz", "김철수"),
            b: node("hash-mother-xyz", "박영희"),
            link: Link::Family { degree: 1 },
        }];
        let recorder = Recorder::default();
        evaluate(&rels, "hash-applicant-xyz", Some(("hash-mother-xyz", "박영희")), Some(&recorder))?;
        let task = recorder.0.lock().unwrap().clone().expect("모델이 호출돼야 합니다");
        let shown = format!("{task:?}");
        for secret in ["김철수", "박영희", "hash-applicant-xyz", "hash-mother-xyz"] {
            assert!(!shown.contains(secret), "모델에 보이면 안 되는 값이 있습니다: {secret}");
        }
        assert!(shown.contains("있음 (1홉)"), "관계망 결과는 사실로 전달돼야 합니다");
        Ok(())
    }

    #[test]
    fn no_landlord_means_nothing_to_check() -> Result<()> {
        let s = evaluate(&[], "applicant", None, None)?;
        assert_eq!(s.judgement.verdict, Verdict::G3EligibleNormal);
        Ok(())
    }

    #[test]
    fn rubber_stamping_a_high_risk_case_is_blocked_without_proof() {
        // 명세 Case 3-D
        assert_eq!(
            officer_approve(Verdict::G3ViolationHighRisk, None),
            Err(Verdict::G3InterceptRubberStamp)
        );
        assert_eq!(
            officer_approve(Verdict::G3ViolationHighRisk, Some(&Justification::CancelMistake)),
            Ok(OfficerOutcome::Reverted)
        );
        assert_eq!(
            officer_approve(
                Verdict::G3ViolationHighRisk,
                Some(&Justification::ProveFalsePositive { evidence_scans: 0 })
            ),
            Err(Verdict::G3InterceptRubberStamp)
        );
        assert_eq!(
            officer_approve(
                Verdict::G3ViolationHighRisk,
                Some(&Justification::ApplyLawException { team_lead_cosign: true, rule_ref: "조례 제5조".into() })
            ),
            Ok(OfficerOutcome::Approved { special_audit_flag: true })
        );
        assert_eq!(
            officer_approve(
                Verdict::G3ViolationHighRisk,
                Some(&Justification::ApplyLawException { team_lead_cosign: false, rule_ref: "조례 제5조".into() })
            ),
            Err(Verdict::G3InterceptRubberStamp)
        );
    }

    #[test]
    fn unmask_needs_seventy_percent_risk() {
        // 명세 Case 3-C: 위험도 1.5% 인 정상 건은 열람 거부
        assert!(!unmask_allowed(0.015));
        assert!(unmask_allowed(0.964));
    }
}
