//! Gate 3: 실질 요건 — 관계망 탐색, 동명이인 분리, 고위험 판정의 승인 통제 (명세 Part 4).
//!
//! 불리한 처분은 관계망에서 나온 증거(가까운 연결 경로)가 있을 때만 성립합니다.
//! 모델 확률이 높다는 것만으로는 불리한 처분이 되지 않습니다.

use anyhow::Result;

use crate::case::Relation;
use crate::decision::{DecisionModel, DecisionTask, Fact};
use crate::graph::{close_link_path, shares_name_with_other_node};
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
                if homonym { "예 (같은 이름이 연결망에 있으나 다른 사람으로 판별)" } else { "아니오" },
            ),
        ],
        options: OPTIONS.to_vec(),
    };
    let judgement = judge_by_criteria(rule, &task, (Verdict::G3EligibleNormal, Verdict::G3ViolationHighRisk), &CRITERIA, model)?;
    Ok(Substantive { judgement, close_path, homonym })
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
    if verdict != Verdict::G3ViolationHighRisk {
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
