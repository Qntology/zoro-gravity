//! 처분 선택지(Verdict)와 판정(Judgement).
//!
//! 코드 이름은 명세(Part 2~5)의 Options 코드를 그대로 따릅니다. 예: `G1_SIGN_VALID`.
//! 판정은 규칙이 정합니다. 규칙은 서류·수치·관계망이라는 증거에서 나옵니다.
//! Zoro-1 은 그 판정에 확신도를 주거나, 다른 선택지를 고르면 반대 의견으로 남습니다.
//! 자동화 정책은 확신도가 있을 때만 자동 확정을 허용합니다 (policy.rs).

use anyhow::{Result, bail};

use crate::decision::{DecisionModel, DecisionTask};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Verdict {
    // Gate 0: 5대 법정 민원 분류
    Cat01Welfare,
    Cat02Subsidy,
    Cat03Permit,
    Cat04Cert,
    Cat05Grievance,
    // Gate 1: 형식심사
    G1SignValid,
    G1SignInvalidRetry,
    G1ExpiredMandatoryReject,
    G1InterceptForceHold,
    // Gate 2: 정합성 교차대조
    G2ForceBankbookTruth,
    G2RejectContradictionLock,
    G2RequireRepairReceipt,
    G2BlockUnprovenEdit,
    // Gate 3: 실질 요건
    G3EligibleNormal,
    G3EligibleHomonymClear,
    G3ViolationHighRisk,
    G3HoldSimpleReceiptMissing,
    G3ViolationRelatedVendorSplit,
    G3InterceptRubberStamp,
    // Gate 4: 문서 생성
    G4ForceSupplementForm,
    G4BlockInternalLeak,
    G4InterceptDocCorruption,
}

pub const ALL: [Verdict; 22] = [
    Verdict::Cat01Welfare,
    Verdict::Cat02Subsidy,
    Verdict::Cat03Permit,
    Verdict::Cat04Cert,
    Verdict::Cat05Grievance,
    Verdict::G1SignValid,
    Verdict::G1SignInvalidRetry,
    Verdict::G1ExpiredMandatoryReject,
    Verdict::G1InterceptForceHold,
    Verdict::G2ForceBankbookTruth,
    Verdict::G2RejectContradictionLock,
    Verdict::G2RequireRepairReceipt,
    Verdict::G2BlockUnprovenEdit,
    Verdict::G3EligibleNormal,
    Verdict::G3EligibleHomonymClear,
    Verdict::G3ViolationHighRisk,
    Verdict::G3HoldSimpleReceiptMissing,
    Verdict::G3ViolationRelatedVendorSplit,
    Verdict::G3InterceptRubberStamp,
    Verdict::G4ForceSupplementForm,
    Verdict::G4BlockInternalLeak,
    Verdict::G4InterceptDocCorruption,
];

/// 자동화 정책이 보는 처분의 성격
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeClass {
    /// 자격 있음 (지급·승인 쪽). 조건을 모두 충족할 때만 자동 확정 후보가 됩니다.
    Favorable,
    /// 불리한 처분 (반려·위반 판정). 어떤 모드에서도 자동 확정하지 않습니다.
    Adverse,
    /// 절차 처리 (키오스크 재시도·보완요구·진행 차단·문서 가드). 공무원 확인 대상입니다.
    Procedural,
}

impl Verdict {
    pub fn code(self) -> &'static str {
        match self {
            Verdict::Cat01Welfare => "CAT_01_WELFARE",
            Verdict::Cat02Subsidy => "CAT_02_SUBSIDY",
            Verdict::Cat03Permit => "CAT_03_PERMIT",
            Verdict::Cat04Cert => "CAT_04_CERT",
            Verdict::Cat05Grievance => "CAT_05_GRIEVANCE",
            Verdict::G1SignValid => "G1_SIGN_VALID",
            Verdict::G1SignInvalidRetry => "G1_SIGN_INVALID_RETRY",
            Verdict::G1ExpiredMandatoryReject => "G1_EXPIRED_MANDATORY_REJECT",
            Verdict::G1InterceptForceHold => "G1_INTERCEPT_FORCE_HOLD",
            Verdict::G2ForceBankbookTruth => "G2_FORCE_BANKBOOK_TRUTH",
            Verdict::G2RejectContradictionLock => "G2_REJECT_CONTRADICTION_LOCK",
            Verdict::G2RequireRepairReceipt => "G2_REQUIRE_REPAIR_RECEIPT",
            Verdict::G2BlockUnprovenEdit => "G2_BLOCK_UNPROVEN_EDIT",
            Verdict::G3EligibleNormal => "G3_ELIGIBLE_NORMAL",
            Verdict::G3EligibleHomonymClear => "G3_ELIGIBLE_HOMONYM_CLEAR",
            Verdict::G3ViolationHighRisk => "G3_VIOLATION_HIGH_RISK",
            Verdict::G3HoldSimpleReceiptMissing => "G3_HOLD_SIMPLE_RECEIPT_MISSING",
            Verdict::G3ViolationRelatedVendorSplit => "G3_VIOLATION_RELATED_VENDOR_SPLIT",
            Verdict::G3InterceptRubberStamp => "G3_INTERCEPT_RUBBER_STAMP",
            Verdict::G4ForceSupplementForm => "G4_FORCE_SUPPLEMENT_FORM",
            Verdict::G4BlockInternalLeak => "G4_BLOCK_INTERNAL_LEAK",
            Verdict::G4InterceptDocCorruption => "G4_INTERCEPT_DOC_CORRUPTION",
        }
    }

    pub fn from_code(code: &str) -> Option<Self> {
        ALL.iter().copied().find(|v| v.code() == code)
    }

    pub fn class(self) -> OutcomeClass {
        match self {
            Verdict::G3EligibleNormal | Verdict::G3EligibleHomonymClear => OutcomeClass::Favorable,
            Verdict::G1ExpiredMandatoryReject | Verdict::G3ViolationHighRisk | Verdict::G3ViolationRelatedVendorSplit => OutcomeClass::Adverse,
            _ => OutcomeClass::Procedural,
        }
    }
}

/// 규칙 판정과 (선택적) Zoro-1 의견
#[derive(Debug, Clone, PartialEq)]
pub struct Judgement {
    /// 규칙(증거)이 정한 선택지
    pub verdict: Verdict,
    /// Zoro-1 이 이 선택지에 준 확률. 기준별로 물었으면 그중 가장 낮은 확률입니다.
    /// 모델이 없거나, 모델이 다른 판단을 냈으면 None. None 이면 자동 확정은 할 수 없습니다.
    pub confidence: Option<f32>,
}

impl Judgement {
    pub fn by_rule(verdict: Verdict) -> Self {
        Self { verdict, confidence: None }
    }
}

/// 규칙 판정에 모델 의견을 붙입니다.
///
/// 모델은 `task.options` 순서대로 확률을 냅니다. 모델이 `rule` 을 고르면 그 확률(합으로 다시 나눈 값)이 확신도가 되고,
/// 다른 선택지를 고르면 확신도는 None 입니다. 모델이 없으면 규칙 판정만 남습니다.
pub fn judge(rule: Verdict, task: &DecisionTask, model: Option<&dyn DecisionModel>) -> Result<Judgement> {
    let Some(model) = model else {
        return Ok(Judgement::by_rule(rule));
    };
    let Some((pick, probs)) = ask(task, model)? else {
        return Ok(Judgement::by_rule(rule));
    };
    let p_rule = task.options.iter().position(|v| *v == rule).map(|i| probs[i]);
    let confidence = if task.options[pick] == rule { p_rule } else { None };
    Ok(Judgement { verdict: rule, confidence })
}

/// 모델의 답 `(가장 높은 선택지, 합이 1 인 확률)`. 모델이 쓸 수 없는 값을 내면 None 입니다.
fn ask(task: &DecisionTask, model: &dyn DecisionModel) -> Result<Option<(usize, Vec<f32>)>> {
    let probs = model.probabilities(task)?;
    if probs.len() != task.options.len() {
        bail!("모델 확률 {}개가 선택지 {}개와 다릅니다", probs.len(), task.options.len());
    }
    let total: f32 = probs.iter().sum();
    if probs.is_empty() || total.is_nan() || total <= 0.0 || probs.iter().any(|p| !p.is_finite() || *p < 0.0) {
        return Ok(None);
    }
    let pick = probs.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    Ok(pick.map(|i| (i, probs.iter().map(|p| p / total).collect())))
}

/// 유리한 판정의 확신도를 기준별로 따로 물어 정합니다.
///
/// 유리한 처분은 여러 기준을 모두 만족해야 성립하는데, 한 질문으로 모아 물으면 모델이 위반 신호 하나를 놓칩니다.
/// 그래서 전체 질문에서 모델도 유리한 쪽을 골랐는지 보고, 기준마다 `(유리, 불리)` 두 선택지로 다시 물어 가장 낮은 확률을 확신도로 씁니다.
/// 어느 기준에서든 불리한 쪽이 앞서면 확신도는 None 입니다. 측정은 `docs/automation-mode.md`.
///
/// 불리하거나 절차인 판정은 자동 확정 대상이 아니므로 더 묻지 않고 [`judge`] 와 같습니다.
pub fn judge_by_criteria(
    rule: Verdict,
    task: &DecisionTask,
    pair: (Verdict, Verdict),
    criteria: &[&str],
    model: Option<&dyn DecisionModel>,
) -> Result<Judgement> {
    let Some(model) = model else {
        return Ok(Judgement::by_rule(rule));
    };
    if rule.class() != OutcomeClass::Favorable {
        return judge(rule, task, Some(model));
    }
    let Some((pick, _)) = ask(task, model)? else {
        return Ok(Judgement::by_rule(rule));
    };
    if task.options[pick].class() != OutcomeClass::Favorable {
        return Ok(Judgement::by_rule(rule));
    }
    let mut lowest = 1.0f32;
    for criterion in criteria {
        let question = DecisionTask {
            title: format!("{} 중 {criterion} 만 판단", task.title),
            facts: task.facts.clone(),
            options: vec![pair.0, pair.1],
        };
        let Some((pick, probs)) = ask(&question, model)? else {
            return Ok(Judgement::by_rule(rule));
        };
        if pick != 0 {
            return Ok(Judgement::by_rule(rule));
        }
        lowest = lowest.min(probs[0]);
    }
    Ok(Judgement { verdict: rule, confidence: Some(lowest) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Fact;

    /// 지정한 선택지에 확률을 몰아 주는 시험용 모델
    struct Prefers(Verdict);

    impl DecisionModel for Prefers {
        fn probabilities(&self, task: &DecisionTask) -> Result<Vec<f32>> {
            let n = task.options.len();
            Ok(task
                .options
                .iter()
                .map(|v| if *v == self.0 { 0.9 } else { 0.1 / (n - 1) as f32 })
                .collect())
        }
    }

    /// 선택지 수와 다른 길이를 돌려주는 잘못된 모델
    struct WrongLength;

    impl DecisionModel for WrongLength {
        fn probabilities(&self, _task: &DecisionTask) -> Result<Vec<f32>> {
            Ok(vec![1.0])
        }
    }

    fn task() -> DecisionTask {
        DecisionTask {
            title: "민원 종류 분류".into(),
            facts: vec![Fact::new("스캔된 서류", "신청서")],
            options: ALL[..5].to_vec(),
        }
    }

    #[test]
    fn codes_round_trip() {
        for v in ALL {
            assert_eq!(Verdict::from_code(v.code()), Some(v));
        }
        assert_eq!(Verdict::from_code("NOPE"), None);
    }

    #[test]
    fn classes_match_the_policy_split() {
        assert_eq!(Verdict::G3EligibleNormal.class(), OutcomeClass::Favorable);
        assert_eq!(Verdict::G3ViolationHighRisk.class(), OutcomeClass::Adverse);
        assert_eq!(Verdict::G1ExpiredMandatoryReject.class(), OutcomeClass::Adverse);
        assert_eq!(Verdict::G1SignInvalidRetry.class(), OutcomeClass::Procedural);
    }

    #[test]
    fn rule_only_judgement_has_no_confidence() -> Result<()> {
        let j = judge(Verdict::Cat01Welfare, &task(), None)?;
        assert_eq!(j, Judgement::by_rule(Verdict::Cat01Welfare));
        Ok(())
    }

    #[test]
    fn model_agreement_gives_confidence() -> Result<()> {
        let model = Prefers(Verdict::Cat01Welfare);
        let j = judge(Verdict::Cat01Welfare, &task(), Some(&model))?;
        assert_eq!(j.verdict, Verdict::Cat01Welfare);
        let c = j.confidence.expect("모델이 규칙과 같은 답을 냈으므로 확신도가 있어야 합니다");
        assert!((c - 0.9).abs() < 1e-5, "confidence {c}");
        Ok(())
    }

    #[test]
    fn model_disagreement_drops_confidence() -> Result<()> {
        let model = Prefers(Verdict::Cat02Subsidy);
        let j = judge(Verdict::Cat01Welfare, &task(), Some(&model))?;
        assert_eq!(j.verdict, Verdict::Cat01Welfare, "판정은 규칙이 정합니다");
        assert_eq!(j.confidence, None);
        Ok(())
    }

    #[test]
    fn a_model_with_the_wrong_number_of_probabilities_is_an_error() {
        assert!(judge(Verdict::Cat01Welfare, &task(), Some(&WrongLength)).is_err());
    }

    const PAIR: (Verdict, Verdict) = (Verdict::G3EligibleNormal, Verdict::G3ViolationHighRisk);
    const CRITERIA: [&str; 2] = ["친족 여부", "세대 여부"];

    /// 제목에 따라 정해 둔 유리한 쪽 확률을 답하는 대역. 질문을 세어 둡니다.
    struct Scripted {
        whole: Vec<f32>,
        by_criterion: Vec<(&'static str, f32)>,
        asked: std::sync::Mutex<Vec<String>>,
    }

    impl DecisionModel for Scripted {
        fn probabilities(&self, task: &DecisionTask) -> Result<Vec<f32>> {
            self.asked.lock().unwrap().push(task.title.clone());
            for (name, p) in &self.by_criterion {
                if task.title.contains(name) {
                    return Ok(vec![*p, 1.0 - *p]);
                }
            }
            Ok(self.whole.clone())
        }
    }

    fn scripted(whole: Vec<f32>, by_criterion: Vec<(&'static str, f32)>) -> Scripted {
        Scripted { whole, by_criterion, asked: std::sync::Mutex::new(Vec::new()) }
    }

    fn gate3() -> DecisionTask {
        DecisionTask {
            title: "임차 요건 심사".into(),
            facts: vec![Fact::new("가까운 연결 경로", "없음")],
            options: vec![Verdict::G3EligibleNormal, Verdict::G3EligibleHomonymClear, Verdict::G3ViolationHighRisk],
        }
    }

    #[test]
    fn the_lowest_criterion_sets_the_confidence() -> Result<()> {
        let m = scripted(vec![0.9, 0.05, 0.05], vec![("친족", 0.99), ("세대", 0.96)]);
        let j = judge_by_criteria(Verdict::G3EligibleNormal, &gate3(), PAIR, &CRITERIA, Some(&m))?;
        assert_eq!(j.verdict, Verdict::G3EligibleNormal);
        assert!((j.confidence.unwrap() - 0.96).abs() < 1e-6, "{j:?}");
        let asked = m.asked.lock().unwrap();
        assert_eq!(asked.len(), 3, "전체 질문 + 기준 두 개: {asked:?}");
        assert!(asked[1].contains("친족 여부 만 판단"));
        Ok(())
    }

    #[test]
    fn a_criterion_that_leans_adverse_removes_the_confidence() -> Result<()> {
        let m = scripted(vec![0.95, 0.03, 0.02], vec![("친족", 0.99), ("세대", 0.3)]);
        let j = judge_by_criteria(Verdict::G3EligibleNormal, &gate3(), PAIR, &CRITERIA, Some(&m))?;
        assert_eq!(j, Judgement::by_rule(Verdict::G3EligibleNormal), "판정은 규칙대로 두고 자동 확정 근거만 없앱니다");
        Ok(())
    }

    #[test]
    fn the_whole_question_may_name_another_favorable_option() -> Result<()> {
        // 규칙은 동명이인 정리, 모델은 정상을 골랐습니다. 둘 다 유리한 처분이므로 기준을 묻습니다.
        let m = scripted(vec![0.9, 0.05, 0.05], vec![("친족", 0.98), ("세대", 0.97)]);
        let j = judge_by_criteria(Verdict::G3EligibleHomonymClear, &gate3(), PAIR, &CRITERIA, Some(&m))?;
        assert_eq!(j.verdict, Verdict::G3EligibleHomonymClear);
        assert!((j.confidence.unwrap() - 0.97).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn a_whole_question_that_leans_adverse_stops_before_the_criteria() -> Result<()> {
        let m = scripted(vec![0.1, 0.1, 0.8], vec![("친족", 0.99), ("세대", 0.99)]);
        let j = judge_by_criteria(Verdict::G3EligibleNormal, &gate3(), PAIR, &CRITERIA, Some(&m))?;
        assert_eq!(j.confidence, None);
        assert_eq!(m.asked.lock().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn adverse_rulings_are_not_asked_about_criteria() -> Result<()> {
        let m = scripted(vec![0.01, 0.01, 0.98], vec![("친족", 0.99), ("세대", 0.99)]);
        let j = judge_by_criteria(Verdict::G3ViolationHighRisk, &gate3(), PAIR, &CRITERIA, Some(&m))?;
        assert!((j.confidence.unwrap() - 0.98).abs() < 1e-6, "위반은 한 번만 묻습니다");
        assert_eq!(m.asked.lock().unwrap().len(), 1);
        Ok(())
    }

    #[test]
    fn without_a_model_only_the_rule_remains() -> Result<()> {
        let j = judge_by_criteria(Verdict::G3EligibleNormal, &gate3(), PAIR, &CRITERIA, None)?;
        assert_eq!(j, Judgement::by_rule(Verdict::G3EligibleNormal));
        Ok(())
    }
}
