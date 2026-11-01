//! 판단 한 번의 입력(사실·선택지)과 모델 인터페이스.
//!
//! 규칙 엔진은 서류·수치·관계망에서 판정을 정하고, 모델은 같은 질문에 선택지별 확률을 냅니다.
//! 모델에 주는 `facts` 에는 개인 식별 정보(이름, 주민등록번호, 계좌번호)를 넣지 않습니다.
//! 엔진은 이 인터페이스에만 기대므로, 실제 모델(백본 + `jev_head`)이든 시험용 대역이든 같은 방식으로 꽂힙니다.

use anyhow::Result;

use crate::verdict::Verdict;

/// 모델에 주는 사실 한 줄
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fact {
    pub key: String,
    pub value: String,
}

impl Fact {
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Self { key: key.into(), value: value.into() }
    }
}

/// 판단 질문: 제목, 사실, 선택지. 선택지 순서가 확률 벡터의 순서입니다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionTask {
    pub title: String,
    pub facts: Vec<Fact>,
    pub options: Vec<Verdict>,
}

pub trait DecisionModel: Send + Sync {
    /// `task.options` 순서대로의 확률. 합은 1 입니다.
    fn probabilities(&self, task: &DecisionTask) -> Result<Vec<f32>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn facts_keep_their_order() {
        let task = DecisionTask {
            title: "t".into(),
            facts: vec![Fact::new("b", "2"), Fact::new("a", "1")],
            options: vec![Verdict::G3EligibleNormal],
        };
        assert_eq!(task.facts[0].key, "b");
        assert_eq!(task.facts[1], Fact::new("a", "1"));
    }
}
