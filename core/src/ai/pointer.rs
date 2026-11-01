//! 포인터 판단 프로토콜: 판단 질문을 프롬프트로 만들고, 백본의 은닉 상태를 `jev_head` 로 점수화합니다.
//!
//! 프롬프트는 명세 Part 6 의 표준 JSON (`zoro_jev_head_options` 배열) 을 그대로 쓰고, 채팅 턴 안에 넣습니다.
//! - 질의 h: 프롬프트 맨 끝(어시스턴트 턴이 열린 직후) 바이트를 덮는 토큰의 은닉 상태
//! - 선택지 e_i: 옵션 배열의 i 번째 코드를 닫는 따옴표를 덮는 토큰의 은닉 상태. 코드를 다 읽은 다음 토큰입니다.
//!
//! 위치는 토큰 번호가 아니라 바이트 위치로 넘깁니다. 토큰화는 백본 쪽 일이고, 토크나이저가 주는 offsets 로 대응하면 됩니다.
//! 모델 폴더에는 이 포인터를 위한 전용 토큰이 없어서 (Qwen2 방식 토크나이저 + `<|mask|>` 뿐), 위 약속은 파일에서 읽은 것이 아니라
//! 정답 사례로 재서 고른 것입니다. 선택지 위치만 판단을 크게 바꿨습니다 (docs/jev-head.md 6절).

use anyhow::{Result, bail};

use super::jev_head::{HIDDEN, JevHead};
use crate::decision::{DecisionModel, DecisionTask};

/// 시스템 지시문. 형식이 바뀌면 점수의 의미가 달라지므로 이곳 한 군데에만 둡니다.
pub const SYSTEM_PROMPT: &str = "행정 민원 심사 보조입니다. 사실을 보고 zoro_jev_head_options 중 하나를 고르세요.";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointerPrompt {
    pub text: String,
    /// 질의 위치: 이 바이트를 덮는 토큰
    pub query_byte: usize,
    /// 선택지 i 의 위치: 이 바이트를 덮는 토큰 (코드를 닫는 따옴표)
    pub option_bytes: Vec<usize>,
}

fn json_string(s: &str) -> String {
    // 문자열 하나를 직렬화하는 일은 실패하지 않습니다.
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}

impl PointerPrompt {
    pub fn render(task: &DecisionTask) -> Self {
        let mut text = String::new();
        text.push_str("<|im_start|>system\n");
        text.push_str(SYSTEM_PROMPT);
        text.push_str("<|im_end|>\n<|im_start|>user\n");
        text.push_str("{\n  \"task\": ");
        text.push_str(&json_string(&task.title));
        text.push_str(",\n  \"facts\": {");
        for (i, fact) in task.facts.iter().enumerate() {
            if i > 0 {
                text.push_str(", ");
            }
            text.push_str(&json_string(&fact.key));
            text.push_str(": ");
            text.push_str(&json_string(&fact.value));
        }
        text.push_str("},\n  \"zoro_jev_head_options\": [");
        let mut option_bytes = Vec::with_capacity(task.options.len());
        for (i, option) in task.options.iter().enumerate() {
            if i > 0 {
                text.push_str(", ");
            }
            text.push('"');
            // 코드는 ASCII([A-Z0-9_]) 라서 이스케이프가 필요 없습니다.
            text.push_str(option.code());
            option_bytes.push(text.len());
            text.push('"');
        }
        text.push_str("]\n}<|im_end|>\n<|im_start|>assistant\n");
        let query_byte = text.len() - 1;
        Self { text, query_byte, option_bytes }
    }
}

/// 백본이 돌려주는 은닉 상태 (최종 정규화 뒤, 각 2560 차원)
#[derive(Debug, Clone, PartialEq)]
pub struct PointerHidden {
    /// 질의 위치의 은닉 상태 (길이 2560)
    pub query: Vec<f32>,
    /// 선택지 위치의 은닉 상태 (행 우선, 길이 선택지 수 × 2560)
    pub options: Vec<f32>,
}

/// 프롬프트를 읽어 지정한 위치의 은닉 상태를 돌려주는 백본.
/// 실제 구현은 `ai::zoro::ZoroBackbone` 이고, 시험에서는 대역을 꽂습니다.
pub trait Backbone: Send + Sync {
    fn encode(&self, prompt: &PointerPrompt) -> Result<PointerHidden>;
}

/// 백본 + `jev_head` 로 판단 모델을 만듭니다. 엔진은 이것을 `DecisionModel` 로 씁니다.
pub struct PointerModel {
    backbone: Box<dyn Backbone>,
    head: JevHead,
}

impl PointerModel {
    pub fn new(backbone: Box<dyn Backbone>, head: JevHead) -> Self {
        Self { backbone, head }
    }
}

impl DecisionModel for PointerModel {
    fn probabilities(&self, task: &DecisionTask) -> Result<Vec<f32>> {
        if task.options.is_empty() {
            bail!("선택지가 없습니다");
        }
        let prompt = PointerPrompt::render(task);
        let hidden = self.backbone.encode(&prompt)?;
        if hidden.query.len() != HIDDEN {
            bail!("질의 은닉 상태 길이 {} != {HIDDEN}", hidden.query.len());
        }
        let expected = task.options.len() * HIDDEN;
        if hidden.options.len() != expected {
            bail!("선택지 은닉 상태 길이 {} != {expected}", hidden.options.len());
        }
        self.head.probabilities(&hidden.query, &hidden.options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::jev_head::QK_DIM;
    use crate::decision::Fact;
    use crate::verdict::Verdict;
    use candle_core::{DType, Device, Tensor};

    fn task() -> DecisionTask {
        DecisionTask {
            title: "임차 요건 심사".into(),
            facts: vec![
                Fact::new("가까운 연결 경로", "있음 (1홉)"),
                Fact::new("메모", "따옴표 \" 와 줄바꿈\n포함"),
            ],
            options: vec![
                Verdict::G3EligibleNormal,
                Verdict::G3EligibleHomonymClear,
                Verdict::G3ViolationHighRisk,
            ],
        }
    }

    /// 앞쪽 256 차원을 그대로 투영하는 시험용 헤드 (질의와 키가 같은 방향을 보면 점수가 높음)
    fn aligned_head() -> Result<JevHead> {
        let mut w = vec![0f32; QK_DIM * HIDDEN];
        for i in 0..QK_DIM {
            w[i * HIDDEN + i] = 1.0;
        }
        let proj = Tensor::from_vec(w, (QK_DIM, HIDDEN), &Device::Cpu)?;
        let zeros = Tensor::zeros(QK_DIM, DType::F32, &Device::Cpu)?;
        JevHead::from_tensors(proj.clone(), zeros.clone(), proj, zeros)
    }

    /// 지정한 선택지의 은닉 상태가 질의와 같은 방향을 보게 만드는 시험용 백본
    struct Scripted {
        prefer: usize,
    }

    impl Backbone for Scripted {
        fn encode(&self, prompt: &PointerPrompt) -> Result<PointerHidden> {
            let n = prompt.option_bytes.len();
            let mut query = vec![0f32; HIDDEN];
            query[3] = 8.0;
            let mut options = vec![0f32; n * HIDDEN];
            for i in 0..n {
                options[i * HIDDEN + if i == self.prefer { 3 } else { 10 }] = 8.0;
            }
            Ok(PointerHidden { query, options })
        }
    }

    struct WrongSize;

    impl Backbone for WrongSize {
        fn encode(&self, _prompt: &PointerPrompt) -> Result<PointerHidden> {
            Ok(PointerHidden { query: vec![0.0; HIDDEN], options: vec![0.0; HIDDEN] })
        }
    }

    #[test]
    fn option_positions_point_at_the_quote_closing_each_code() {
        let t = task();
        let p = PointerPrompt::render(&t);
        assert!(p.text.starts_with("<|im_start|>system\n"));
        assert!(p.text.ends_with("<|im_start|>assistant\n"));
        assert_eq!(p.query_byte, p.text.len() - 1);
        assert_eq!(p.option_bytes.len(), t.options.len());
        for (i, option) in t.options.iter().enumerate() {
            let end = p.option_bytes[i];
            assert!(p.text[..end].ends_with(option.code()), "선택지 {i}: {}", option.code());
            assert_eq!(&p.text[end..end + 1], "\"", "코드 바로 뒤의 닫는 따옴표를 가리켜야 합니다");
        }
        assert!(p.option_bytes.windows(2).all(|w| w[0] < w[1]), "선택지는 프롬프트 순서대로");
    }

    #[test]
    fn the_embedded_json_is_valid_and_keeps_facts_and_options() {
        let t = task();
        let p = PointerPrompt::render(&t);
        let start = p.text.find("<|im_start|>user\n").unwrap() + "<|im_start|>user\n".len();
        let end = p.text.find("<|im_end|>\n<|im_start|>assistant").unwrap();
        let json: serde_json::Value = serde_json::from_str(&p.text[start..end]).expect("JSON 이어야 합니다");
        assert_eq!(json["task"], "임차 요건 심사");
        assert_eq!(json["facts"]["메모"], "따옴표 \" 와 줄바꿈\n포함");
        let options: Vec<&str> = json["zoro_jev_head_options"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert_eq!(options, t.options.iter().map(|v| v.code()).collect::<Vec<_>>());
    }

    #[test]
    fn a_pointer_model_returns_probabilities_in_option_order() -> Result<()> {
        let t = task();
        for prefer in 0..t.options.len() {
            let model = PointerModel::new(Box::new(Scripted { prefer }), aligned_head()?);
            let p = model.probabilities(&t)?;
            assert_eq!(p.len(), t.options.len());
            assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            let best = p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
            assert_eq!(best, Some(prefer), "{p:?}");
        }
        Ok(())
    }

    #[test]
    fn wrong_sized_hidden_states_and_empty_options_are_errors() -> Result<()> {
        let model = PointerModel::new(Box::new(WrongSize), aligned_head()?);
        assert!(model.probabilities(&task()).is_err());
        let mut empty = task();
        empty.options.clear();
        assert!(model.probabilities(&empty).is_err());
        Ok(())
    }
}
