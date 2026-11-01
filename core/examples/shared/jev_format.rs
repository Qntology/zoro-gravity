//! `jev_head` 프롬프트 형식 후보. 탐색(`jev_format_search`)과 자동화 실험(`automation_eval`)이 같은 글을 쓰도록 여기에 둡니다.

use zorogravity_core::ai::pointer::{PointerPrompt, SYSTEM_PROMPT};

pub const WRAPPERS: [&str; 7] = ["시스템+턴", "턴", "시스템+턴+think", "시스템+턴+빈think", "턴+think", "턴+빈think", "틀없음"];
pub const BODIES: [&str; 7] = ["task/facts", "최상위키", "context/options", "글", "한줄JSON", "줄목록", "제목+줄목록"];
pub const QUERIES: [&str; 4] = ["끝", "턴닫음", "assistant", "선택지닫음"];
pub const OPTION_POS: [&str; 5] = ["끝글자", "첫글자", "평균", "여는따옴표", "닫는따옴표"];
pub const NORMS: [&str; 2] = ["정규화후", "정규화전"];

/// `시스템+턴`, `한줄JSON`, `끝`, `끝글자`, `정규화후`: 탐색 전에 쓰던 조합
pub const BEFORE: Format = Format { wrapper: 0, body: 4, query: 0, option: 0, norm: 0 };
/// 탐색 뒤에 고른 조합: 선택지 위치만 닫는따옴표로 바꿨습니다. `PointerPrompt::render` 와 같은 글입니다.
pub const CHOSEN: Format = Format { option: 4, ..BEFORE };

fn json(s: &str) -> String {
    serde_json::to_string(s).unwrap()
}

/// 값 양끝의 따옴표는 표기일 뿐이라 벗깁니다.
fn plain(v: &str) -> &str {
    v.strip_prefix('"').and_then(|x| x.strip_suffix('"')).unwrap_or(v)
}

/// 본문과 선택지별 코드의 바이트 범위 (본문 안의 위치)
fn body_of(kind: usize, title: &str, facts: &[(String, String)], options: &[&str]) -> (String, Vec<(usize, usize)>) {
    let mut t = String::new();
    let lines = |sep: &str, indent: &str| -> String {
        facts.iter().map(|(k, v)| format!("{indent}{}: {}", json(k), json(plain(v)))).collect::<Vec<_>>().join(sep)
    };
    let mut spans = Vec::new();
    let list = |t: &mut String, spans: &mut Vec<(usize, usize)>, indent: &str, between: &str| {
        for (i, o) in options.iter().enumerate() {
            if i > 0 {
                t.push_str(between);
            }
            t.push_str(indent);
            t.push('"');
            spans.push((t.len(), t.len() + o.len()));
            t.push_str(o);
            t.push('"');
        }
    };
    match kind {
        0 => {
            t.push_str(&format!("{{\n  \"task\": {},\n  \"facts\": {{\n{}\n  }},\n  \"zoro_jev_head_options\": [\n", json(title), lines(",\n", "    ")));
            list(&mut t, &mut spans, "    ", ",\n");
            t.push_str("\n  ]\n}");
        }
        1 => {
            t.push_str(&format!("{{\n{},\n  \"zoro_jev_head_options\": [\n", lines(",\n", "  ")));
            list(&mut t, &mut spans, "    ", ",\n");
            t.push_str("\n  ]\n}");
        }
        2 => {
            t.push_str(&format!("{{\n  \"context\": {{\n{}\n  }},\n  \"options\": [\n", lines(",\n", "    ")));
            list(&mut t, &mut spans, "    ", ",\n");
            t.push_str("\n  ]\n}");
        }
        3 => {
            t.push_str("Context:\n");
            for (k, v) in facts {
                t.push_str(&format!("- {k}: {}\n", plain(v)));
            }
            t.push_str("Options:\n");
            list(&mut t, &mut spans, "- ", ",\n");
        }
        4 => {
            t.push_str(&format!("{{\n  \"task\": {},\n  \"facts\": {{{}}},\n  \"zoro_jev_head_options\": [", json(title), lines(", ", "")));
            list(&mut t, &mut spans, "", ", ");
            t.push_str("]\n}");
        }
        _ => {
            // 따옴표 없이 줄마다 코드 하나. 닫는따옴표 자리에는 줄바꿈이 옵니다.
            if kind == 6 {
                t.push_str(&format!("Task: {title}\n"));
            }
            t.push_str("Context:\n");
            for (k, v) in facts {
                t.push_str(&format!("- {k}: {}\n", plain(v)));
            }
            t.push_str("Options:\n");
            for o in options {
                t.push_str("- ");
                spans.push((t.len(), t.len() + o.len()));
                t.push_str(o);
                t.push('\n');
            }
        }
    }
    (t, spans)
}

pub struct Rendered {
    pub text: String,
    pub spans: Vec<(usize, usize)>,
    /// 질의 후보의 바이트 위치
    pub queries: [usize; 4],
}

pub fn render(wrapper: usize, body: usize, title: &str, facts: &[(String, String)], options: &[&str]) -> Rendered {
    let (b, spans) = body_of(body, title, facts, options);
    let system = matches!(wrapper, 0 | 2 | 3);
    let chat = wrapper != 6;
    let mut text = String::new();
    if system {
        text.push_str(&format!("<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"));
    }
    if chat {
        text.push_str("<|im_start|>user\n");
    }
    let start = text.len();
    text.push_str(&b);
    let close = text.len() - 1;
    let mut user_end = text.len() - 1;
    let mut role = text.len() - 1;
    if chat {
        user_end = text.len();
        text.push_str("<|im_end|>\n<|im_start|>assistant\n");
        role = text.len() - 2;
    }
    match wrapper {
        2 | 4 => text.push_str("<think>\n"),
        3 | 5 => text.push_str("<think>\n\n</think>\n\n"),
        _ => {}
    }
    let last = text.len() - 1;
    Rendered { spans: spans.iter().map(|(s, e)| (s + start, e + start)).collect(), queries: [last, user_end, role, close], text }
}

/// 조합 하나
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Format {
    pub wrapper: usize,
    pub body: usize,
    pub query: usize,
    pub option: usize,
    pub norm: usize,
}

impl Format {
    /// `틀,본문,질의,선택지,정규화` 번호 (예 `1,0,1,4,1`)
    pub fn parse(s: &str) -> Option<Self> {
        let v: Vec<usize> = s.split(',').map(|x| x.trim().parse().ok()).collect::<Option<_>>()?;
        match v[..] {
            [wrapper, body, query, option, norm]
                if wrapper < WRAPPERS.len() && body < BODIES.len() && query < QUERIES.len() && option < OPTION_POS.len() && norm < NORMS.len() =>
            {
                Some(Self { wrapper, body, query, option, norm })
            }
            _ => None,
        }
    }

    pub fn label(&self) -> String {
        format!("{} · {} · 질의 {} · 선택지 {} · {}", WRAPPERS[self.wrapper], BODIES[self.body], QUERIES[self.query], OPTION_POS[self.option], NORMS[self.norm])
    }

    /// 코드의 바이트 범위에서 선택지 위치를 고릅니다. 평균은 위치 하나로 나타낼 수 없습니다.
    pub fn option_byte(&self, (s, e): (usize, usize)) -> usize {
        match self.option {
            0 => e - 1,
            1 => s,
            3 => s - 1,
            4 => e,
            other => panic!("선택지 위치 {} 는 위치 하나가 아닙니다", OPTION_POS[other]),
        }
    }

    pub fn prompt(&self, title: &str, facts: &[(String, String)], options: &[&str]) -> PointerPrompt {
        let r = render(self.wrapper, self.body, title, facts, options);
        let option_bytes = r.spans.iter().map(|&s| self.option_byte(s)).collect();
        PointerPrompt { query_byte: r.queries[self.query], option_bytes, text: r.text }
    }
}
