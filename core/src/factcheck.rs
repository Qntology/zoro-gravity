//! 모델이 쓴 글의 숫자를 원문과 대조합니다.
//!
//! 생성 모델은 그럴듯한 숫자를 지어냅니다. 실측에서 원문의 "저녁 여덟 시"를 "18시"로 바꿨습니다.
//! 그래서 글에 나온 숫자는 모두 원문에도 있어야 합니다. 표기가 달라도 같은 값이면 같은 숫자로 봅니다
//! (`20만 원` = `200,000원`, `일곱 시` = `7시`, `이십만 원` = `200,000원`).
//! 단위는 보지 않고 값만 봅니다. 놓치는 것보다 더 많이 걸러 내는 쪽을 택했습니다. 걸러진 글은 공무원이 봅니다.

use std::collections::BTreeSet;

/// 소수점 아래를 정수로 다루려고 값에 곱하는 수
const SCALE: f64 = 1000.0;

/// 글에 나온 숫자 하나. `values` 중 하나라도 원문에 있으면 맞는 것으로 봅니다 (`저녁 8시` 는 8 이자 20 입니다).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mention {
    pub text: String,
    pub values: Vec<i64>,
}

fn is_hangul(c: char) -> bool {
    ('가'..='힣').contains(&c)
}

fn sino_digit(c: char) -> Option<i64> {
    "영일이삼사오육칠팔구".chars().position(|d| d == c).map(|i| i as i64)
}

fn small_unit(c: char) -> Option<i64> {
    match c {
        '십' => Some(10),
        '백' => Some(100),
        '천' => Some(1000),
        _ => None,
    }
}

fn big_unit(c: char) -> Option<i64> {
    match c {
        '만' => Some(10_000),
        '억' => Some(100_000_000),
        '조' => Some(1_000_000_000_000),
        _ => None,
    }
}

/// 한자어 수 뒤에 올 수 있는 단위의 첫 글자
const SINO_COUNTERS: &str = "원명년월일시분초회번개층호세살건장달주평배퍼프";
/// 고유어 수 뒤에 오는 단위
const NATIVE_COUNTERS: [&str; 18] = ["시간", "시", "살", "명", "개", "번", "차례", "달", "곳", "장", "분", "건", "대", "마리", "주", "해", "권", "가지"];
const TENS: [(&str, i64); 10] = [("열", 10), ("스물", 20), ("스무", 20), ("서른", 30), ("마흔", 40), ("쉰", 50), ("예순", 60), ("일흔", 70), ("여든", 80), ("아흔", 90)];
const ONES: [(&str, i64); 15] = [("하나", 1), ("한", 1), ("둘", 2), ("두", 2), ("셋", 3), ("세", 3), ("석", 3), ("넷", 4), ("네", 4), ("넉", 4), ("다섯", 5), ("여섯", 6), ("일곱", 7), ("여덟", 8), ("아홉", 9)];
/// 이 말이 앞에 있으면 12시간제의 오후입니다.
const AFTERNOON: [&str; 4] = ["오후", "저녁", "밤", "낮"];

fn starts_with(c: &[char], at: usize, word: &str) -> bool {
    word.chars().enumerate().all(|(k, w)| c.get(at + k) == Some(&w))
}

/// 공백을 건너뛴 다음 위치
fn skip_space(c: &[char], mut i: usize) -> usize {
    while c.get(i).is_some_and(|x| x.is_whitespace()) {
        i += 1;
    }
    i
}

/// 시각이면 `저녁 8시` 처럼 오후 표현을 살펴 값을 더합니다.
fn with_clock(c: &[char], start: usize, end: usize, value: i64) -> Vec<i64> {
    let mut values = vec![value * SCALE as i64];
    let next = skip_space(c, end);
    let clock = c.get(next) == Some(&'시') && c.get(next + 1) != Some(&'간');
    if clock && (1..=12).contains(&value) {
        let before: String = c[start.saturating_sub(6)..start].iter().collect();
        if AFTERNOON.iter().any(|w| before.contains(w)) {
            values.push(((value % 12) + 12) * SCALE as i64);
        }
    }
    values
}

/// 아라비아 숫자 한 묶음 (`500,000`, `20`, `119.8`). `(값, 다음 위치)`
fn read_arabic(c: &[char], start: usize) -> (f64, usize) {
    let mut i = start;
    let mut digits = String::new();
    while let Some(x) = c.get(i) {
        if x.is_ascii_digit() {
            digits.push(*x);
            i += 1;
        } else if *x == ',' && c.len() > i + 3 && c[i + 1..=i + 3].iter().all(char::is_ascii_digit) && !c.get(i + 4).is_some_and(char::is_ascii_digit) {
            i += 1;
        } else {
            break;
        }
    }
    let mut value: f64 = digits.parse().unwrap_or(0.0);
    // `2026.03.01` 같은 날짜는 소수가 아닙니다. 앞에 `숫자.` 가 있으면 날짜의 뒷부분입니다.
    let in_date = start >= 2 && c[start - 1] == '.' && c[start - 2].is_ascii_digit();
    if c.get(i) == Some(&'.') && c.get(i + 1).is_some_and(char::is_ascii_digit) && digits.len() < 4 && !in_date {
        let mut j = i + 1;
        let mut frac = String::new();
        while let Some(x) = c.get(j).filter(|x| x.is_ascii_digit()) {
            frac.push(*x);
            j += 1;
        }
        if c.get(j) != Some(&'.') {
            value = format!("{digits}.{frac}").parse().unwrap_or(value);
            i = j;
        }
    }
    (value, i)
}

/// 아라비아 숫자에 붙은 큰 단위까지 읽습니다 (`20만`, `1억 2천만`, `3천`).
fn read_number(c: &[char], start: usize) -> (i64, usize) {
    let mut total = 0f64;
    let mut i = start;
    loop {
        let (mut value, next) = read_arabic(c, i);
        i = next;
        if let Some(m) = c.get(i).copied().and_then(small_unit) {
            value *= m as f64;
            i += 1;
        }
        match c.get(i).copied().and_then(big_unit) {
            Some(m) => {
                total += value * m as f64;
                i += 1;
                let after = skip_space(c, i);
                if c.get(after).is_some_and(char::is_ascii_digit) {
                    i = after;
                    continue;
                }
                break;
            }
            None => {
                total += value;
                break;
            }
        }
    }
    ((total * SCALE).round() as i64, i)
}

/// 한자어 수 (`이십만`, `삼백오십`, `십오`). 큰 자리 단위가 하나도 없으면 수로 보지 않습니다 (`일이`, `이`, `사` 같은 낱말 때문입니다).
fn read_sino(c: &[char], start: usize) -> Option<(i64, usize)> {
    if start > 0 && is_hangul(c[start - 1]) {
        return None;
    }
    let (mut total, mut section, mut digit) = (0i64, 0i64, None::<i64>);
    let mut i = start;
    let mut has_unit = false;
    while let Some(&x) = c.get(i) {
        if let Some(d) = sino_digit(x) {
            if digit.is_some() {
                break;
            }
            digit = Some(d);
        } else if let Some(m) = small_unit(x) {
            section += digit.take().unwrap_or(1) * m;
            has_unit = true;
        } else if let Some(m) = big_unit(x) {
            let part = section + digit.take().unwrap_or(0);
            total += part.max(1) * m;
            section = 0;
            has_unit = true;
        } else {
            break;
        }
        i += 1;
    }
    let follows = c.get(i).is_none_or(|n| !is_hangul(*n) || SINO_COUNTERS.contains(*n));
    (has_unit && follows).then_some((total + section + digit.unwrap_or(0), i))
}

/// 고유어 수 (`일곱 시`, `두 차례`, `스물다섯 살`). 뒤에 단위가 있어야 수로 봅니다.
fn read_native(c: &[char], start: usize) -> Option<(i64, usize)> {
    if start > 0 && is_hangul(c[start - 1]) {
        return None;
    }
    let mut i = start;
    let mut value = 0;
    if let Some((w, v)) = TENS.iter().find(|(w, _)| starts_with(c, i, w)) {
        value += v;
        i += w.chars().count();
    }
    if let Some((w, v)) = ONES.iter().find(|(w, _)| starts_with(c, i, w)) {
        value += v;
        i += w.chars().count();
    }
    if value == 0 {
        return None;
    }
    let next = skip_space(c, i);
    NATIVE_COUNTERS.iter().any(|w| starts_with(c, next, w)).then_some((value, i))
}

/// 글에 나온 숫자를 모두 찾습니다.
pub fn mentions(text: &str) -> Vec<Mention> {
    let c: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    let push = |out: &mut Vec<Mention>, start: usize, end: usize, values: Vec<i64>| {
        out.push(Mention { text: c[start..end].iter().collect(), values });
    };
    while i < c.len() {
        if c[i].is_ascii_digit() {
            let (value, end) = read_number(&c, i);
            let values = if value % SCALE as i64 == 0 { with_clock(&c, i, end, value / SCALE as i64) } else { vec![value] };
            push(&mut out, i, end, values);
            i = end;
        } else if let Some((value, end)) = read_native(&c, i) {
            push(&mut out, i, end, with_clock(&c, i, end, value));
            i = end;
        } else if let Some((value, end)) = read_sino(&c, i) {
            push(&mut out, i, end, with_clock(&c, i, end, value));
            i = end;
        } else {
            i += 1;
        }
    }
    out
}

/// 줄머리의 번호(`1.`, `2)`, `①`)는 내용이 아니므로 뺍니다.
fn strip_numbering(text: &str) -> String {
    text.lines()
        .map(|line| {
            let t = line.trim_start();
            let digits = t.chars().take_while(char::is_ascii_digit).count();
            let rest = &t[digits..];
            if digits > 0 && digits <= 2 && (rest.starts_with('.') || rest.starts_with(')')) {
                rest[1..].to_string()
            } else {
                t.trim_start_matches(|c| ('①'..='⑳').contains(&c)).to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// 원문에 없는 숫자. 비어 있으면 통과입니다.
pub fn unsupported(draft: &str, source: &str) -> Vec<String> {
    let known: BTreeSet<i64> = mentions(source).into_iter().flat_map(|m| m.values).collect();
    let mut bad: Vec<String> = mentions(&strip_numbering(draft))
        .into_iter()
        .filter(|m| !m.values.iter().any(|v| known.contains(v)))
        .map(|m| m.text)
        .collect();
    bad.dedup();
    bad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn values(text: &str) -> Vec<Vec<i64>> {
        mentions(text).into_iter().map(|m| m.values.iter().map(|v| v / 1000).collect()).collect()
    }

    #[test]
    fn digits_are_read_with_commas_and_big_units() {
        assert_eq!(values("월세 500,000원"), vec![vec![500_000]]);
        assert_eq!(values("20만 원"), vec![vec![200_000]]);
        assert_eq!(values("20만원"), vec![vec![200_000]]);
        assert_eq!(values("3천만 원"), vec![vec![30_000_000]]);
        assert_eq!(values("1억 2천만 원"), vec![vec![120_000_000]]);
        assert_eq!(values("3만 5천 원"), vec![vec![35_000]]);
    }

    #[test]
    fn decimals_stay_decimals_and_dates_split() {
        assert_eq!(mentions("119.8㎡")[0].values, vec![119_800]);
        assert_eq!(values("2026.03.01"), vec![vec![2026], vec![3], vec![1]]);
        assert_eq!(values("2026-10-31"), vec![vec![2026], vec![10], vec![31]]);
        assert_eq!(values("2026년 10월 31일"), vec![vec![2026], vec![10], vec![31]]);
    }

    #[test]
    fn korean_numerals_are_read() {
        assert_eq!(values("이십만 원"), vec![vec![200_000]]);
        assert_eq!(values("오십만원"), vec![vec![500_000]]);
        assert_eq!(values("삼백오십 명"), vec![vec![350]]);
        assert_eq!(values("십오일"), vec![vec![15]]);
        assert_eq!(values("두 차례"), vec![vec![2]]);
        assert_eq!(values("스물다섯 살"), vec![vec![25]]);
        assert_eq!(values("열두 시간"), vec![vec![12]]);
        assert_eq!(values("석 달째"), vec![vec![3]]);
        assert_eq!(values("넉 달"), vec![vec![4]]);
        assert!(values("좌석 대여").is_empty(), "낱말 안의 글자는 수가 아닙니다");
    }

    #[test]
    fn ordinary_words_are_not_numbers() {
        for text in ["일이 많습니다", "이 사람", "사고가 났습니다", "구청에 갔습니다", "천천히 하세요", "세상에", "한국", "두 사람은", "네 알겠습니다", "구청 민원실"] {
            assert!(mentions(text).is_empty(), "{text}: {:?}", mentions(text));
        }
    }

    #[test]
    fn evening_clock_times_are_both_twelve_hour_readings() {
        assert_eq!(values("저녁 여덟 시"), vec![vec![8, 20]]);
        assert_eq!(values("오후 3시"), vec![vec![3, 15]]);
        assert_eq!(values("아침 일곱 시"), vec![vec![7]]);
        assert_eq!(values("여덟 시간"), vec![vec![8]], "시간은 시각이 아닙니다");
    }

    #[test]
    fn the_measured_failure_is_caught() {
        // 실제 모델이 쓴 글: 원문의 저녁 여덟 시(20시)가 18시로 바뀌었습니다.
        let source = "매일 아침 일곱 시부터 저녁 여덟 시까지 공사 소음이 이어져 일상생활이 어렵습니다. 구청에 두 차례 전화를 드렸습니다.";
        let draft = "인근 공사장의 소음(매일 7시~18시)로 인한 생활 피해가 발생하고 있습니다. 구청에 두 차례 문의했습니다.";
        assert_eq!(unsupported(draft, source), vec!["18".to_string()]);
    }

    #[test]
    fn the_same_fact_written_differently_passes() {
        let source = "매일 아침 일곱 시부터 저녁 여덟 시까지 소음이 납니다. 수리비 20만 원을 냈습니다.";
        assert!(unsupported("오전 7시부터 오후 8시까지 소음이 납니다. 수리비 200,000원입니다.", source).is_empty());
        assert!(unsupported("7시부터 20시까지 소음이 납니다.", source).is_empty());
        assert!(unsupported("수리비 이십만 원을 냈습니다.", source).is_empty());
    }

    #[test]
    fn invented_amounts_and_dates_are_caught() {
        let source = "월세 500,000원, 2026-03-01 계약";
        assert_eq!(unsupported("월세 300,000원", source), vec!["300,000".to_string()]);
        assert_eq!(unsupported("2026년 4월 계약", source), vec!["4".to_string()]);
        assert!(unsupported("2026년 3월 1일 계약, 월세 50만 원", source).is_empty());
    }

    #[test]
    fn line_numbers_are_not_facts() {
        let source = "소음 민원입니다.";
        assert!(unsupported("1. 소음 민원입니다.\n2) 조치 요청\n③ 부서 배정", source).is_empty());
        assert_eq!(unsupported("1. 소음 3건", source), vec!["3".to_string()]);
    }
}
