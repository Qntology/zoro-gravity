//! 비식별과 누출 방지.
//!
//! - 주민등록번호·사업자번호는 솔트 SHA-256 해시로만 관계망에 올립니다 (명세 Table 3). 같은 사람 매칭만 가능하고 역추적은 안 됩니다.
//! - 민원인 교부용 문서에는 내부 감사 용어가 들어가면 안 됩니다 (명세 Gate 4, Case 4-B).
//! - 제3자 실명 열람은 위험도 70% 이상일 때만 허용합니다 (명세 Case 3-C).

use sha2::{Digest, Sha256};

/// 제3자 실명 열람 허용 기준 위험도 (명세: 70% 이상)
pub const UNMASK_RISK_THRESHOLD: f32 = 0.70;

/// 민원인 교부용 문서에 들어가면 안 되는 내부 용어 (명세 Gate 4)
pub const INTERNAL_TERMS: &[&str] = &["LatticeDB", "위장", "96.4%", "G3_", "jev_head", "Zoro", "마스킹"];

/// 솔트 SHA-256 노드 해시 (소문자 hex). 솔트는 저장소 밖에 보관해야 합니다.
pub fn node_hash(salt: &[u8], raw_identifier: &str) -> String {
    let mut h = Sha256::new();
    h.update(salt);
    h.update(raw_identifier.as_bytes());
    hex::encode(h.finalize())
}

/// 이름 마스킹. 세 글자 `박영희` → `박*희`, 두 글자 `박영` → `박*`.
pub fn mask_name(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    match chars.len() {
        0 => String::new(),
        1 => "*".to_string(),
        2 => format!("{}*", chars[0]),
        n => format!("{}{}{}", chars[0], "*".repeat(n - 2), chars[n - 1]),
    }
}

/// 문서 본문에서 찾은 누출 용어. 비어 있으면 안전합니다.
pub fn find_leaks<'a, S: AsRef<str>>(text: &str, terms: &'a [S]) -> Vec<&'a str> {
    terms
        .iter()
        .map(AsRef::as_ref)
        .filter(|t| !t.is_empty() && text.contains(t))
        .collect()
}

/// 전화번호 · 주민등록번호 · 계좌번호처럼 숫자가 일곱 개 이상 이어지는 부분을 가립니다 (`-` 는 이어진 것으로 봅니다).
/// 모델에 글을 넘기기 전에 씁니다. 금액의 쉼표는 끊어진 것으로 보므로 `500,000원` 은 남습니다.
pub fn mask_long_numbers(text: &str) -> String {
    const MIN_DIGITS: usize = 7;
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let start = i;
            let mut digits = 0;
            while i < chars.len() && (chars[i].is_ascii_digit() || (chars[i] == '-' && chars.get(i + 1).is_some_and(char::is_ascii_digit))) {
                digits += chars[i].is_ascii_digit() as usize;
                i += 1;
            }
            if digits >= MIN_DIGITS {
                out.push_str("***");
            } else {
                out.extend(&chars[start..i]);
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// 위험도 70% 이상일 때만 제3자 실명 열람을 허용합니다.
pub fn unmask_allowed(risk: f32) -> bool {
    risk >= UNMASK_RISK_THRESHOLD
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_numbers_are_masked_and_amounts_stay() {
        assert_eq!(mask_long_numbers("전화 010-1234-5678 입니다"), "전화 *** 입니다");
        assert_eq!(mask_long_numbers("900101-1234567"), "***");
        assert_eq!(mask_long_numbers("계좌 110123991023"), "계좌 ***");
        assert_eq!(mask_long_numbers("월세 500,000원, 2026-10-31 접수"), "월세 500,000원, *** 접수");
        assert_eq!(mask_long_numbers("7시부터 20시까지, 3-1번지"), "7시부터 20시까지, 3-1번지");
    }

    #[test]
    fn hash_is_salted_and_stable() {
        let a = node_hash(b"salt-1", "900101-1234567");
        assert_eq!(a, node_hash(b"salt-1", "900101-1234567"));
        assert_ne!(a, node_hash(b"salt-2", "900101-1234567"));
        assert_eq!(a.len(), 64);
        assert!(!a.contains("900101"));
    }

    #[test]
    fn masks_names_like_the_spec() {
        assert_eq!(mask_name("박영희"), "박*희");
        assert_eq!(mask_name("박영"), "박*");
        assert_eq!(mask_name("김"), "*");
    }

    #[test]
    fn detects_internal_terms_in_citizen_text() {
        let text = "LatticeDB 추적 결과 위장계약 96.4% 확실";
        let found = find_leaks(text, INTERNAL_TERMS);
        assert!(found.contains(&"LatticeDB"));
        assert!(found.contains(&"위장"));
        assert!(found.contains(&"96.4%"));
        assert!(find_leaks("임대차계약서와 이체내역 차액을 보완해 주세요", INTERNAL_TERMS).is_empty());
    }

    #[test]
    fn unmask_threshold_is_seventy_percent() {
        assert!(!unmask_allowed(0.015));
        assert!(!unmask_allowed(0.699));
        assert!(unmask_allowed(0.70));
        assert!(unmask_allowed(0.964));
    }
}
