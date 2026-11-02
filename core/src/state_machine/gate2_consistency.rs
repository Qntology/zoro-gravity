//! Gate 2: 서류 간 정합성 교차대조 (명세 Part 3).
//!
//! 진실 우선순위: 금융기관 인쇄 원장(통장·이체내역) > 날인된 계약서 > 민원인 수기 서류.
//! 차액이 생기면 소명 종류에 따라 정합 / 모순 기각 / 증빙 요구 / 공무원 확인으로 나눕니다.
//! 금액 계산은 정수 연산(checked)으로만 합니다. 모델은 숫자를 계산하지 않습니다.

use crate::case::{BankFacts, Excuse, LeaseFacts, Receipt, SubsidyFacts};
use crate::verdict::Verdict;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccountLock {
    /// 확정된 계좌번호 (인쇄 원장 값)
    pub value: String,
    /// 수기 기재가 인쇄 값과 달라서 덮어썼는지
    pub overwrote_handwritten: bool,
}

/// 계좌번호는 인쇄된 통장 원장 값으로 고정합니다 (명세 Case 2-A).
pub fn lock_account(bank: &BankFacts) -> AccountLock {
    AccountLock {
        value: bank.printed_account.clone(),
        overwrote_handwritten: bank.handwritten_account != bank.printed_account,
    }
}

/// 월세 차액 = 최근 이체액 − 계약 월세. 음수면 덜 이체된 상태입니다.
pub fn rent_gap(lease: &LeaseFacts) -> Option<i64> {
    lease.latest_transfer.checked_sub(lease.contract_rent)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Consistency {
    /// 정합. `note` 가 있으면 정정한 내용을 함께 남깁니다.
    Consistent { note: Option<String> },
    /// 민원인 소명이 서류와 물리적으로 모순됩니다. 그 선택지를 잠급니다 (명세 Case 2-B).
    ClaimContradicted { verdict: Verdict, reason: String },
    /// 증빙 스캔을 요구합니다 (키오스크 재요청).
    EvidenceRequested { verdict: Verdict, reason: String },
    /// 공무원이 증빙을 보고 판단해야 합니다.
    OfficerReview { reason: String },
}

pub fn check_rent(lease: &LeaseFacts, excuse: Option<&Excuse>) -> Consistency {
    let Some(gap) = rent_gap(lease) else {
        return Consistency::OfficerReview {
            reason: "금액 차이가 계산 범위를 넘었습니다".into(),
        };
    };
    let application_differs = lease.applied_rent != lease.contract_rent;

    if gap == 0 {
        return Consistency::Consistent {
            note: application_differs
                .then(|| format!("신청서 금액을 인쇄 계약서 금액 {}원으로 정정", lease.contract_rent)),
        };
    }

    match excuse {
        None => Consistency::OfficerReview {
            reason: format!("이체액과 계약 월세가 {gap}원 차이 납니다. 소명이 필요합니다"),
        },
        Some(Excuse::TypoInApplication) if !application_differs => Consistency::ClaimContradicted {
            verdict: Verdict::G2RejectContradictionLock,
            reason: "인쇄된 계약서에도 신청서와 같은 금액이 적혀 있어 신청서 오타가 아닙니다".into(),
        },
        Some(Excuse::TypoInApplication) => Consistency::OfficerReview {
            reason: "신청서 오타는 인정되지만 이체 차액은 따로 소명해야 합니다".into(),
        },
        Some(Excuse::RepairDeduction { has_receipt: false }) => Consistency::EvidenceRequested {
            verdict: Verdict::G2RequireRepairReceipt,
            reason: "수리비 영수증을 스캔해 주세요".into(),
        },
        Some(Excuse::RepairDeduction { has_receipt: true }) => Consistency::OfficerReview {
            reason: "수리비 영수증을 확인하세요".into(),
        },
        Some(Excuse::CashPaidDirectly) => Consistency::OfficerReview {
            reason: "현금 지급 주장입니다. 현금영수증 등 증빙을 확인하세요".into(),
        },
        Some(Excuse::FreeText(_)) => Consistency::OfficerReview {
            reason: "자유 서술 소명입니다. 공무원이 내용을 확인하세요".into(),
        },
    }
}

/// 공무원이 인정 금액을 키보드로 고칠 때의 가드 (명세 Case 2-D).
/// 증빙된 금액 이하로 내리는 것은 허용합니다. 올리려면 그 차이만큼 증빙 스캔이 있어야 합니다.
pub fn check_manual_amount(verified_sum: i64, requested: i64, evidence_sum: i64) -> Result<(), Verdict> {
    if requested <= verified_sum {
        return Ok(());
    }
    let headroom = verified_sum
        .checked_add(evidence_sum)
        .ok_or(Verdict::G2BlockUnprovenEdit)?;
    if requested <= headroom {
        Ok(())
    } else {
        Err(Verdict::G2BlockUnprovenEdit)
    }
}

/// 입찰 회피를 보는 금액 기준 (명세 CAT_02: 50만 원)
pub const BID_LIMIT: i64 = 500_000;

/// 지원금으로 인정하지 않는 품목 낱말 (명세 CAT_02: 주류 등)
const BARRED_ITEMS: [&str; 3] = ["주류", "유흥", "노래방"];

/// 영수증 대조 결과. 합계는 규칙이 정수로 계산합니다.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReceiptReview {
    /// 중복과 비인정 품목을 뺀 인정 합계
    pub verified_sum: i64,
    /// 같은 승인번호로 두 번 들어온 영수증
    pub duplicates: Vec<String>,
    /// 인정하지 않는 품목의 승인번호
    pub barred: Vec<String>,
    /// 같은 날 같은 거래처로 나눠 결제해 입찰 기준을 넘긴 거래처
    pub splits: Vec<String>,
    /// 신청 금액 - 인정 합계. 양수면 증빙이 모자랍니다.
    pub gap: i64,
}

/// 영수증을 대조합니다. 중복은 한 장으로 합치고, 비인정 품목은 빼고, 분할 결제를 찾습니다 (명세 Part 6 CAT_02).
pub fn review_receipts(facts: &SubsidyFacts) -> ReceiptReview {
    let mut seen: Vec<&str> = Vec::new();
    let mut r = ReceiptReview::default();
    let mut unique: Vec<&Receipt> = Vec::new();
    for receipt in &facts.receipts {
        if seen.contains(&receipt.approval.as_str()) {
            r.duplicates.push(receipt.approval.clone());
            continue;
        }
        seen.push(&receipt.approval);
        unique.push(receipt);
        let barred = facts.barred_vendors.contains(&receipt.vendor) || BARRED_ITEMS.iter().any(|w| receipt.item.contains(w));
        if barred {
            r.barred.push(receipt.approval.clone());
        } else {
            r.verified_sum = r.verified_sum.saturating_add(receipt.amount);
        }
    }
    for receipt in &unique {
        let same_day: Vec<&&Receipt> = unique.iter().filter(|o| o.vendor == receipt.vendor && o.date == receipt.date).collect();
        let total: i64 = same_day.iter().map(|o| o.amount).sum();
        if same_day.len() >= 2 && total >= BID_LIMIT && same_day.iter().all(|o| o.amount < BID_LIMIT) && !r.splits.contains(&receipt.vendor) {
            r.splits.push(receipt.vendor.clone());
        }
    }
    r.gap = facts.claimed_total.saturating_sub(r.verified_sum);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lease(applied: i64, contract: i64, transfer: i64) -> LeaseFacts {
        LeaseFacts { applied_rent: applied, contract_rent: contract, latest_transfer: transfer }
    }

    #[test]
    fn printed_bankbook_wins_over_handwriting() {
        // 명세 Case 2-A: 수기 ...8, 통장 인쇄 ...3
        let lock = lock_account(&BankFacts {
            handwritten_account: "110-123-991028".into(),
            printed_account: "110-123-991023".into(),
        });
        assert_eq!(lock.value, "110-123-991023");
        assert!(lock.overwrote_handwritten);
    }

    #[test]
    fn typo_claim_is_contradicted_when_the_printed_contract_agrees_with_the_application() {
        // 명세 Case 2-B: 신청서 50만, 계약서 50만, 이체 30만 → "신청서 오타" 주장은 모순
        let c = check_rent(&lease(500_000, 500_000, 300_000), Some(&Excuse::TypoInApplication));
        assert!(matches!(c, Consistency::ClaimContradicted { verdict: Verdict::G2RejectContradictionLock, .. }));
    }

    #[test]
    fn typo_is_accepted_when_only_the_application_is_wrong() {
        let c = check_rent(&lease(600_000, 500_000, 500_000), Some(&Excuse::TypoInApplication));
        assert!(matches!(c, Consistency::Consistent { note: Some(_) }));
    }

    #[test]
    fn cash_claim_goes_to_the_officer_and_repair_claim_asks_for_a_receipt() {
        assert!(matches!(
            check_rent(&lease(500_000, 500_000, 300_000), Some(&Excuse::CashPaidDirectly)),
            Consistency::OfficerReview { .. }
        ));
        assert!(matches!(
            check_rent(&lease(500_000, 500_000, 300_000), Some(&Excuse::RepairDeduction { has_receipt: false })),
            Consistency::EvidenceRequested { verdict: Verdict::G2RequireRepairReceipt, .. }
        ));
    }

    #[test]
    fn unexplained_gap_is_never_silently_accepted() {
        assert!(matches!(
            check_rent(&lease(500_000, 500_000, 300_000), None),
            Consistency::OfficerReview { .. }
        ));
    }

    #[test]
    fn manual_amount_edit_needs_evidence_to_go_up() {
        // 명세 Case 2-D: 증빙 820,000 → 1,000,000 로 올리려면 180,000 증빙이 필요
        assert_eq!(check_manual_amount(820_000, 1_000_000, 0), Err(Verdict::G2BlockUnprovenEdit));
        assert_eq!(check_manual_amount(820_000, 1_000_000, 180_000), Ok(()));
        assert_eq!(check_manual_amount(820_000, 700_000, 0), Ok(()), "내리는 것은 허용");
    }

    fn receipt(approval: &str, vendor: &str, date: &str, amount: i64, item: &str) -> Receipt {
        Receipt { approval: approval.into(), vendor: vendor.into(), date: date.into(), time: "10:00".into(), amount, item: item.into() }
    }

    #[test]
    fn receipts_merge_duplicates_drop_barred_items_and_find_splits() {
        let facts = SubsidyFacts {
            claimed_total: 1_000_000,
            receipts: vec![
                receipt("1", "태성기획", "2026-10-14", 480_000, "인쇄"),
                receipt("2", "태성기획", "2026-10-14", 340_000, "인쇄"),
                receipt("2", "태성기획", "2026-10-14", 340_000, "인쇄"),
                receipt("3", "달빛주점", "2026-10-15", 100_000, "회식"),
            ],
            barred_vendors: vec!["달빛주점".into()],
            ..SubsidyFacts::default()
        };
        let r = review_receipts(&facts);
        assert_eq!(r.duplicates, vec!["2".to_string()]);
        assert_eq!(r.barred, vec!["3".to_string()]);
        assert_eq!(r.verified_sum, 820_000, "명세 Case 2-D 의 인정 합계");
        assert_eq!(r.gap, 180_000);
        assert_eq!(r.splits, vec!["태성기획".to_string()]);
    }

    #[test]
    fn one_large_payment_or_small_payments_below_the_limit_are_not_splits() {
        let facts = SubsidyFacts {
            claimed_total: 650_000,
            receipts: vec![receipt("1", "우리인쇄", "2026-10-01", 600_000, "인쇄"), receipt("2", "대한문구", "2026-10-01", 20_000, "문구"), receipt("3", "대한문구", "2026-10-01", 30_000, "문구")],
            ..SubsidyFacts::default()
        };
        let r = review_receipts(&facts);
        assert!(r.splits.is_empty(), "{:?}", r.splits);
        assert_eq!(r.gap, 0);
    }
}
