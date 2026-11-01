//! Gate 2: 서류 간 정합성 교차대조 (명세 Part 3).
//!
//! 진실 우선순위: 금융기관 인쇄 원장(통장·이체내역) > 날인된 계약서 > 민원인 수기 서류.
//! 차액이 생기면 소명 종류에 따라 정합 / 모순 기각 / 증빙 요구 / 공무원 확인으로 나눕니다.
//! 금액 계산은 정수 연산(checked)으로만 합니다. 모델은 숫자를 계산하지 않습니다.

use crate::case::{BankFacts, Excuse, LeaseFacts};
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
}
