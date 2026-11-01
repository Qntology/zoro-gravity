//! 패치 경쟁 (NMS): 칸마다 종류끼리 겨뤄 주인을 정합니다. logis-center 의 `nms_arena` 와 같은 방식이고 코드는 직접 구현했습니다.
//!
//! 1. 칸마다 1위와 2위 종류를 찾습니다. 점수가 0 이하인 종류는 그 칸을 주장할 수 없습니다.
//! 2. 두 종류 이상이 다투는 칸의 차이 중앙값 × 0.5 가 확정 기준입니다.
//! 3. 기준을 넘는 칸을 하나도 얻지 못한 종류는 탈락하고, 그 칸을 남은 종류끼리 다시 겨룹니다 (최대 4라운드).
//!
//! 칸 하나의 판단은 잡음이 크지만, 조각이 덮는 칸의 주인을 세면 평균 점수보다 튀는 값에 덜 흔들립니다.

/// 재경쟁 최대 횟수
pub const MAX_ROUNDS: usize = 4;
const MARGIN_RATIO: f32 = 0.5;
/// 이보다 다투는 칸이 적으면 기준을 세우지 않습니다.
const MIN_CONTESTED: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CellVerdict {
    pub owner: Option<usize>,
    /// 1위와 2위의 차이. 혼자 주장하면 자기 점수입니다.
    pub margin: f32,
    pub contested: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Arena {
    pub cells: Vec<CellVerdict>,
    /// 끝까지 남은 종류
    pub alive: Vec<bool>,
    pub gate: f32,
    pub rounds: usize,
}

impl Arena {
    /// 칸이 이 종류의 확정 영토인가
    pub fn owns(&self, kind: usize, cell: usize) -> bool {
        let v = &self.cells[cell];
        v.owner == Some(kind) && v.margin >= self.gate
    }
}

/// `scores[종류][칸]`. `active` 는 겨룰 칸입니다 (잉크가 있는 칸).
fn compete(scores: &[&[f32]], alive: &[bool], active: &[bool]) -> Vec<CellVerdict> {
    (0..active.len())
        .map(|i| {
            let (mut best, mut second) = (None::<(usize, f32)>, None::<f32>);
            if active[i] {
                for (k, s) in scores.iter().enumerate() {
                    let v = s[i];
                    if !alive[k] || !v.is_finite() || v <= 0.0 {
                        continue;
                    }
                    match best {
                        Some((_, b)) if v <= b => second = Some(second.map_or(v, |x: f32| x.max(v))),
                        Some((_, b)) => {
                            second = Some(b);
                            best = Some((k, v));
                        }
                        None => best = Some((k, v)),
                    }
                }
            }
            match (best, second) {
                (Some((k, b)), Some(s)) => CellVerdict { owner: Some(k), margin: b - s, contested: true },
                (Some((k, b)), None) => CellVerdict { owner: Some(k), margin: b, contested: false },
                _ => CellVerdict { owner: None, margin: 0.0, contested: false },
            }
        })
        .collect()
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

/// 경쟁을 돌립니다. `protected` 인 종류는 영토가 없어도 탈락시키지 않습니다.
pub fn run(scores: &[&[f32]], active: &[bool], protected: &[bool]) -> Arena {
    let k = scores.len();
    let mut alive = vec![true; k];
    let contested: Vec<f32> = compete(scores, &alive, active).iter().filter(|c| c.contested).map(|c| c.margin).collect();
    let gate = if contested.len() < MIN_CONTESTED { 0.0 } else { median(contested) * MARGIN_RATIO };

    let mut rounds = 0;
    for _ in 0..MAX_ROUNDS {
        rounds += 1;
        let cells = compete(scores, &alive, active);
        let mut firm = vec![0usize; k];
        for c in &cells {
            if let Some(o) = c.owner.filter(|_| c.margin >= gate) {
                firm[o] += 1;
            }
        }
        let losers: Vec<usize> = (0..k).filter(|&i| alive[i] && firm[i] == 0 && !protected[i]).collect();
        if losers.is_empty() {
            break;
        }
        losers.iter().for_each(|&i| alive[i] = false);
    }
    Arena { cells: compete(scores, &alive, active), alive, gate, rounds }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_cell_goes_to_its_strongest_kind() {
        let a = [0.9f32, 0.1, 0.5, 0.0];
        let b = [0.2f32, 0.8, 0.5, -0.1];
        let got = run(&[&a, &b], &[true; 4], &[false, false]);
        assert_eq!(got.cells[0].owner, Some(0));
        assert_eq!(got.cells[1].owner, Some(1));
        assert!((got.cells[0].margin - 0.7).abs() < 1e-6);
        assert_eq!(got.cells[3].owner, None, "점수가 0 이하면 주인이 없습니다");
    }

    #[test]
    fn cells_without_ink_are_not_contested() {
        let a = [0.9f32, 0.9];
        let got = run(&[&a], &[true, false], &[false]);
        assert_eq!(got.cells[0].owner, Some(0));
        assert_eq!(got.cells[1].owner, None);
    }

    #[test]
    fn a_lone_claim_keeps_its_own_score_as_the_margin() {
        let a = [0.4f32, 0.0];
        let b = [0.0f32, 0.3];
        let got = run(&[&a, &b], &[true, true], &[false, false]);
        assert_eq!(got.cells[0].margin, 0.4);
        assert!(!got.cells[0].contested);
    }

    /// 두 종류가 나눠 갖는 칸에서 근소한 2등만 하는 종류는 탈락하고, 그 칸은 남은 종류끼리 다시 겨룹니다.
    #[test]
    fn a_kind_that_never_wins_firmly_is_eliminated() {
        let n = 8;
        let strong: Vec<f32> = (0..n).map(|i| if i < 4 { 0.9 } else { 0.1 }).collect();
        let rival: Vec<f32> = (0..n).map(|i| if i < 4 { 0.1 } else { 0.9 }).collect();
        let weak: Vec<f32> = vec![0.05; n];
        let got = run(&[&strong, &rival, &weak], &vec![true; n], &[false, false, false]);
        assert_eq!(got.alive, vec![true, true, false]);
        assert!(got.rounds >= 2);
        assert!(got.owns(0, 0) && got.owns(1, 7));
        assert!(!got.owns(1, 0));
    }

    #[test]
    fn a_protected_kind_survives_without_territory() {
        let n = 8;
        let strong: Vec<f32> = (0..n).map(|i| if i < 4 { 0.9 } else { 0.1 }).collect();
        let rival: Vec<f32> = (0..n).map(|i| if i < 4 { 0.1 } else { 0.9 }).collect();
        let weak: Vec<f32> = vec![0.05; n];
        let got = run(&[&strong, &rival, &weak], &vec![true; n], &[false, false, true]);
        assert!(got.alive[2]);
    }

    #[test]
    fn too_few_contested_cells_set_no_gate() {
        let a = [0.5f32, 0.4];
        let b = [0.1f32, 0.3];
        assert_eq!(run(&[&a, &b], &[true, true], &[false, false]).gate, 0.0);
    }
}
