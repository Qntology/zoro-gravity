//! 크롭용 시각 앵커: 패치 특징 → 영역 종류 점수. `scripts/train_crop_probe.py` 가 만듭니다.
//!
//! 문구 앵커와 같은 자리에 쓰는 벡터입니다. 다른 점은 문구를 인코딩한 값이 아니라 정답 서식으로 학습한 값이라는 것뿐입니다.
//! 패치 해상도에서 한국어 문구는 값과 제목을 구분하지 못해서(AUC 0.5~0.6) 이 방식을 씁니다. 근거는 `docs/document-crop.md` 에 있습니다.

use std::path::Path;

use anyhow::{bail, Context, Result};

use super::PatchGrid;

const MAGIC: &[u8; 4] = b"ZZP1";

#[derive(Debug, Clone, PartialEq)]
pub struct ProbeBank {
    pub names: Vec<String>,
    pub dim: usize,
    mean: Vec<f32>,
    /// `[종류, dim]`
    weights: Vec<f32>,
    intercept: Vec<f32>,
}

struct Reader<'a> {
    raw: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.raw.len());
        let Some(end) = end else {
            bail!("앵커 파일이 잘렸습니다 ({} 바이트 위치)", self.at);
        };
        let out = &self.raw[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<usize> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    }

    fn f32s(&mut self, n: usize) -> Result<Vec<f32>> {
        Ok(self.take(n * 4)?.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect())
    }
}

impl ProbeBank {
    pub fn from_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("앵커 파일을 읽을 수 없습니다: {}", path.display()))?;
        Self::from_bytes(&raw).with_context(|| format!("앵커 파일 형식이 맞지 않습니다: {}", path.display()))
    }

    pub fn from_bytes(raw: &[u8]) -> Result<Self> {
        let mut r = Reader { raw, at: 0 };
        if r.take(4)? != MAGIC {
            bail!("머리 표식이 다릅니다");
        }
        let (k, dim) = (r.u32()?, r.u32()?);
        let mut names = Vec::with_capacity(k);
        for _ in 0..k {
            let n = r.u32()?;
            names.push(String::from_utf8(r.take(n)?.to_vec())?);
        }
        let mean = r.f32s(dim)?;
        let weights = r.f32s(k * dim)?;
        let intercept = r.f32s(k)?;
        Ok(Self { names, dim, mean, weights, intercept })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&(self.names.len() as u32).to_le_bytes());
        out.extend_from_slice(&(self.dim as u32).to_le_bytes());
        for n in &self.names {
            out.extend_from_slice(&(n.len() as u32).to_le_bytes());
            out.extend_from_slice(n.as_bytes());
        }
        for v in self.mean.iter().chain(&self.weights).chain(&self.intercept) {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// 패치마다 종류별 점수. 결과는 `[종류][패치]` 입니다.
    pub fn score(&self, grid: &PatchGrid) -> Result<Vec<Vec<f32>>> {
        if grid.dim != self.dim {
            bail!("앵커 차원 {} 과 패치 차원 {} 이 다릅니다", self.dim, grid.dim);
        }
        let n = grid.len();
        let mut out = vec![vec![0f32; n]; self.names.len()];
        let mut centred = vec![0f32; self.dim];
        for i in 0..n {
            centred.iter_mut().zip(grid.patch(i)).zip(&self.mean).for_each(|((c, x), m)| *c = x - m);
            for (k, row) in out.iter_mut().enumerate() {
                let w = &self.weights[k * self.dim..(k + 1) * self.dim];
                row[i] = self.intercept[k] + w.iter().zip(&centred).map(|(a, b)| a * b).sum::<f32>();
            }
        }
        Ok(out)
    }
}

/// 격자를 3×3 평균으로 부드럽게 만듭니다. 패치 점수는 칸마다 잡음이 커서 이웃과 평균하면 구분력이 크게 오릅니다.
pub fn smooth(scores: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let (mut sum, mut n) = (0f32, 0f32);
            for rr in r.saturating_sub(1)..(r + 2).min(rows) {
                for cc in c.saturating_sub(1)..(c + 2).min(cols) {
                    sum += scores[rr * cols + cc];
                    n += 1.0;
                }
            }
            out[r * cols + c] = sum / n;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bank() -> ProbeBank {
        ProbeBank {
            names: vec!["a".into(), "b".into()],
            dim: 2,
            mean: vec![0.5, 0.0],
            weights: vec![1.0, 0.0, 0.0, 2.0],
            intercept: vec![0.1, -0.1],
        }
    }

    #[test]
    fn scores_are_linear_in_the_centred_features() -> Result<()> {
        let grid = PatchGrid { rows: 1, cols: 2, dim: 2, patches: vec![1.0, 0.0, 0.0, 1.0], pooled: vec![0.0, 0.0], width: 2, height: 1 };
        let s = bank().score(&grid)?;
        assert!((s[0][0] - 0.6).abs() < 1e-6 && (s[0][1] + 0.4).abs() < 1e-6);
        assert!((s[1][0] + 0.1).abs() < 1e-6 && (s[1][1] - 1.9).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn the_file_format_round_trips_and_rejects_damage() -> Result<()> {
        let b = bank();
        let raw = b.to_bytes();
        assert_eq!(ProbeBank::from_bytes(&raw)?, b);
        assert_eq!(b.index_of("b"), Some(1));
        assert!(ProbeBank::from_bytes(&raw[..raw.len() - 1]).is_err(), "잘린 파일");
        let mut bad = raw.clone();
        bad[0] = b'X';
        assert!(ProbeBank::from_bytes(&bad).is_err(), "머리 표식");
        let grid = PatchGrid { rows: 1, cols: 1, dim: 3, patches: vec![0.0; 3], pooled: vec![0.0; 3], width: 1, height: 1 };
        assert!(b.score(&grid).is_err(), "차원이 다르면 거부");
        Ok(())
    }

    #[test]
    fn smoothing_averages_the_available_neighbours() {
        let s = smooth(&[9.0, 0.0, 0.0, 0.0], 2, 2);
        assert!(s.iter().all(|v| (*v - 2.25).abs() < 1e-6), "{s:?}");
        let s = smooth(&[3.0, 0.0, 0.0], 1, 3);
        assert_eq!(s[2], 0.0, "모서리는 있는 이웃만 평균합니다");
        assert_eq!(s[0], 1.5);
    }
}
