//! siglip2 입력 전처리 (NaFlex): 가로세로 비를 유지하면서 패치 수가 상한 안에 들도록 패치 크기의 배수로 맞춥니다.
//!
//! 격자가 이미지마다 달라지므로 위치 임베딩도 그 격자로 다시 뽑습니다 ([`interpolate_grid`]).

use anyhow::{bail, Result};
use image::imageops::{resize, FilterType};
use image::RgbImage;

/// 패치 격자의 행 · 열. 이진 탐색으로 상한 안에서 가장 큰 배율을 찾습니다.
pub fn naflex_grid(height: u32, width: u32, patch: usize, max_patches: usize) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        bail!("빈 이미지입니다: {height}×{width}");
    }
    if patch == 0 || max_patches == 0 {
        bail!("패치 크기와 상한은 0 보다 커야 합니다");
    }
    const EPS: f64 = 1e-5;
    let cells = |scale: f64, size: u32| ((scale * size as f64 / patch as f64).ceil() as usize).max(1);
    let (mut lo, mut hi) = (EPS / 10.0, 100.0f64);
    while hi - lo >= EPS {
        let mid = (lo + hi) / 2.0;
        if cells(mid, height) * cells(mid, width) <= max_patches {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok((cells(lo, height), cells(lo, width)))
}

/// 전처리된 이미지. `pixels` 는 패치 수 × (p × p × 3) 이고 패치 안은 (행, 열, 채널) 순서입니다.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchInput {
    pub pixels: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    /// 원본 크기 (픽셀)
    pub width: u32,
    pub height: u32,
}

impl PatchInput {
    pub fn n_patches(&self) -> usize {
        self.rows * self.cols
    }
}

/// 이미지를 격자에 맞게 줄이고 `[-1, 1]` 로 정규화합니다. 축소는 안티에일리어싱 쌍선형(삼각 필터)입니다.
pub fn prepare(img: &RgbImage, patch: usize, max_patches: usize) -> Result<PatchInput> {
    let (height, width) = img.dimensions_hw();
    let (rows, cols) = naflex_grid(height, width, patch, max_patches)?;
    let (rw, rh) = ((cols * patch) as u32, (rows * patch) as u32);
    let resized = resize(img, rw, rh, FilterType::Triangle);
    Ok(PatchInput { pixels: patchify(&resized, patch), rows, cols, width, height })
}

trait HeightWidth {
    fn dimensions_hw(&self) -> (u32, u32);
}

impl HeightWidth for RgbImage {
    fn dimensions_hw(&self) -> (u32, u32) {
        (self.height(), self.width())
    }
}

fn patchify(img: &RgbImage, patch: usize) -> Vec<f32> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.as_raw();
    let mut out = Vec::with_capacity(w * h * 3);
    for r in 0..h / patch {
        for c in 0..w / patch {
            for y in r * patch..(r + 1) * patch {
                for x in c * patch..(c + 1) * patch {
                    for ch in 0..3 {
                        out.push(raw[(y * w + x) * 3 + ch] as f32 / 255.0 * 2.0 - 1.0);
                    }
                }
            }
        }
    }
    out
}

/// 한 축의 안티에일리어싱 쌍선형 가중치 `(시작 위치, 가중치들)`. 넓히기에서는 보통의 쌍선형과 같고, 줄이기에서는 삼각 필터 폭을 비율만큼 넓힙니다.
fn axis_weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    let stretch = scale.max(1.0);
    (0..dst)
        .map(|i| {
            let center = scale * (i as f64 + 0.5);
            let first = ((center - stretch + 0.5).floor().max(0.0)) as usize;
            let last = ((center + stretch + 0.5).floor() as usize).min(src);
            let mut w: Vec<f64> = (first..last)
                .map(|x| {
                    let t = ((x as f64 + 0.5 - center) / stretch).abs();
                    (1.0 - t).max(0.0)
                })
                .collect();
            let sum: f64 = w.iter().sum();
            w.iter_mut().for_each(|v| *v /= sum);
            (first, w.into_iter().map(|v| v as f32).collect())
        })
        .collect()
}

/// `[side², d]` 정사각 격자를 `[rows × cols, d]` 로 다시 뽑습니다 (행 우선).
pub fn interpolate_grid(table: &[f32], side: usize, d: usize, rows: usize, cols: usize) -> Vec<f32> {
    assert_eq!(table.len(), side * side * d, "위치 임베딩 표의 크기가 맞지 않습니다");
    if (rows, cols) == (side, side) {
        return table.to_vec();
    }
    let (wr, wc) = (axis_weights(side, rows), axis_weights(side, cols));
    // 열 방향을 먼저 줄이고, 그 결과를 행 방향으로 합칩니다.
    let mut by_row = vec![0f32; side * cols * d];
    for r in 0..side {
        for (c, (first, weights)) in wc.iter().enumerate() {
            let out = &mut by_row[(r * cols + c) * d..(r * cols + c + 1) * d];
            for (k, w) in weights.iter().enumerate() {
                let src = &table[(r * side + first + k) * d..(r * side + first + k + 1) * d];
                out.iter_mut().zip(src).for_each(|(o, s)| *o += w * s);
            }
        }
    }
    let mut out = vec![0f32; rows * cols * d];
    for (r, (first, weights)) in wr.iter().enumerate() {
        let row = &mut out[r * cols * d..(r + 1) * cols * d];
        for (k, w) in weights.iter().enumerate() {
            let src = &by_row[(first + k) * cols * d..(first + k + 1) * cols * d];
            row.iter_mut().zip(src).for_each(|(o, s)| *o += w * s);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn a_square_image_fills_the_limit_exactly() -> Result<()> {
        assert_eq!(naflex_grid(512, 512, 16, 256)?, (16, 16));
        assert_eq!(naflex_grid(100, 100, 16, 256)?, (16, 16), "작은 이미지는 키웁니다");
        Ok(())
    }

    #[test]
    fn the_grid_keeps_the_aspect_ratio_and_stays_within_the_limit() -> Result<()> {
        for (h, w, max) in [(1754, 1240, 256), (3508, 2480, 256), (1240, 1754, 576), (300, 1200, 256), (2000, 40, 256)] {
            let (r, c) = naflex_grid(h, w, 16, max)?;
            assert!(r * c <= max, "{h}×{w}: {r}×{c}");
            assert!(r * c * 10 >= max * 8 || r.min(c) <= 2, "{h}×{w}: 상한을 거의 다 써야 합니다 ({r}×{c})");
            let (want, got) = (h as f64 / w as f64, r as f64 / c as f64);
            assert!((want / got).max(got / want) < 1.35 || r.min(c) <= 2, "{h}×{w}: {r}×{c}");
        }
        assert!(naflex_grid(0, 10, 16, 256).is_err());
        Ok(())
    }

    #[test]
    fn patches_are_ordered_by_row_then_column_and_scaled_to_plus_minus_one() -> Result<()> {
        // 2×3 패치 이미지: 패치마다 다른 값으로 칠합니다.
        let p = 2usize;
        let mut img = RgbImage::new(6, 4);
        for y in 0..4u32 {
            for x in 0..6u32 {
                let v = (y / 2 * 3 + x / 2) as u8 * 40;
                img.put_pixel(x, y, Rgb([v, v, 255 - v]));
            }
        }
        let out = patchify(&img, p);
        assert_eq!(out.len(), 6 * p * p * 3);
        let at = |n: usize, ch: usize| ((out[n * p * p * 3 + ch] + 1.0) / 2.0 * 255.0).round() as u32;
        assert_eq!((0..6).map(|n| at(n, 0)).collect::<Vec<_>>(), vec![0, 40, 80, 120, 160, 200]);
        assert_eq!(at(1, 2), 215, "채널이 픽셀 안에서 이어집니다");
        assert!(out.iter().all(|v| (-1.0..=1.0).contains(v)));
        Ok(())
    }

    /// 넓히기는 `(i + 0.5) × src / dst − 0.5` 좌표의 보통 쌍선형과 같아야 합니다.
    #[test]
    fn enlarging_matches_plain_bilinear_with_half_pixel_centres() {
        let (side, rows, cols) = (4usize, 7usize, 9usize);
        let table: Vec<f32> = (0..side * side).map(|i| ((i * 37) % 11) as f32).collect();
        let got = interpolate_grid(&table, side, 1, rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                let fy = ((r as f32 + 0.5) * side as f32 / rows as f32 - 0.5).clamp(0.0, (side - 1) as f32);
                let fx = ((c as f32 + 0.5) * side as f32 / cols as f32 - 0.5).clamp(0.0, (side - 1) as f32);
                let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
                let (y1, x1) = ((y0 + 1).min(side - 1), (x0 + 1).min(side - 1));
                let (wy, wx) = (fy - y0 as f32, fx - x0 as f32);
                let at = |y: usize, x: usize| table[y * side + x];
                let want = (at(y0, x0) * (1.0 - wx) + at(y0, x1) * wx) * (1.0 - wy) + (at(y1, x0) * (1.0 - wx) + at(y1, x1) * wx) * wy;
                assert!((got[r * cols + c] - want).abs() < 1e-5, "({r},{c}): {} != {want}", got[r * cols + c]);
            }
        }
    }

    #[test]
    fn shrinking_averages_neighbours_instead_of_skipping_them() {
        // 번갈아 나오는 값은 절반으로 줄이면 평균에 가까워져야 합니다 (안티에일리어싱).
        let side = 8usize;
        let table: Vec<f32> = (0..side * side).map(|i| if (i % side).is_multiple_of(2) { 0.0 } else { 1.0 }).collect();
        let got = interpolate_grid(&table, side, 1, side, 4);
        for v in &got {
            assert!((v - 0.5).abs() < 0.2, "{v}");
        }
    }

    #[test]
    fn interpolation_keeps_a_constant_and_each_channel_separately() {
        let (side, d) = (4usize, 3usize);
        let table: Vec<f32> = (0..side * side * d).map(|i| [1.0, -2.0, 0.5][i % d]).collect();
        let got = interpolate_grid(&table, side, d, 5, 2);
        assert_eq!(got.len(), 5 * 2 * d);
        for (i, v) in got.iter().enumerate() {
            assert!((v - [1.0, -2.0, 0.5][i % d]).abs() < 1e-5);
        }
        assert_eq!(interpolate_grid(&table, side, d, side, side), table, "같은 격자는 그대로");
    }
}
