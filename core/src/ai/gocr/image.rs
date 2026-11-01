//! gocr 입력 전처리: 이미지 → 정규화된 패치.
//!
//! `processor_config.json` 을 따릅니다: 긴 변 · 짧은 변을 28(패치 14 × 병합 2)의 배수로 맞추고, 바이큐빅으로 줄이고,
//! CLIP 평균 · 표준편차로 정규화합니다. 패치는 2×2 병합 단위가 이어지도록 (병합 행, 병합 열, 2×2 안의 행, 열) 순서로 둡니다.

use anyhow::{bail, Result};
use image::imageops::{resize, FilterType};
use image::RgbImage;

pub const IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_7];
pub const IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];
/// 시간축 패치 크기. 이미지는 같은 프레임을 두 번 넣은 것과 같습니다.
pub const TEMPORAL_PATCH: usize = 2;
/// `processor_config.json` 의 `shortest_edge` · `longest_edge` (높이 × 너비 × 시간축의 하한 · 상한)
pub const MIN_PIXELS: usize = 12_544;
pub const MAX_PIXELS: usize = 9_633_792;

/// 전처리된 이미지. `patches` 는 패치 수 × (3 × 14 × 14) 입니다.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedImage {
    pub patches: Vec<f32>,
    /// 패치 단위 높이 · 너비 (병합 크기의 배수)
    pub grid_h: usize,
    pub grid_w: usize,
}

impl PreparedImage {
    pub fn n_patches(&self) -> usize {
        self.grid_h * self.grid_w
    }
}

/// 비전 입력 크기를 정합니다. `factor` 는 패치 × 병합(28)입니다.
pub fn smart_resize(height: usize, width: usize, factor: usize) -> Result<(usize, usize)> {
    if height == 0 || width == 0 {
        bail!("빈 이미지입니다: {height}×{width}");
    }
    let (mut h, mut w) = (height as f64, width as f64);
    let f = factor as f64;
    if h < f || w < f {
        let scale = (f / h).max(f / w);
        h = (h * scale).floor();
        w = (w * scale).floor();
    }
    if h.max(w) / h.min(w) > 200.0 {
        bail!("가로세로 비가 200 을 넘습니다: {height}×{width}");
    }
    let t = TEMPORAL_PATCH as f64;
    let mut h_bar = (h / f).round() * f;
    let mut w_bar = (w / f).round() * f;
    if t * h_bar * w_bar > MAX_PIXELS as f64 {
        let beta = (t * h * w / MAX_PIXELS as f64).sqrt();
        h_bar = f.max((h / beta / f).floor() * f);
        w_bar = f.max((w / beta / f).floor() * f);
    } else if t * h_bar * w_bar < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (t * h * w)).sqrt();
        h_bar = (h * beta / f).ceil() * f;
        w_bar = (w * beta / f).ceil() * f;
    }
    Ok((h_bar as usize, w_bar as usize))
}

/// 바이큐빅으로 맞춘 RGB 이미지를 패치로 자릅니다.
pub fn patchify(img: &RgbImage, patch: usize, merge: usize) -> Result<PreparedImage> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if !h.is_multiple_of(patch * merge) || !w.is_multiple_of(patch * merge) {
        bail!("이미지 크기 {h}×{w} 가 {} 의 배수가 아닙니다", patch * merge);
    }
    let (grid_h, grid_w) = (h / patch, w / patch);
    let raw = img.as_raw();
    let mut patches = Vec::with_capacity(grid_h * grid_w * 3 * patch * patch);
    for mr in 0..grid_h / merge {
        for mc in 0..grid_w / merge {
            for i in 0..merge {
                for j in 0..merge {
                    let (r0, c0) = ((mr * merge + i) * patch, (mc * merge + j) * patch);
                    for ch in 0..3 {
                        for y in r0..r0 + patch {
                            for x in c0..c0 + patch {
                                let v = raw[(y * w + x) * 3 + ch] as f32 / 255.0;
                                patches.push((v - IMAGE_MEAN[ch]) / IMAGE_STD[ch]);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(PreparedImage { patches, grid_h, grid_w })
}

/// 이미지 한 장을 비전 입력으로 바꿉니다.
pub fn prepare(img: &RgbImage, patch: usize, merge: usize) -> Result<PreparedImage> {
    let (h, w) = (img.height() as usize, img.width() as usize);
    let (nh, nw) = smart_resize(h, w, patch * merge)?;
    if (nh, nw) == (h, w) {
        return patchify(img, patch, merge);
    }
    patchify(&resize(img, nw as u32, nh as u32, FilterType::CatmullRom), patch, merge)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn sizes_become_multiples_of_the_factor() {
        assert_eq!(smart_resize(336, 336, 28).unwrap(), (336, 336));
        assert_eq!(smart_resize(1000, 700, 28).unwrap(), (1008, 700));
        let (h, w) = smart_resize(40, 600, 28).unwrap();
        assert!(h % 28 == 0 && w % 28 == 0 && h >= 28, "{h}×{w}");
    }

    #[test]
    fn very_small_images_grow_and_very_large_images_shrink() {
        let (h, w) = smart_resize(10, 10, 28).unwrap();
        assert!(2 * h * w >= MIN_PIXELS, "{h}×{w}");
        let (h, w) = smart_resize(6000, 4000, 28).unwrap();
        assert!(2 * h * w <= MAX_PIXELS, "{h}×{w}");
        assert!(h % 28 == 0 && w % 28 == 0);
        // 비율은 대체로 유지
        assert!((h as f64 / w as f64 - 1.5).abs() < 0.05);
    }

    #[test]
    fn extreme_aspect_ratios_and_empty_images_are_rejected() {
        assert!(smart_resize(10, 5000, 28).is_err());
        assert!(smart_resize(0, 100, 28).is_err());
    }

    #[test]
    fn patches_follow_merge_order_and_are_normalized() -> Result<()> {
        // 4×4 패치(28×28 px × 2) 이미지: 각 패치를 자기 위치 값으로 칠합니다.
        let (p, m) = (2usize, 2usize);
        let (gh, gw) = (2usize, 4usize);
        let mut img = RgbImage::new((gw * p) as u32, (gh * p) as u32);
        for y in 0..gh * p {
            for x in 0..gw * p {
                let v = (y / p * gw + x / p) as u8;
                img.put_pixel(x as u32, y as u32, Rgb([v, v, v]));
            }
        }
        let out = patchify(&img, p, m)?;
        assert_eq!((out.grid_h, out.grid_w), (gh, gw));
        let plen = 3 * p * p;
        assert_eq!(out.patches.len(), out.n_patches() * plen);
        // 병합 순서: 왼쪽 2×2 블록(0,1,4,5) 다음 오른쪽 2×2 블록(2,3,6,7)
        let at = |n: usize| ((out.patches[n * plen] * IMAGE_STD[0] + IMAGE_MEAN[0]) * 255.0).round() as u8;
        assert_eq!((0..8).map(at).collect::<Vec<_>>(), vec![0, 1, 4, 5, 2, 3, 6, 7]);
        assert!(patchify(&RgbImage::new(3, 4), 2, 2).is_err(), "배수가 아니면 거부");
        Ok(())
    }
}
