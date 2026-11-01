//! 픽셀 수준 분석: 잉크 지도, 기울기 보정, 글자 행, 글자 높이. GPU 를 쓰지 않습니다.
//!
//! 패치 격자는 한 칸이 글자 여러 줄을 덮을 만큼 거칠어서 경계는 여기서 잡습니다.

use image::{GrayImage, Rgb, RgbImage};

pub type Bbox = (u32, u32, u32, u32);

/// 크롭을 키우는 상한. 그 이상은 없는 획이 생길 뿐입니다.
pub const MAX_UPSCALE: f32 = 4.0;
/// 크롭의 긴 변 상한
pub const CROP_MAX_SIDE: u32 = 2048;
/// 종이 밝기보다 이만큼 어두우면 잉크입니다. 표 머리 칸의 옅은 음영은 잉크가 아닙니다.
const INK_DROP: u8 = 70;
/// 기울기 탐색 범위와 간격 (도)
const SKEW_RANGE: f32 = 3.0;
const SKEW_STEP: f32 = 0.2;
const MIN_BAND_PX: u32 = 4;
const MERGE_GAP_PX: u32 = 4;
const BAND_RATIO: f32 = 0.04;
/// 행 · 열의 이 비율 이상이 잉크면 괘선입니다.
pub const RULE_RATIO: f32 = 0.8;
/// 페이지 높이 대비 글자 높이 (A4 10 pt ≈ 1.2%)
const TEXT_UNIT: f32 = 0.012;
/// 세로 괘선은 이 높이(글자 높이 배수) 이상인 조각에서만 찾습니다.
const RULE_MIN_UNITS: f32 = 3.5;

/// 잉크 지도. 행 · 열 프로파일과 경계 찾기를 맡습니다.
pub struct InkPage {
    pub width: u32,
    pub height: u32,
    mask: Vec<u8>,
}

/// 밝기 히스토그램의 90 백분위. 글자보다 종이가 훨씬 많다는 가정입니다.
fn paper_level(gray: &GrayImage) -> u8 {
    let mut hist = [0u64; 256];
    gray.as_raw().iter().for_each(|v| hist[*v as usize] += 1);
    let goal = gray.as_raw().len() as u64 * 9 / 10;
    let mut seen = 0;
    for (level, n) in hist.iter().enumerate() {
        seen += n;
        if seen >= goal {
            return level as u8;
        }
    }
    255
}

impl InkPage {
    pub fn new(gray: &GrayImage) -> Self {
        let gate = paper_level(gray).saturating_sub(INK_DROP).max(32);
        let mask = gray.as_raw().iter().map(|v| (*v < gate) as u8).collect();
        Self { width: gray.width(), height: gray.height(), mask }
    }

    pub fn full(&self) -> Bbox {
        (0, 0, self.width, self.height)
    }

    /// 해상도와 무관하게 여백을 재는 눈금: 페이지 높이에서 짐작한 글자 높이 (A4 10 pt)
    pub fn text_unit(&self) -> f32 {
        self.height as f32 * TEXT_UNIT
    }

    fn row(&self, y: u32, x0: u32, x1: u32) -> &[u8] {
        let w = self.width as usize;
        let s = y as usize * w;
        &self.mask[s + x0 as usize..s + x1 as usize]
    }

    /// 행마다 잉크 픽셀 수
    pub fn row_counts(&self, (x0, y0, x1, y1): Bbox) -> Vec<u32> {
        (y0..y1).map(|y| self.row(y, x0, x1).iter().map(|v| *v as u32).sum()).collect()
    }

    /// 열마다 잉크 픽셀 수
    pub fn col_counts(&self, (x0, y0, x1, y1): Bbox) -> Vec<u32> {
        let mut out = vec![0u32; (x1 - x0) as usize];
        for y in y0..y1 {
            out.iter_mut().zip(self.row(y, x0, x1)).for_each(|(o, v)| *o += *v as u32);
        }
        out
    }

    pub fn ink_in(&self, (x0, y0, x1, y1): Bbox) -> u64 {
        (y0..y1).map(|y| self.row(y, x0, x1).iter().map(|v| *v as u64).sum::<u64>()).sum()
    }

    /// 상자 안 잉크의 경계. 드문 점은 잉크로 치지 않습니다. 잉크가 없으면 None.
    pub fn tighten(&self, bbox: Bbox) -> Option<Bbox> {
        if bbox.2 <= bbox.0 || bbox.3 <= bbox.1 {
            return None;
        }
        let span = |counts: &[u32], noise: u32| -> Option<(u32, u32)> {
            let first = counts.iter().position(|c| *c > noise)?;
            let last = counts.iter().rposition(|c| *c > noise)?;
            Some((first as u32, last as u32 + 1))
        };
        let rows = self.row_counts(bbox);
        let (y0, y1) = span(&rows, 1)?;
        let cols = self.col_counts((bbox.0, bbox.1 + y0, bbox.2, bbox.1 + y1));
        let (x0, x1) = span(&cols, 1)?;
        Some((bbox.0 + x0, bbox.1 + y0, bbox.0 + x1, bbox.1 + y1))
    }
}

impl InkPage {
    /// 괘선을 뺀 글자의 경계. 입력란은 표 테두리가 페이지 끝까지 가도 글자는 일부에만 있어서, 이 경계로 자르면 패치가 크게 줄어듭니다.
    pub fn text_box(&self, bbox: Bbox) -> Option<Bbox> {
        let (x0, y0, x1, y1) = self.tighten(bbox)?;
        let rule = |n: u32| (n as f32 * RULE_RATIO) as u32;
        let rows = self.row_counts((x0, y0, x1, y1));
        let limit = rule(x1 - x0);
        let text_rows: Vec<u32> = (y0..y1).filter(|y| rows[(y - y0) as usize] < limit).collect();
        if text_rows.is_empty() {
            return None;
        }
        let mut cols = vec![0u32; (x1 - x0) as usize];
        for y in &text_rows {
            cols.iter_mut().zip(self.row(*y, x0, x1)).for_each(|(c, v)| *c += *v as u32);
        }
        // 줄 하나짜리 조각에서는 세로 획도 조각 높이를 다 채우므로 세로 괘선으로 치지 않습니다.
        let limit = if ((y1 - y0) as f32) < RULE_MIN_UNITS * self.text_unit() { u32::MAX } else { rule(text_rows.len() as u32) };
        let keep: Vec<bool> = cols.iter().map(|c| *c > 1 && *c < limit).collect();
        let first = keep.iter().position(|k| *k)? as u32;
        let last = keep.iter().rposition(|k| *k)? as u32;
        // 세로 괘선을 뺀 칸에서 글자 행의 위아래를 다시 잡습니다.
        let has_text = |y: &u32| self.row(*y, x0, x1).iter().zip(&keep).filter(|(v, k)| **v != 0 && **k).count() > 1;
        let top = *text_rows.iter().find(|y| has_text(y))?;
        let bottom = *text_rows.iter().rfind(|y| has_text(y))?;
        Some((x0 + first, top, x0 + last + 1, bottom + 1))
    }
}

/// 잉크 프로파일에서 글자 행을 찾습니다. 세로 괘선이 만드는 바닥값 위로 조금만 올라와도 행입니다.
/// 기준을 높이면 긴 줄 옆의 짧은 줄(날짜 등)이 행으로 잡히지 않아 문단과 함께 버려질 수 있습니다.
fn bands(profile: &[u32], offset: u32) -> Vec<(u32, u32)> {
    if profile.len() < (MIN_BAND_PX * 2) as usize {
        return Vec::new();
    }
    let mut sorted: Vec<f32> = profile.iter().map(|v| *v as f32).collect();
    sorted.sort_by(f32::total_cmp);
    let (floor, peak) = (sorted[sorted.len() / 10], sorted[sorted.len() * 9 / 10]);
    if peak - floor < 1.0 {
        return Vec::new();
    }
    let gate = floor + ((peak - floor) * BAND_RATIO).max(2.0);

    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut start = None;
    for (i, v) in profile.iter().enumerate() {
        match (*v as f32 > gate, start) {
            (true, None) => start = Some(i as u32),
            (false, Some(s)) => {
                runs.push((s, i as u32));
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        runs.push((s, profile.len() as u32));
    }
    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (s, e) in runs {
        match merged.last_mut() {
            Some(last) if s.saturating_sub(last.1) <= MERGE_GAP_PX => last.1 = e,
            _ => merged.push((s, e)),
        }
    }
    merged.retain(|(s, e)| e - s >= MIN_BAND_PX);
    merged.into_iter().map(|(s, e)| (offset + s, offset + e)).collect()
}

/// 상자 안의 글자 행 `(y0, y1)` 목록 (페이지 좌표)
pub fn row_bands(ink: &InkPage, bbox: Bbox) -> Vec<(u32, u32)> {
    if bbox.2 <= bbox.0 || bbox.3 <= bbox.1 {
        return Vec::new();
    }
    bands(&ink.row_counts(bbox), bbox.1)
}

/// 크롭의 글자 높이. 행 높이의 하위 1/4 입니다. 작은 글자가 읽힐 때까지 키워야 하므로 큰 글자에 맞추지 않습니다.
pub fn text_height(ink: &InkPage, bbox: Bbox) -> Option<f32> {
    let mut h: Vec<f32> = row_bands(ink, bbox).iter().map(|(s, e)| (e - s) as f32).collect();
    if h.is_empty() {
        return None;
    }
    h.sort_by(f32::total_cmp);
    Some(if h.len() >= 4 { h[h.len() / 4] } else { h[0] })
}

/// 글자 높이가 `target` 이 되도록 배율을 정합니다. 가로세로 비를 지키도록 긴 변 상한 안에서만 키웁니다.
pub fn upscale_factor(text_h: Option<f32>, target: f32, width: u32, height: u32) -> f32 {
    let want = text_h.map_or(1.0, |h| (target / h.max(1.0)).clamp(1.0, MAX_UPSCALE));
    let cap = CROP_MAX_SIDE as f32 / width.max(height).max(1) as f32;
    want.min(cap.max(1.0))
}

/// 글자 행이 곧을수록 행 프로파일의 변동이 커지는 점을 이용해 기울기를 찾습니다 (도, 시계 방향이 양수).
pub fn estimate_skew(gray: &GrayImage) -> f32 {
    // 행은 전부, 열은 띄엄띄엄 봅니다. 행을 건너뛰면 곧은 페이지에서 표본 간격 때문에 각도가 틀어집니다.
    const COL_STEP: usize = 3;
    const BIN: f32 = 2.0;
    let ink = InkPage::new(gray);
    let (w, h) = (ink.width as usize, ink.height as usize);
    let mut points: Vec<(f32, f32)> = Vec::new();
    for y in 0..h {
        for x in (0..w).step_by(COL_STEP) {
            if ink.mask[y * w + x] != 0 {
                points.push((x as f32 - w as f32 / 2.0, y as f32 - h as f32 / 2.0));
            }
        }
    }
    if points.len() < 64 {
        return 0.0;
    }
    let rows = (h + w / 8) / BIN as usize + 2;
    let contrast = |deg: f32| -> f32 {
        let tan = deg.to_radians().tan();
        let mut hist = vec![0f32; rows];
        for (x, y) in &points {
            // 시계 방향으로 deg 만큼 기운 줄은 x 가 커질수록 y 가 tan(deg) 만큼 내려갑니다.
            let row = (y - x * tan) / BIN + rows as f32 / 2.0;
            if row >= 0.0 && (row as usize) < rows {
                hist[row as usize] += 1.0;
            }
        }
        hist.iter().map(|v| v * v).sum()
    };
    let n = (SKEW_RANGE / SKEW_STEP).round() as i32;
    let (mut best, mut at) = (f32::MIN, 0.0);
    for i in -n..=n {
        let deg = i as f32 * SKEW_STEP;
        let c = contrast(deg);
        if c > best {
            best = c;
            at = deg;
        }
    }
    at
}

/// 페이지를 `deg` 만큼 되돌립니다 (`estimate_skew` 의 값을 그대로 넘기면 곧아집니다). 크기는 그대로이고 빈 곳은 흰색입니다.
pub fn deskew(img: &RgbImage, deg: f32) -> RgbImage {
    if deg == 0.0 {
        return img.clone();
    }
    let (w, h) = img.dimensions();
    let (sin, cos) = deg.to_radians().sin_cos();
    let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
    let sample = |x: f32, y: f32| -> Rgb<u8> {
        if x < 0.0 || y < 0.0 || x >= (w - 1) as f32 || y >= (h - 1) as f32 {
            return Rgb([255, 255, 255]);
        }
        let (x0, y0) = (x as u32, y as u32);
        let (fx, fy) = (x - x0 as f32, y - y0 as f32);
        let px = |xx: u32, yy: u32, c: usize| img.get_pixel(xx, yy)[c] as f32;
        let mut out = [0u8; 3];
        for (c, o) in out.iter_mut().enumerate() {
            let top = px(x0, y0, c) * (1.0 - fx) + px(x0 + 1, y0, c) * fx;
            let bot = px(x0, y0 + 1, c) * (1.0 - fx) + px(x0 + 1, y0 + 1, c) * fx;
            *o = (top * (1.0 - fy) + bot * fy + 0.5) as u8;
        }
        Rgb(out)
    };
    RgbImage::from_fn(w, h, |x, y| {
        let (dx, dy) = (x as f32 + 0.5 - cx, y as f32 + 0.5 - cy);
        sample(cx + dx * cos - dy * sin - 0.5, cy + dx * sin + dy * cos - 0.5)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    /// 높이 `h` 의 글자 행을 `pitch` 간격으로 `n` 줄 그린 페이지
    fn lines(w: u32, n: u32, h: u32, pitch: u32) -> GrayImage {
        let mut g = GrayImage::from_pixel(w, n * pitch + 40, Luma([252]));
        for i in 0..n {
            let y0 = 20 + i * pitch;
            for y in y0..y0 + h {
                for x in 10..w - 10 {
                    if x % 5 < 3 {
                        g.put_pixel(x, y, Luma([30]));
                    }
                }
            }
        }
        g
    }

    #[test]
    fn the_paper_level_ignores_the_ink() {
        let g = lines(300, 5, 18, 40);
        assert_eq!(paper_level(&g), 252);
    }

    #[test]
    fn a_shaded_header_is_not_ink() {
        let mut g = GrayImage::from_pixel(100, 100, Luma([250]));
        for y in 0..20 {
            for x in 0..100 {
                g.put_pixel(x, y, Luma([225]));
            }
        }
        let ink = InkPage::new(&g);
        assert_eq!(ink.ink_in(ink.full()), 0);
    }

    #[test]
    fn bands_follow_the_text_lines() {
        let ink = InkPage::new(&lines(300, 5, 18, 40));
        let b = row_bands(&ink, ink.full());
        assert_eq!(b.len(), 5, "{b:?}");
        for (i, (s, e)) in b.iter().enumerate() {
            let want = 20 + i as u32 * 40;
            assert!(s.abs_diff(want) <= 1 && e.abs_diff(want + 18) <= 1, "{s}..{e}");
        }
    }

    #[test]
    fn bands_use_page_coordinates() {
        let ink = InkPage::new(&lines(300, 5, 18, 40));
        let b = row_bands(&ink, (0, 90, 300, ink.height));
        assert_eq!(b[0].0, 100, "{b:?}");
    }

    #[test]
    fn a_blank_area_has_no_bands_and_no_ink_box() {
        let ink = InkPage::new(&GrayImage::from_pixel(100, 100, Luma([250])));
        assert!(row_bands(&ink, ink.full()).is_empty());
        assert_eq!(ink.tighten(ink.full()), None);
        assert_eq!(text_height(&ink, ink.full()), None);
    }

    #[test]
    fn tightening_finds_the_ink_and_skips_specks() {
        let mut g = GrayImage::from_pixel(100, 100, Luma([250]));
        for y in 30..40 {
            for x in 20..70 {
                g.put_pixel(x, y, Luma([10]));
            }
        }
        g.put_pixel(3, 3, Luma([10]));
        let ink = InkPage::new(&g);
        assert_eq!(ink.tighten(ink.full()), Some((20, 30, 70, 40)));
    }

    #[test]
    fn the_text_box_ignores_the_rules_around_the_text() {
        let mut g = GrayImage::from_pixel(400, 200, Luma([250]));
        let mut rect = |x0: u32, y0: u32, x1: u32, y1: u32| {
            for y in y0..y1 {
                for x in x0..x1 {
                    g.put_pixel(x, y, Luma([15]));
                }
            }
        };
        // 표 테두리와 그 안의 왼쪽 글자
        rect(10, 20, 390, 22);
        rect(10, 180, 390, 182);
        rect(10, 20, 12, 182);
        rect(388, 20, 390, 182);
        rect(50, 60, 150, 100);
        let ink = InkPage::new(&g);
        assert_eq!(ink.tighten(ink.full()), Some((10, 20, 390, 182)));
        assert_eq!(ink.text_box(ink.full()), Some((50, 60, 150, 100)));
    }

    #[test]
    fn a_box_with_only_rules_has_no_text() {
        let mut g = GrayImage::from_pixel(100, 100, Luma([250]));
        for x in 0..100 {
            g.put_pixel(x, 50, Luma([15]));
        }
        let ink = InkPage::new(&g);
        assert_eq!(ink.text_box(ink.full()), None);
    }

    #[test]
    fn small_text_is_enlarged_up_to_the_limits() {
        let ink = InkPage::new(&lines(300, 6, 15, 40));
        let h = text_height(&ink, ink.full()).unwrap();
        assert!((14.0..=16.0).contains(&h), "{h}");
        assert!((upscale_factor(Some(15.0), 30.0, 300, 200) - 2.0).abs() < 1e-6);
        assert_eq!(upscale_factor(Some(60.0), 30.0, 300, 200), 1.0, "큰 글자는 그대로");
        assert_eq!(upscale_factor(Some(2.0), 30.0, 300, 200), MAX_UPSCALE, "배율 상한");
        assert!((upscale_factor(Some(10.0), 30.0, 1600, 200) - 2048.0 / 1600.0).abs() < 1e-5, "긴 변 상한");
        assert_eq!(upscale_factor(Some(10.0), 30.0, 3000, 200), 1.0, "이미 긴 변을 넘으면 줄이지 않습니다");
        assert_eq!(upscale_factor(None, 30.0, 300, 200), 1.0);
    }

    fn rgb_of(g: &GrayImage) -> RgbImage {
        RgbImage::from_fn(g.width(), g.height(), |x, y| {
            let v = g.get_pixel(x, y)[0];
            Rgb([v, v, v])
        })
    }

    fn gray_of(img: &RgbImage) -> GrayImage {
        GrayImage::from_fn(img.width(), img.height(), |x, y| Luma([img.get_pixel(x, y)[0]]))
    }

    #[test]
    fn a_tilted_page_is_measured_and_straightened() {
        let straight = lines(800, 18, 18, 42);
        assert_eq!(estimate_skew(&straight), 0.0);
        for deg in [-1.2f32, 0.8, 2.0] {
            // 곧은 페이지를 돌려 기운 페이지를 만들고, 잰 각도로 되돌립니다.
            let tilted = deskew(&rgb_of(&straight), -deg);
            let found = estimate_skew(&gray_of(&tilted));
            assert!((found - deg).abs() <= 0.21, "{deg} 도를 {found} 도로 재었습니다");
            let fixed = deskew(&tilted, found);
            let ink = InkPage::new(&gray_of(&fixed));
            assert!(row_bands(&ink, ink.full()).len() >= 16, "곧게 편 뒤에는 행이 나뉘어야 합니다 ({deg} 도)");
        }
    }

    #[test]
    fn a_blank_page_has_no_skew() {
        assert_eq!(estimate_skew(&GrayImage::from_pixel(200, 200, Luma([250]))), 0.0);
    }
}
