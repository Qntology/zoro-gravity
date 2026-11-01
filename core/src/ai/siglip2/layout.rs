//! 페이지를 읽을 조각으로 나눕니다. 여백을 따라 자르고(XY-cut), 큰 덩어리는 글자 행 사이에서 다시 나눕니다.
//!
//! 잉크가 있는 곳은 모두 어떤 조각에든 들어갑니다. 버릴지는 `plan` 이 SigLIP2 점수로 정합니다.
//! gocr 비전의 주의 계산은 패치 수의 제곱에 비례해서, 큰 크롭 하나보다 작은 크롭 여럿이 훨씬 빠릅니다.

use super::ink::{row_bands, Bbox, InkPage, RULE_RATIO};

/// 위아래로 이만큼 비어 있으면 다른 덩어리입니다 (글자 높이 배수). 문단의 줄 간격보다 큽니다.
const ROW_GAP_UNITS: f32 = 0.65;
/// 좌우로 페이지 너비의 이 비율만큼 비어 있어야 나눕니다.
const COL_GAP_RATIO: f32 = 0.035;
/// 좌우로 나눌 때 양쪽 다 페이지 너비의 이 비율 이상이어야 합니다. 항목 이름과 값을 떼어 놓지 않기 위해서입니다.
const COL_SIDE_RATIO: f32 = 0.2;
/// 페이지 좌우 끝의 이 비율 안쪽 띠에 있는 좁은 잉크는 내용이 아니라 펀치 구멍이나 스캐너 자국입니다.
const EDGE_STRIP_RATIO: f32 = 0.06;
/// 줄 하나짜리 덩어리는 이 비율 이상 벌어져야 좌우로 나눕니다.
const LINE_COL_GAP_RATIO: f32 = 0.25;
/// 이 높이(글자 높이 배수) 이하면 줄 하나로 봅니다.
const LINE_UNITS: f32 = 2.6;
/// 줄 하나짜리 덩어리들을 이어 붙이는 세로 간격 (글자 높이 배수)
const STACK_GAP_UNITS: f32 = 1.6;
/// 이보다 작은 덩어리는 티끌입니다 (글자 높이 배수의 제곱)
const MIN_AREA_UNITS: f32 = 0.5;
/// 문단으로 치는 최소 줄 수
const PARAGRAPH_MIN_LINES: usize = 2;
/// 문단의 줄 높이가 중앙값에서 벗어나도 되는 비율
const PARAGRAPH_HEIGHT_TOLERANCE: f32 = 0.4;
/// 문단의 줄이 덩어리 너비를 채워야 하는 비율
const PARAGRAPH_FILL: f32 = 0.6;
/// 문단으로 치는 글자 높이의 상한 (글자 높이 배수). 고정 안내문은 작은 글씨이고, 자필 진정서는 이보다 큽니다.
const PARAGRAPH_MAX_UNITS: f32 = 1.3;
const PATCH: u32 = 14;
/// 비어 있는 행 · 열로 치는 잉크 비율 (부스러기는 무시합니다)
const EMPTY_RATIO: f32 = 0.002;
/// 위아래로 나눌 수 없는 덩어리를 강제로 자를 때의 겹침
const FORCED_OVERLAP: f32 = 0.2;

/// 가장 긴 빈 구간 `(시작, 끝)`. 양 끝의 빈 구간은 세지 않습니다 (이미 잉크 경계에 맞춘 상자입니다).
fn widest_gap(counts: &[u32], limit: u32) -> Option<(u32, u32)> {
    let (mut best, mut at, mut start) = (0u32, None, None);
    for (i, c) in counts.iter().enumerate() {
        let i = i as u32;
        match (*c <= limit, start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                if i - s > best && s > 0 {
                    best = i - s;
                    at = Some((s, i));
                }
                start = None;
            }
            _ => {}
        }
    }
    at
}

fn cut(ink: &InkPage, bbox: Bbox, out: &mut Vec<Bbox>, depth: u32) {
    let Some(bbox) = ink.tighten(bbox) else {
        return;
    };
    let (x0, y0, x1, y1) = bbox;
    let (w, h) = (x1 - x0, y1 - y0);
    let u = ink.text_unit();
    if depth < 32 {
        let row_gap = widest_gap(&ink.row_counts(bbox), (w as f32 * EMPTY_RATIO) as u32);
        let col_gap = widest_gap(&ink.col_counts(bbox), (h as f32 * EMPTY_RATIO) as u32);
        let row = row_gap.filter(|(s, e)| (e - s) as f32 >= ROW_GAP_UNITS * u).map(|(s, e)| (e - s, s, e));
        let is_line = (h as f32) <= LINE_UNITS * u;
        let page_w = ink.width as f32;
        let col = col_gap
            .filter(|(s, e)| {
                let gap = (e - s) as f32;
                let (left, right) = (*s as f32, (w - e) as f32);
                let strip = EDGE_STRIP_RATIO * page_w;
                let edge_strip = ((x0 + s) as f32) <= strip || (x0 + e) as f32 >= page_w - strip;
                if edge_strip {
                    gap >= COL_GAP_RATIO * page_w * 0.5
                } else if is_line {
                    gap >= LINE_COL_GAP_RATIO * page_w
                } else {
                    gap >= COL_GAP_RATIO * page_w && left >= COL_SIDE_RATIO * page_w && right >= COL_SIDE_RATIO * page_w
                }
            })
            .map(|(s, e)| (e - s, s, e));
        match (row, col) {
            (Some((a, s, e)), c) if c.is_none_or(|(b, ..)| a >= b) => {
                cut(ink, (x0, y0, x1, y0 + s), out, depth + 1);
                cut(ink, (x0, y0 + e, x1, y1), out, depth + 1);
                return;
            }
            (_, Some((_, s, e))) => {
                cut(ink, (x0, y0, x0 + s, y1), out, depth + 1);
                cut(ink, (x0 + e, y0, x1, y1), out, depth + 1);
                return;
            }
            _ => {}
        }
    }
    let strip = (EDGE_STRIP_RATIO * ink.width as f32) as u32;
    if (w * h) as f32 >= MIN_AREA_UNITS * u * u && x1 > strip && x0 < ink.width - strip {
        out.push(bbox);
    }
}

/// 세로로 이어진 줄 하나짜리 덩어리들을 다시 잇습니다. 줄 간격이 넓은 문단은 여백을 따라 자르면 줄마다 나뉘기 때문입니다.
fn stack_lines(ink: &InkPage, mut pieces: Vec<Bbox>) -> Vec<Bbox> {
    let u = ink.text_unit();
    let is_line = |h: u32| h as f32 <= LINE_UNITS * u;
    pieces.sort_by_key(|b| b.1);
    // (상자, 마지막 줄의 높이)
    let mut out: Vec<(Bbox, u32)> = Vec::new();
    for b in pieces {
        let h = b.3 - b.1;
        let above = out.iter().rposition(|(a, _)| {
            let shared = a.2.min(b.2).saturating_sub(a.0.max(b.0));
            shared * 2 >= (a.2 - a.0).min(b.2 - b.0)
        });
        if let Some(i) = above {
            let (a, last) = out[i];
            if is_line(h) && is_line(last) && (b.1.saturating_sub(a.3) as f32) <= STACK_GAP_UNITS * u {
                out[i] = ((a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)), h);
                continue;
            }
        }
        out.push((b, h));
    }
    out.into_iter().map(|(b, _)| b).collect()
}

/// 여백을 따라 자른 덩어리들 (잉크 경계에 맞춘 상자). 위에서 아래 순서입니다.
pub fn blocks(ink: &InkPage) -> Vec<Bbox> {
    let mut out = Vec::new();
    cut(ink, ink.full(), &mut out, 0);
    stack_lines(ink, out)
}

/// 덩어리의 한 부분. `paragraph` 는 괘선 없이 고르고 작은 글자 줄이 이어진 부분입니다. 안내문은 이 모양이고, 입력란 · 표 · 서명 줄 · 큰 손글씨는 아닙니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub bbox: Bbox,
    pub paragraph: bool,
}

/// 덩어리 안쪽(위아래 끝 제외)에 가로 괘선이 있는가
fn has_inner_rule(ink: &InkPage, (x0, y0, x1, y1): Bbox) -> bool {
    let inset = ink.text_unit() as u32;
    if y1 - y0 <= inset * 2 {
        return false;
    }
    let limit = ((x1 - x0) as f32 * RULE_RATIO) as u32;
    ink.row_counts((x0, y0 + inset, x1, y1 - inset)).iter().any(|c| *c >= limit)
}

/// 덩어리를 문단과 나머지로 나눕니다. 표나 입력란처럼 괘선이 있는 덩어리는 통째로 문단이 아닙니다.
pub fn runs(ink: &InkPage, bbox: Bbox) -> Vec<Run> {
    let whole = Run { bbox, paragraph: false };
    if has_inner_rule(ink, bbox) {
        return vec![whole];
    }
    let bands = row_bands(ink, bbox);
    if bands.len() < PARAGRAPH_MIN_LINES {
        return vec![whole];
    }
    let (x0, y0, x1, y1) = bbox;
    let w = x1 - x0;
    let mut heights: Vec<u32> = bands.iter().map(|(s, e)| e - s).collect();
    heights.sort_unstable();
    let median = heights[heights.len() / 2] as f32;
    if median > PARAGRAPH_MAX_UNITS * ink.text_unit() {
        return vec![whole];
    }
    // 고른 줄: 높이가 비슷하고 덩어리 너비를 대부분 채웁니다. 문단의 마지막 줄만 짧아도 됩니다.
    let flags: Vec<(bool, bool)> = bands
        .iter()
        .map(|&(s, e)| {
            let even = ((e - s) as f32 - median).abs() <= median * PARAGRAPH_HEIGHT_TOLERANCE;
            let filled = ink.tighten((x0, s, x1, e)).is_some_and(|t| t.2 - t.0 >= (w as f32 * PARAGRAPH_FILL) as u32);
            (even, filled)
        })
        .collect();
    // 줄을 묶습니다. (시작 줄, 끝 줄, 문단인가)
    let mut groups: Vec<(usize, usize, bool)> = Vec::new();
    let mut i = 0;
    while i < bands.len() {
        let mut j = i;
        if flags[i].0 && flags[i].1 {
            while j + 1 < bands.len() && flags[j + 1].0 && flags[j + 1].1 {
                j += 1;
            }
            if j + 1 < bands.len() && flags[j + 1].0 {
                j += 1; // 마지막 짧은 줄
            }
        }
        let paragraph = j + 1 - i >= PARAGRAPH_MIN_LINES;
        match groups.last_mut() {
            // 문단이 아닌 줄들은 하나로 잇습니다.
            Some(last) if !paragraph && !last.2 => last.1 = j,
            _ => groups.push((i, j, paragraph)),
        }
        i = j + 1;
    }
    // 줄 k 와 k+1 의 사이
    let edge = |k: usize| -> u32 { (bands[k].1 + bands[k + 1].0) / 2 };
    let last = groups.len() - 1;
    groups
        .iter()
        .enumerate()
        .map(|(g, &(a, b, paragraph))| {
            let top = if g == 0 { y0 } else { edge(a - 1) };
            let bottom = if g == last { y1 } else { edge(b) };
            Run { bbox: (x0, top, x1, bottom), paragraph }
        })
        .collect()
}

/// 덩어리를 글자 행 사이에서 나눕니다. 조각의 넓이가 `max_patches` 를 넘지 않게 하되 행 하나는 쪼개지 않습니다.
/// 나눌 행을 찾지 못한 덩어리만 겹치게 자릅니다.
pub fn tiles(ink: &InkPage, bbox: Bbox, scale: f32, max_patches: u32) -> Vec<Bbox> {
    let (x0, y0, x1, y1) = bbox;
    let w = x1 - x0;
    // 넓이 상한을 이 크롭의 높이 상한으로 바꿉니다.
    let max_h = ((max_patches * PATCH * PATCH) as f32 / (w as f32 * scale * scale).max(1.0)) as u32;
    let max_h = max_h.max(PATCH * 4);
    if y1 - y0 <= max_h {
        return vec![bbox];
    }
    let bands = row_bands(ink, bbox);
    if bands.len() < 2 {
        return overlapping(bbox, max_h);
    }
    let mut out = Vec::new();
    let mut top = y0;
    let mut prev_end = bands[0].1;
    for (s, e) in bands.iter().skip(1).copied() {
        if e - top > max_h {
            let at = (prev_end + s) / 2;
            out.push((x0, top, x1, at));
            top = at;
        }
        prev_end = e;
    }
    out.push((x0, top, x1, y1));
    out.into_iter()
        .flat_map(|t| if t.3 - t.1 > max_h * 3 / 2 { overlapping(t, max_h) } else { vec![t] })
        .collect()
}

/// 행을 찾지 못한 덩어리: 글자가 잘려도 다음 조각에 온전히 들어가도록 겹쳐 자릅니다.
fn overlapping((x0, y0, x1, y1): Bbox, max_h: u32) -> Vec<Bbox> {
    let step = (max_h as f32 * (1.0 - FORCED_OVERLAP)) as u32;
    let mut out = Vec::new();
    let mut y = y0;
    loop {
        let end = (y + max_h).min(y1);
        out.push((x0, y, x1, end));
        if end == y1 {
            return out;
        }
        y += step;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    fn rect(g: &mut GrayImage, x0: u32, y0: u32, x1: u32, y1: u32) {
        for y in y0..y1 {
            for x in x0..x1 {
                g.put_pixel(x, y, Luma([20]));
            }
        }
    }

    fn page() -> GrayImage {
        GrayImage::from_pixel(1000, 1400, Luma([250]))
    }

    /// 글자 행처럼 끊어진 띠
    fn text_line(g: &mut GrayImage, x0: u32, x1: u32, y: u32, h: u32) {
        let mut x = x0;
        while x < x1 {
            rect(g, x, y, (x + 6).min(x1), y + h);
            x += 10;
        }
    }

    #[test]
    fn distant_blocks_are_separated_and_a_paragraph_stays_whole() {
        let mut g = page();
        // 제목
        text_line(&mut g, 300, 700, 80, 40);
        // 문단 네 줄: 줄 간격은 덩어리 경계보다 작습니다.
        for i in 0..4 {
            text_line(&mut g, 100, 900, 220 + i * 28, 20);
        }
        // 아래쪽 덩어리
        text_line(&mut g, 100, 900, 700, 20);
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        assert_eq!(b.len(), 3, "{b:?}");
        assert!(b[1].3 - b[1].1 > 100, "문단은 줄마다 나뉘지 않습니다: {b:?}");
    }

    #[test]
    fn side_by_side_columns_are_separated_but_a_label_stays_with_its_value() {
        let mut g = page();
        // 영수증 두 장
        for i in 0..6 {
            text_line(&mut g, 80, 420, 100 + i * 30, 20);
            text_line(&mut g, 580, 920, 100 + i * 30, 20);
        }
        // 항목 이름과 값이 벌어진 표 없는 줄
        for i in 0..4 {
            text_line(&mut g, 100, 180, 800 + i * 60, 22);
            text_line(&mut g, 330, 800, 800 + i * 60, 22);
        }
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        let top: Vec<_> = b.iter().filter(|r| r.1 < 400).collect();
        assert_eq!(top.len(), 2, "영수증 두 장: {b:?}");
        let bottom: Vec<_> = b.iter().filter(|r| r.1 >= 700).collect();
        assert!(bottom.iter().all(|r| r.0 <= 100 && r.2 >= 790), "항목 이름과 값은 한 조각입니다: {b:?}");
    }

    #[test]
    fn a_ruled_table_is_one_block() {
        let mut g = page();
        for i in 0..=8 {
            rect(&mut g, 100, 200 + i * 50, 900, 202 + i * 50);
        }
        for x in [100, 300, 600, 898] {
            rect(&mut g, x, 200, x + 2, 602);
        }
        for i in 0..8 {
            text_line(&mut g, 120, 280, 215 + i * 50, 20);
        }
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        assert_eq!(b.len(), 1, "{b:?}");
        assert_eq!(b[0], (100, 200, 900, 602));
    }

    #[test]
    fn specks_do_not_become_crops() {
        let mut g = page();
        text_line(&mut g, 100, 900, 200, 24);
        rect(&mut g, 40, 900, 44, 904);
        let ink = InkPage::new(&g);
        assert_eq!(blocks(&ink).len(), 1);
    }

    #[test]
    fn punch_holes_in_the_margin_are_not_content() {
        let mut g = page();
        // 표 옆 왼쪽 여백의 구멍
        for i in 0..=6 {
            rect(&mut g, 100, 200 + i * 60, 900, 202 + i * 60);
        }
        rect(&mut g, 100, 200, 102, 562);
        rect(&mut g, 898, 200, 900, 562);
        rect(&mut g, 20, 350, 50, 380);
        rect(&mut g, 950, 900, 985, 935);
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        assert_eq!(b, vec![(100, 200, 900, 562)], "구멍이 표 상자를 넓히거나 조각이 되면 안 됩니다");
    }

    #[test]
    fn a_blank_page_has_no_blocks() {
        assert!(blocks(&InkPage::new(&page())).is_empty());
    }

    #[test]
    fn tall_blocks_are_cut_between_lines() {
        let mut g = page();
        for i in 0..30 {
            text_line(&mut g, 100, 900, 100 + i * 40, 22);
        }
        let ink = InkPage::new(&g);
        let bbox = ink.tighten(ink.full()).unwrap();
        let t = tiles(&ink, bbox, 1.0, 1000);
        assert!(t.len() > 2, "{t:?}");
        for w in t.windows(2) {
            assert_eq!(w[0].3, w[1].1, "조각이 이어져야 합니다");
            // 경계는 행의 사이입니다: 40 간격 행 22 + 빈 18
            let inside = (w[0].3 - 100) % 40;
            assert!(inside >= 22, "경계 {} 가 글자를 지납니다", w[0].3);
        }
        for r in &t {
            let patches = (r.2 - r.0) * (r.3 - r.1) / (PATCH * PATCH);
            assert!(patches <= 1000 * 3 / 2, "{patches}");
        }
        assert_eq!((t[0].1, t.last().unwrap().3), (bbox.1, bbox.3), "처음과 끝이 덮입니다");
    }

    #[test]
    fn a_small_block_stays_whole_and_upscaling_tightens_the_limit() {
        let mut g = page();
        for i in 0..10 {
            text_line(&mut g, 100, 900, 100 + i * 40, 22);
        }
        let ink = InkPage::new(&g);
        let bbox = ink.tighten(ink.full()).unwrap();
        assert_eq!(tiles(&ink, bbox, 1.0, 4000).len(), 1);
        assert!(tiles(&ink, bbox, 3.0, 4000).len() > 1, "키우면 패치가 늘어 더 나눕니다");
    }

    #[test]
    fn a_block_without_lines_is_cut_with_overlap() {
        let t = overlapping((0, 0, 100, 1000), 300);
        assert!(t.len() >= 4);
        for w in t.windows(2) {
            assert!(w[1].1 < w[0].3, "겹쳐야 합니다: {t:?}");
        }
        assert_eq!(t.last().unwrap().3, 1000);
    }

    #[test]
    fn a_paragraph_is_separated_from_the_date_line_below_it() {
        let mut g = page();
        for i in 0..5 {
            text_line(&mut g, 100, 900, 200 + i * 26, 18);
        }
        text_line(&mut g, 100, 500, 330, 18); // 문단의 짧은 마지막 줄
        text_line(&mut g, 420, 580, 356, 22); // 가운데 짧은 날짜 줄
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        assert_eq!(b.len(), 1, "{b:?}");
        let r = runs(&ink, b[0]);
        assert_eq!(r.len(), 2, "{r:?}");
        assert!(r[0].paragraph && !r[1].paragraph, "{r:?}");
        assert!(r[1].bbox.1 >= 348 && r[1].bbox.1 <= 356, "날짜 줄 바로 위에서 나뉩니다: {r:?}");
    }

    #[test]
    fn loosely_spaced_lines_are_stacked_back_into_a_paragraph() {
        let mut g = page();
        // 줄 사이가 여백 기준(글자 높이의 0.65배)보다 넓습니다.
        for i in 0..4 {
            text_line(&mut g, 100, 900, 300 + i * 34, 20);
        }
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        assert_eq!(b.len(), 1, "{b:?}");
        let r = runs(&ink, b[0]);
        assert_eq!(r.len(), 1, "{r:?}");
        assert!(r[0].paragraph, "{r:?}");
    }

    #[test]
    fn stacking_stays_within_a_column_and_skips_tall_blocks() {
        let mut g = page();
        // 왼쪽 열의 줄들과 멀리 떨어진 오른쪽 열의 줄들
        for i in 0..3 {
            text_line(&mut g, 80, 420, 100 + i * 34, 20);
            text_line(&mut g, 580, 920, 100 + i * 34, 20);
        }
        // 괘선 표 바로 아래 줄: 표에 붙지 않습니다.
        for i in 0..=4 {
            rect(&mut g, 100, 400 + i * 50, 900, 402 + i * 50);
        }
        rect(&mut g, 100, 400, 102, 602);
        rect(&mut g, 898, 400, 900, 602);
        text_line(&mut g, 100, 900, 625, 20);
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        assert_eq!(b.len(), 4, "{b:?}");
    }

    #[test]
    fn large_handwriting_is_not_a_paragraph() {
        let mut g = page();
        // 페이지 1400 px 의 눈금은 16.8 px 입니다. 글자 34 px 는 고정 안내문이 아닙니다.
        for i in 0..5 {
            text_line(&mut g, 100, 900, 200 + i * 60, 34);
        }
        let ink = InkPage::new(&g);
        let b = blocks(&ink);
        assert!(b.iter().all(|x| runs(&ink, *x).iter().all(|r| !r.paragraph)), "{b:?}");
    }

    #[test]
    fn a_ruled_form_is_never_a_paragraph() {
        let mut g = page();
        for i in 0..=6 {
            rect(&mut g, 100, 200 + i * 60, 900, 202 + i * 60);
        }
        for x in [100, 898] {
            rect(&mut g, x, 200, x + 2, 562);
        }
        for i in 0..6 {
            text_line(&mut g, 120, 880, 220 + i * 60, 20);
        }
        let ink = InkPage::new(&g);
        let b = ink.tighten(ink.full()).unwrap();
        assert_eq!(runs(&ink, b), vec![Run { bbox: b, paragraph: false }]);
    }

    #[test]
    fn a_boxed_notice_is_still_a_paragraph() {
        let mut g = page();
        rect(&mut g, 100, 200, 900, 202);
        rect(&mut g, 100, 340, 900, 342);
        rect(&mut g, 100, 200, 102, 342);
        rect(&mut g, 898, 200, 900, 342);
        for i in 0..4 {
            text_line(&mut g, 120, 880, 215 + i * 28, 18);
        }
        let ink = InkPage::new(&g);
        let b = ink.tighten(ink.full()).unwrap();
        let r = runs(&ink, b);
        assert!(r.iter().any(|x| x.paragraph), "{r:?}");
    }

    #[test]
    fn a_single_line_is_not_a_paragraph() {
        let mut g = page();
        text_line(&mut g, 100, 900, 300, 22);
        let ink = InkPage::new(&g);
        let b = ink.tighten(ink.full()).unwrap();
        assert_eq!(runs(&ink, b), vec![Run { bbox: b, paragraph: false }]);
    }

    #[test]
    fn gaps_at_the_edges_do_not_count() {
        assert_eq!(widest_gap(&[0, 0, 5, 5, 0, 0, 0, 5, 0], 0), Some((4, 7)));
        assert_eq!(widest_gap(&[0, 5, 5, 0], 0), None);
    }
}
