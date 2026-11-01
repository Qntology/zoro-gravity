//! 크롭 계획: 조각마다 SigLIP2 점수를 모아 읽을지 정합니다 (logis-center 의 영역 경쟁과 같은 역할입니다).
//!
//! 조각 안 잉크 칸의 점수를 평균해 종류끼리 겨룹니다. 값을 놓치면 판단이 틀어지지만 안내문을 더 읽는 것은 시간만 더 들 뿐이라서,
//! 두 신호가 모두 맞을 때만 버립니다: 생김새가 문단이고(`layout::runs`), 안내문이 다른 모든 종류를 여유 있게 이겨야 합니다.
//! 시각 앵커 파일이 없으면 버리는 조각 없이 전부 읽습니다.

use super::arena::{self, Arena};
use super::ink::{text_height, upscale_factor, Bbox, InkPage};
use super::layout::{blocks, runs, tiles};
use super::probe::{smooth, ProbeBank};
use super::PatchGrid;
use anyhow::{anyhow, Result};

/// 계획의 기준값
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanParams {
    /// 안내문이 나머지 최고 점수를 이만큼 넘어야 버립니다.
    pub drop_margin: f32,
    /// 이 점수를 넘는 칸의 덩어리를 서명 · 날인으로 봅니다.
    pub sign_gate: f32,
    /// 크롭을 키워 맞출 글자 높이 (px)
    pub text_px: f32,
    /// 크롭 하나의 패치 수 상한 (gocr 패치 14 px). 작을수록 값을 건너뛰는 일이 줄고 디코딩이 빨라집니다.
    pub max_tile_patches: u32,
    /// 조각의 종류를 평균 점수 대신 패치 경쟁의 영토로 정합니다.
    pub use_arena: bool,
    /// 패치 경쟁을 쓸 때 문단을 버리는 기준: 잉크 칸 중 안내문의 확정 영토 비율
    pub drop_share: f32,
}

impl Default for PlanParams {
    fn default() -> Self {
        Self { drop_margin: 0.15, sign_gate: 0.2, text_px: 30.0, max_tile_patches: 1200, use_arena: true, drop_share: 0.5 }
    }
}

/// 제목은 페이지 위쪽 이 비율 안에서 찾습니다.
const TITLE_REGION: f32 = 0.45;
/// 조각 위아래 · 좌우에 남겨 두는 여백 (px). 획 끝이 잘리지 않게 합니다.
const PAD: u32 = 6;
/// 크롭 넓이의 이 비율 이상이 더 큰 크롭 안에 있으면 같은 내용으로 보고 뺍니다.
const CONTAINED: f32 = 0.8;
/// 안내문의 영토로 치는 칸의 최소 점수 차이. 칸 하나의 값이라 조각 평균의 `drop_margin` 보다 작습니다.
const MIN_CELL_MARGIN: f32 = 0.05;
/// 패치 경쟁의 종류 순서 (`Scores::all`)
const FORM: usize = 0;
const TABLE: usize = 1;
const GUIDE: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CropKind {
    Title,
    Form,
    Table,
    Sign,
}

impl CropKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Title => "제목",
            Self::Form => "입력란",
            Self::Table => "표",
            Self::Sign => "서명",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CropPlan {
    pub kind: CropKind,
    pub bbox: Bbox,
    /// 이 크롭을 키울 배율 (글자 높이 기준)
    pub upscale: f32,
    /// 안내문 점수 − 나머지 최고 점수. 낮을수록 읽을 내용이 뚜렷합니다. 점수를 재지 않은 조각은 없습니다.
    pub guide_margin: Option<f32>,
}

impl CropPlan {
    /// gocr 비전 패치 수의 추정 (14 px)
    pub fn patches(&self) -> u32 {
        let (w, h) = ((self.bbox.2 - self.bbox.0) as f32, (self.bbox.3 - self.bbox.1) as f32);
        (w * h * self.upscale * self.upscale / (14.0 * 14.0)) as u32
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct PagePlan {
    pub width: u32,
    pub height: u32,
    /// 읽기 순서 (위 → 아래, 왼쪽 → 오른쪽)
    pub crops: Vec<CropPlan>,
    /// 안내문으로 보고 버린 조각
    pub dropped: Vec<Bbox>,
    /// 시각 앵커 없이 전부 읽는 계획인가
    pub unfiltered: bool,
}

impl PagePlan {
    /// 페이지 넓이 대비 크롭 넓이의 합 (서명 제외, 겹침 없음)
    pub fn kept_area(&self) -> f32 {
        let total: u64 = self.crops.iter().filter(|c| c.kind != CropKind::Sign).map(|c| (c.bbox.2 - c.bbox.0) as u64 * (c.bbox.3 - c.bbox.1) as u64).sum();
        total as f32 / (self.width as u64 * self.height as u64).max(1) as f32
    }
}

/// 격자 칸마다의 종류별 점수 (3×3 평균)
pub struct Scores {
    rows: usize,
    cols: usize,
    form: Vec<f32>,
    /// 서명 · 날인은 한두 칸짜리라 평균하지 않은 값을 씁니다.
    sign: Vec<f32>,
    table: Vec<f32>,
    title: Vec<f32>,
    guide: Vec<f32>,
}

impl Scores {
    pub fn new(bank: &ProbeBank, grid: &PatchGrid) -> Result<Self> {
        let all = bank.score(grid)?;
        let raw = |name: &str| -> Result<&Vec<f32>> {
            let k = bank.index_of(name).ok_or_else(|| anyhow!("시각 앵커에 {name} 이 없습니다"))?;
            Ok(&all[k])
        };
        let pick = |name: &str| -> Result<Vec<f32>> { Ok(smooth(raw(name)?, grid.rows, grid.cols)) };
        Ok(Self { rows: grid.rows, cols: grid.cols, form: pick("form")?, sign: raw("sign")?.clone(), table: pick("table")?, title: pick("title")?, guide: pick("guide")? })
    }

    fn all(&self) -> [&[f32]; 4] {
        [&self.form, &self.table, &self.title, &self.guide]
    }

    /// 패치 경쟁. 읽을 종류는 영토가 없어도 탈락시키지 않습니다. 안내문만 탈락할 수 있습니다.
    fn compete(&self, ink: &InkPage) -> Arena {
        let (w, h) = (ink.width as usize, ink.height as usize);
        let active: Vec<bool> = (0..self.rows * self.cols)
            .map(|i| {
                let (r, c) = (i / self.cols, i % self.cols);
                ink.ink_in(((c * w / self.cols) as u32, (r * h / self.rows) as u32, ((c + 1) * w / self.cols) as u32, ((r + 1) * h / self.rows) as u32)) > 0
            })
            .collect();
        arena::run(&self.all(), &active, &[true, true, true, false])
    }

    /// 상자가 덮는 칸들. 잉크가 있는 칸의 인덱스만 돌려줍니다.
    fn cells(&self, ink: &InkPage, (x0, y0, x1, y1): Bbox) -> Vec<usize> {
        let (w, h) = (ink.width as usize, ink.height as usize);
        let mut out = Vec::new();
        for r in (y0 as usize * self.rows / h)..=((y1 as usize - 1) * self.rows / h).min(self.rows - 1) {
            for c in (x0 as usize * self.cols / w)..=((x1 as usize - 1) * self.cols / w).min(self.cols - 1) {
                let cell = ((c * w / self.cols) as u32, (r * h / self.rows) as u32, ((c + 1) * w / self.cols) as u32, ((r + 1) * h / self.rows) as u32);
                let clipped = (cell.0.max(x0), cell.1.max(y0), cell.2.min(x1), cell.3.min(y1));
                if clipped.2 > clipped.0 && clipped.3 > clipped.1 && ink.ink_in(clipped) > 0 {
                    out.push(r * self.cols + c);
                }
            }
        }
        out
    }

    fn mean(values: &[f32], cells: &[usize]) -> f32 {
        cells.iter().map(|&i| values[i]).sum::<f32>() / cells.len().max(1) as f32
    }
}

/// 조각 하나의 평균 점수 `(서식, 표, 제목, 안내문)`
struct Tally {
    form: f32,
    table: f32,
    title: f32,
    guide: f32,
}

fn tally(s: &Scores, ink: &InkPage, bbox: Bbox) -> Tally {
    let cells = s.cells(ink, bbox);
    Tally {
        form: Scores::mean(&s.form, &cells),
        table: Scores::mean(&s.table, &cells),
        title: Scores::mean(&s.title, &cells),
        guide: Scores::mean(&s.guide, &cells),
    }
}

/// 조각 안 잉크 칸 중 이 종류의 확정 영토 수
fn territory(a: &Arena, cells: &[usize], kind: usize) -> usize {
    cells.iter().filter(|&&i| a.owns(kind, i)).count()
}

/// 크롭 NMS: 더 큰 크롭 안에 거의 들어가는 크롭을 뺍니다. 강제로 겹쳐 자른 조각이 같은 글을 두 번 읽지 않게 합니다. 서명 크롭은 따로 씁니다.
fn suppress_contained(crops: &mut Vec<CropPlan>) {
    let area = |b: Bbox| (b.2 - b.0) as u64 * (b.3 - b.1) as u64;
    let inside = |a: Bbox, b: Bbox| -> u64 {
        let (w, h) = (a.2.min(b.2).saturating_sub(a.0.max(b.0)), a.3.min(b.3).saturating_sub(a.1.max(b.1)));
        w as u64 * h as u64
    };
    let snapshot: Vec<(CropKind, Bbox)> = crops.iter().map(|c| (c.kind, c.bbox)).collect();
    let mut keep = vec![true; crops.len()];
    for (i, (kind, b)) in snapshot.iter().enumerate() {
        if *kind == CropKind::Sign || *kind == CropKind::Title {
            continue;
        }
        let swallowed = snapshot.iter().enumerate().any(|(j, (other, o))| {
            j != i && keep[j] && *other != CropKind::Sign && (area(*o) > area(*b) || (area(*o) == area(*b) && j < i)) && inside(*b, *o) as f32 >= area(*b) as f32 * CONTAINED
        });
        keep[i] = !swallowed;
    }
    let mut flags = keep.into_iter();
    crops.retain(|_| flags.next().unwrap_or(true));
}

fn pad(ink: &InkPage, (x0, y0, x1, y1): Bbox) -> Bbox {
    (x0.saturating_sub(PAD), y0.saturating_sub(PAD), (x1 + PAD).min(ink.width), (y1 + PAD).min(ink.height))
}

fn crop_of(ink: &InkPage, p: &PlanParams, kind: CropKind, bbox: Bbox, guide_margin: Option<f32>) -> CropPlan {
    let bbox = pad(ink, bbox);
    let upscale = upscale_factor(text_height(ink, bbox), p.text_px, bbox.2 - bbox.0, bbox.3 - bbox.1);
    CropPlan { kind, bbox, upscale, guide_margin }
}

/// 서명 · 날인 칸의 덩어리. 다른 조각과 겹쳐도 따로 냅니다 (서명 검증과 합성 위치에 씁니다).
fn sign_regions(s: &Scores, ink: &InkPage, gate: f32) -> Vec<Bbox> {
    let (rows, cols) = (s.rows, s.cols);
    let mut seen = vec![false; rows * cols];
    let mut out = Vec::new();
    let (w, h) = (ink.width as usize, ink.height as usize);
    for start in 0..rows * cols {
        if seen[start] || s.sign[start] < gate {
            continue;
        }
        seen[start] = true;
        let mut stack = vec![start];
        let (mut r0, mut r1, mut c0, mut c1) = (rows, 0, cols, 0);
        while let Some(i) = stack.pop() {
            let (r, c) = (i / cols, i % cols);
            (r0, r1, c0, c1) = (r0.min(r), r1.max(r), c0.min(c), c1.max(c));
            for rr in r.saturating_sub(1)..(r + 2).min(rows) {
                for cc in c.saturating_sub(1)..(c + 2).min(cols) {
                    let j = rr * cols + cc;
                    if !seen[j] && s.sign[j] >= gate {
                        seen[j] = true;
                        stack.push(j);
                    }
                }
            }
        }
        let bbox = ((c0 * w / cols) as u32, (r0 * h / rows) as u32, ((c1 + 1) * w / cols) as u32, ((r1 + 1) * h / rows) as u32);
        if let Some(tight) = ink.tighten(bbox) {
            out.push(tight);
        }
    }
    out
}

fn reading_order(crops: &mut [CropPlan]) {
    crops.sort_by_key(|c| (c.bbox.1 / 16, c.bbox.0));
}

/// 앵커 없이 잉크 조각을 모두 읽는 계획
pub fn plan_unfiltered(ink: &InkPage, p: &PlanParams) -> PagePlan {
    let mut crops = Vec::new();
    for block in blocks(ink) {
        let scale = upscale_factor(text_height(ink, block), p.text_px, block.2 - block.0, block.3 - block.1);
        for t in tiles(ink, block, scale, p.max_tile_patches) {
            let Some(t) = ink.text_box(t).or_else(|| ink.tighten(t)) else { continue };
            crops.push(crop_of(ink, p, CropKind::Form, t, None));
        }
    }
    reading_order(&mut crops);
    PagePlan { width: ink.width, height: ink.height, crops, dropped: Vec::new(), unfiltered: true }
}

/// 페이지 한 장의 크롭을 정합니다.
pub fn plan_page(ink: &InkPage, s: &Scores, p: &PlanParams) -> PagePlan {
    let mut crops = Vec::new();
    let mut dropped = Vec::new();
    let mut title: Option<(usize, f32)> = None;
    let contest = p.use_arena.then(|| s.compete(ink));
    for run in blocks(ink).into_iter().flat_map(|b| runs(ink, b)) {
        let Some(bbox) = ink.tighten(run.bbox) else { continue };
        if run.paragraph {
            let guidance = match &contest {
                Some(a) => {
                    let cells = s.cells(ink, bbox);
                    // 경쟁의 기준은 페이지 안의 상대값이라, 근소한 차이는 절대 하한으로 한 번 더 거릅니다.
                    let firm = cells.iter().filter(|&&i| a.owns(GUIDE, i) && a.cells[i].margin >= MIN_CELL_MARGIN).count();
                    !cells.is_empty() && firm as f32 >= cells.len() as f32 * p.drop_share
                }
                None => {
                    let v = tally(s, ink, bbox);
                    v.guide - v.form.max(v.table).max(v.title) >= p.drop_margin
                }
            };
            if guidance {
                dropped.push(bbox);
                continue;
            }
        }
        let scale = upscale_factor(text_height(ink, bbox), p.text_px, bbox.2 - bbox.0, bbox.3 - bbox.1);
        for t in tiles(ink, bbox, scale, p.max_tile_patches) {
            // 괘선만 있는 조각은 읽을 글자가 없습니다.
            let Some(t) = ink.text_box(t) else { continue };
            let v = tally(s, ink, t);
            let table_wins = match &contest {
                Some(a) => {
                    let cells = s.cells(ink, t);
                    territory(a, &cells, TABLE) > territory(a, &cells, FORM)
                }
                None => v.table > v.form,
            };
            let kind = if table_wins { CropKind::Table } else { CropKind::Form };
            if (t.1 as f32) < ink.height as f32 * TITLE_REGION && title.is_none_or(|(_, best)| v.title > best) {
                title = Some((crops.len(), v.title));
            }
            crops.push(crop_of(ink, p, kind, t, Some(v.guide - v.form.max(v.table).max(v.title))));
        }
    }
    if let Some((i, _)) = title {
        crops[i].kind = CropKind::Title;
    }
    suppress_contained(&mut crops);
    for r in sign_regions(s, ink, p.sign_gate) {
        crops.push(crop_of(ink, p, CropKind::Sign, r, None));
    }
    reading_order(&mut crops);
    PagePlan { width: ink.width, height: ink.height, crops, dropped, unfiltered: false }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    /// 위쪽 제목, 가운데 입력 표, 아래쪽 안내문이 있는 페이지와, 그 자리를 그렇게 말하는 점수
    fn page() -> (InkPage, Scores) {
        let mut g = GrayImage::from_pixel(1040, 1456, Luma([250]));
        let mut bar = |x0: u32, x1: u32, y0: u32, h: u32| {
            let mut x = x0;
            while x < x1 {
                for y in y0..y0 + h {
                    for xx in x..(x + 6).min(x1) {
                        g.put_pixel(xx, y, Luma([25]));
                    }
                }
                x += 10;
            }
        };
        bar(300, 740, 100, 48);
        for i in 0..6 {
            bar(120, 920, 320 + i * 60, 28);
        }
        for i in 0..5 {
            bar(120, 920, 1000 + i * 26, 18);
        }
        bar(700, 860, 1330, 40);
        let ink = InkPage::new(&g);
        let (rows, cols) = (14usize, 10usize);
        let zero = vec![0f32; rows * cols];
        let mut s = Scores { rows, cols, form: zero.clone(), sign: zero.clone(), table: zero.clone(), title: zero.clone(), guide: zero };
        for r in 0..rows {
            for c in 0..cols {
                let i = r * cols + c;
                match r {
                    0..=1 => s.title[i] = 0.7,
                    3..=8 => s.form[i] = 0.7,
                    9..=11 => s.guide[i] = 0.7,
                    _ => {
                        s.form[i] = 0.2;
                        if c >= 6 {
                            s.sign[i] = 0.6;
                        }
                    }
                }
            }
        }
        (ink, s)
    }

    fn covers(p: &PagePlan, x: u32, y: u32) -> bool {
        p.crops.iter().any(|c| c.bbox.0 <= x && x < c.bbox.2 && c.bbox.1 <= y && y < c.bbox.3)
    }

    #[test]
    fn guidance_is_dropped_and_everything_else_is_kept() {
        let (ink, s) = page();
        let p = plan_page(&ink, &s, &PlanParams::default());
        assert!(covers(&p, 500, 120), "제목");
        assert!(covers(&p, 500, 500), "입력 표");
        assert!(covers(&p, 750, 1350), "서명");
        assert!(!covers(&p, 500, 1050), "안내문은 버립니다: {:?}", p.crops);
        assert!(!p.dropped.is_empty());
        assert!(!p.unfiltered);
    }

    #[test]
    fn the_title_is_marked_and_reading_order_runs_downwards() {
        let (ink, s) = page();
        let p = plan_page(&ink, &s, &PlanParams::default());
        assert_eq!(p.crops[0].kind, CropKind::Title);
        assert_eq!(p.crops.iter().filter(|c| c.kind == CropKind::Title).count(), 1);
        for w in p.crops.windows(2) {
            assert!(w[0].bbox.1 / 16 <= w[1].bbox.1 / 16, "{:?}", p.crops);
        }
    }

    #[test]
    fn the_signature_gets_its_own_crop() {
        let (ink, s) = page();
        let p = plan_page(&ink, &s, &PlanParams::default());
        let signs: Vec<_> = p.crops.iter().filter(|c| c.kind == CropKind::Sign).collect();
        assert_eq!(signs.len(), 1, "{signs:?}");
        assert!(signs[0].bbox.0 >= 680 && signs[0].bbox.2 <= 880, "{signs:?}");
    }

    #[test]
    fn a_close_race_is_read_rather_than_dropped() {
        let (ink, mut s) = page();
        for i in 9 * 10..12 * 10 {
            s.form[i] = s.guide[i] - MIN_CELL_MARGIN / 2.0;
        }
        assert!(covers(&plan_page(&ink, &s, &PlanParams::default()), 500, 1050), "안내문이 근소하게 앞서면 읽습니다");
        let by_mean = PlanParams { use_arena: false, ..PlanParams::default() };
        for i in 9 * 10..12 * 10 {
            s.form[i] = s.guide[i] - by_mean.drop_margin / 2.0;
        }
        assert!(covers(&plan_page(&ink, &s, &by_mean), 500, 1050), "평균 점수로 고를 때도 같습니다");
    }

    fn crop(kind: CropKind, bbox: Bbox) -> CropPlan {
        CropPlan { kind, bbox, upscale: 1.0, guide_margin: None }
    }

    #[test]
    fn a_crop_inside_a_larger_crop_is_removed() {
        let mut c = vec![crop(CropKind::Form, (0, 0, 100, 100)), crop(CropKind::Form, (10, 10, 50, 50)), crop(CropKind::Form, (0, 90, 100, 200))];
        suppress_contained(&mut c);
        assert_eq!(c.len(), 2, "{c:?}");
        assert_eq!(c[1].bbox, (0, 90, 100, 200), "조금 겹치는 크롭은 남습니다");
    }

    #[test]
    fn the_title_and_signatures_are_never_suppressed() {
        let mut c = vec![crop(CropKind::Form, (0, 0, 100, 100)), crop(CropKind::Title, (10, 10, 50, 50))];
        suppress_contained(&mut c);
        assert_eq!(c.len(), 2);
        let mut c = vec![crop(CropKind::Sign, (0, 0, 100, 100)), crop(CropKind::Form, (10, 10, 50, 50))];
        suppress_contained(&mut c);
        assert_eq!(c.len(), 2, "서명 크롭은 글자로 읽지 않으므로 다른 크롭을 없애면 안 됩니다");
    }

    #[test]
    fn two_identical_crops_become_one() {
        let mut c = vec![crop(CropKind::Form, (0, 0, 50, 50)), crop(CropKind::Table, (0, 0, 50, 50))];
        suppress_contained(&mut c);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn mean_scores_give_the_same_plan_on_a_clear_page() {
        let (ink, s) = page();
        let p = plan_page(&ink, &s, &PlanParams { use_arena: false, ..PlanParams::default() });
        assert!(covers(&p, 500, 120) && covers(&p, 500, 500) && covers(&p, 750, 1350));
        assert!(!covers(&p, 500, 1050), "안내문은 평균 점수로도 버립니다: {:?}", p.crops);
        assert_eq!(p.crops[0].kind, CropKind::Title);
    }

    #[test]
    fn without_anchors_everything_with_ink_is_read() {
        let (ink, _) = page();
        let p = plan_unfiltered(&ink, &PlanParams::default());
        assert!(p.unfiltered && p.dropped.is_empty());
        for (x, y) in [(500, 120), (500, 500), (500, 1050), (750, 1350)] {
            assert!(covers(&p, x, y), "({x}, {y})");
        }
        // 서명 칸도 빠지지 않습니다.
    }

    #[test]
    fn crops_are_padded_inside_the_page_and_small_text_is_enlarged() {
        let (ink, s) = page();
        let p = plan_page(&ink, &s, &PlanParams::default());
        for c in &p.crops {
            assert!(c.bbox.2 <= ink.width && c.bbox.3 <= ink.height);
            assert!(c.upscale >= 1.0);
            assert!(c.patches() > 0);
        }
        assert!(p.kept_area() > 0.05 && p.kept_area() < 1.0, "{}", p.kept_area());
    }
}
