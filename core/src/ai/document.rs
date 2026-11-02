//! 문서 한 장 읽기: 기울기 보정 → SigLIP2 로 읽을 조각 계획 → gocr 로 조각 인식 (명세 3.1 의 Stage 1 ~ 3).
//!
//! 단계마다 필요한 가중치만 올라옵니다. SigLIP2 비전 → (내림) → gocr 비전 → (내림) → gocr 언어 순서이고,
//! 모델 전환은 상주 관리자가 맡습니다. 시각 앵커 파일이 없으면 버리는 조각 없이 잉크 조각을 모두 읽습니다.
//!
//! 여러 장은 [`DocumentReader::read_pages`] 로 함께 읽어야 합니다. 모델을 장마다 오가지 않고 단계마다 한 번만 올립니다.
//! 픽셀 계산은 페이지 · 조각 단위로 병렬이고, 모델 계산은 한 번에 한 묶음입니다.

use std::path::Path;
use std::time::Instant;

use anyhow::{anyhow, Result};
use image::imageops::{resize, FilterType};
use image::{GrayImage, RgbImage};
use rayon::prelude::*;
use serde_json::json;

use super::gocr::image::prepare as gocr_prepare;
use super::gocr::{open as open_gocr, GocrEngine, GocrInput, OcrOutput, Strategy, EARLY_EXIT, TEXT_PROMPT};
use super::residency::SharedResidency;
use super::siglip2::config::DEFAULT_MAX_PATCHES;
use super::siglip2::image::{prepare as siglip2_prepare, PatchInput};
use super::siglip2::ink::{deskew, estimate_skew, Bbox, InkPage};
use super::siglip2::plan::{plan_page, plan_unfiltered, CropKind, CropPlan, PagePlan, PlanParams, Scores};
use super::siglip2::probe::ProbeBank;
use super::siglip2::Siglip2;
use super::trace;

/// 시각 앵커 파일 이름 (siglip2 모델 폴더 안)
pub const PROBE_FILE: &str = "crop_probe.bin";
const TABLE_PROMPT: &str = "Table Recognition:";
/// 조각 하나에서 읽을 최대 토큰 수
const MAX_TOKENS: usize = 1024;
const GOCR_PATCH: usize = 14;
const GOCR_MERGE: usize = 2;

/// 조각 하나를 읽은 결과
#[derive(Debug, Clone, PartialEq)]
pub struct CropReading {
    pub kind: CropKind,
    /// 곧게 편 페이지의 좌표
    pub bbox: Bbox,
    pub output: OcrOutput,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Timings {
    pub layout_ms: f64,
    pub siglip2_ms: f64,
    pub ocr_ms: f64,
}

#[derive(Debug, Clone)]
pub struct PageReading {
    /// 곧게 펴려고 돌린 각도 (도)
    pub skew: f32,
    pub plan: PagePlan,
    pub crops: Vec<CropReading>,
    pub timings: Timings,
}

/// 읽는 방식
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReadOptions {
    pub plan: PlanParams,
    /// 입력란도 표 프롬프트로 읽습니다. 항목과 값이 칸으로 묶여 나오고, 값을 건너뛰는 일이 줄어듭니다.
    pub forms_as_tables: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self { plan: PlanParams::default(), forms_as_tables: true }
    }
}

pub struct DocumentReader {
    siglip2: Siglip2,
    probes: Option<ProbeBank>,
    gocr: GocrEngine,
    options: ReadOptions,
}

/// 이미지의 상자를 잘라 배율만큼 키웁니다.
pub fn cut(img: &RgbImage, (x0, y0, x1, y1): Bbox, upscale: f32) -> RgbImage {
    let piece = image::imageops::crop_imm(img, x0, y0, x1 - x0, y1 - y0).to_image();
    if upscale <= 1.01 {
        return piece;
    }
    let (w, h) = (((x1 - x0) as f32 * upscale).round() as u32, ((y1 - y0) as f32 * upscale).round() as u32);
    resize(&piece, w.max(1), h.max(1), FilterType::Lanczos3)
}

fn gray_of(img: &RgbImage) -> GrayImage {
    GrayImage::from_fn(img.width(), img.height(), |x, y| {
        let p = img.get_pixel(x, y).0;
        image::Luma([((p[0] as u32 * 77 + p[1] as u32 * 150 + p[2] as u32 * 29) >> 8) as u8])
    })
}

impl DocumentReader {
    /// `models_dir` 아래의 `siglip2` · `gocr` 모델을 엽니다. 가중치는 읽을 때 단계별로 올라옵니다.
    pub fn open(models_dir: &Path, residency: SharedResidency) -> Result<Self> {
        let dir = models_dir.join("siglip2");
        let siglip2 = Siglip2::open(&dir, residency.clone())?;
        let probe = dir.join(PROBE_FILE);
        let probes = probe.exists().then(|| ProbeBank::from_file(&probe)).transpose()?;
        let gocr = open_gocr(&models_dir.join("gocr"), residency, MAX_TOKENS, Strategy::Speculative { early_exit: Some(EARLY_EXIT) })?;
        Ok(Self { siglip2, probes, gocr, options: ReadOptions::default() })
    }

    pub fn with_options(mut self, options: ReadOptions) -> Self {
        self.options = options;
        self
    }

    /// 페이지를 곧게 펴고 읽을 조각을 정합니다. SigLIP2 비전 타워만 씁니다.
    pub fn plan(&self, page: &RgbImage) -> Result<(RgbImage, f32, PagePlan, Timings)> {
        self.plan_pages(std::slice::from_ref(page))?.pop().ok_or_else(|| anyhow!("페이지 결과가 없습니다"))
    }

    /// 픽셀 계산은 페이지마다 병렬로 하고, SigLIP2 는 격자가 같은 페이지를 묶어 한 번만 올려 계산합니다.
    fn plan_pages(&self, pages: &[RgbImage]) -> Result<Vec<(RgbImage, f32, PagePlan, Timings)>> {
        let t = Instant::now();
        let straight: Vec<(RgbImage, f32, InkPage)> = pages
            .par_iter()
            .map(|page| {
                let skew = estimate_skew(&gray_of(page));
                let image = deskew(page, skew);
                let ink = InkPage::new(&gray_of(&image));
                (image, skew, ink)
            })
            .collect();
        let per_page = |t: Instant| t.elapsed().as_secs_f64() * 1e3 / pages.len().max(1) as f64;
        let layout_ms = per_page(t);
        trace::record("규칙", "기울기 보정 · 잉크 분석", t.elapsed().as_secs_f64() * 1e3, json!({ "pages": pages.len() }));

        let Some(bank) = &self.probes else {
            let plan = |(image, skew, ink): (RgbImage, f32, InkPage)| {
                let plan = plan_unfiltered(&ink, &self.options.plan);
                (image, skew, plan, Timings { layout_ms, ..Timings::default() })
            };
            return Ok(straight.into_iter().map(plan).collect());
        };
        let t = Instant::now();
        let patch = self.siglip2.config().patch;
        let inputs: Vec<PatchInput> = straight.par_iter().map(|(image, ..)| siglip2_prepare(image, patch, DEFAULT_MAX_PATCHES)).collect::<Result<_>>()?;
        let grids = self.siglip2.encode_images(&inputs)?;
        let siglip2_ms = per_page(t);
        trace::record("siglip2", "패치 특징", t.elapsed().as_secs_f64() * 1e3, json!({ "pages": pages.len(), "patches": inputs.iter().map(|i| i.rows * i.cols).sum::<usize>() }));
        let t = Instant::now();
        let planned: Vec<(RgbImage, f32, PagePlan)> = straight
            .into_par_iter()
            .zip(grids)
            .map(|((image, skew, ink), grid)| Ok((image, skew, plan_page(&ink, &Scores::new(bank, &grid)?, &self.options.plan))))
            .collect::<Result<_>>()?;
        let layout_ms = layout_ms + per_page(t);
        if trace::enabled() {
            let pages: Vec<_> = planned
                .iter()
                .map(|(_, skew, plan)| json!({ "skew": skew, "crops": plan.crops.iter().map(|c| json!({ "kind": c.kind.name(), "bbox": [c.bbox.0, c.bbox.1, c.bbox.2, c.bbox.3] })).collect::<Vec<_>>() }))
                .collect();
            trace::record("규칙", "조각 계획", t.elapsed().as_secs_f64() * 1e3, json!({ "pages": pages }));
        }
        Ok(planned.into_iter().map(|(image, skew, plan)| (image, skew, plan, Timings { layout_ms, siglip2_ms, ocr_ms: 0.0 })).collect())
    }

    fn prompt_of(&self, kind: CropKind) -> &'static str {
        match kind {
            CropKind::Table => TABLE_PROMPT,
            CropKind::Form if self.options.forms_as_tables => TABLE_PROMPT,
            _ => TEXT_PROMPT,
        }
    }

    pub fn read(&self, page: &RgbImage) -> Result<PageReading> {
        let mut out = self.read_pages(std::slice::from_ref(page))?;
        out.pop().ok_or_else(|| anyhow!("페이지 결과가 없습니다"))
    }

    /// 페이지들을 읽습니다. 모델을 페이지마다 오가지 않도록 계획을 모두 세운 뒤 조각을 한꺼번에 읽습니다.
    /// 서명 조각은 글자로 읽지 않고 좌표만 계획에 남깁니다.
    pub fn read_pages(&self, pages: &[RgbImage]) -> Result<Vec<PageReading>> {
        let planned = self.plan_pages(pages)?;
        let t = Instant::now();
        let crops: Vec<(&RgbImage, &CropPlan)> = planned.iter().flat_map(|(straight, _, plan, _)| plan.crops.iter().filter(|c| c.kind != CropKind::Sign).map(move |c| (straight, c))).collect();
        let inputs: Vec<GocrInput> = crops
            .into_par_iter()
            .map(|(straight, c)| Ok(GocrInput { image: gocr_prepare(&cut(straight, c.bbox, c.upscale), GOCR_PATCH, GOCR_MERGE)?, prompt: self.prompt_of(c.kind).to_string() }))
            .collect::<Result<_>>()?;
        let outputs = self.gocr.recognize(&inputs)?;
        let ocr_ms = t.elapsed().as_secs_f64() * 1e3 / pages.len().max(1) as f64;
        if trace::enabled() {
            let texts: Vec<_> = if trace::with_text() { outputs.iter().map(|o| json!({ "text": o.text, "conf": o.min_confidence })).collect() } else { Vec::new() };
            let detail = json!({
                "crops": outputs.len(),
                "tokens": outputs.iter().map(|o| o.tokens.len()).sum::<usize>(),
                "image_tokens": outputs.iter().map(|o| o.image_tokens).sum::<usize>(),
                "drafts": outputs.iter().map(|o| o.drafts).sum::<usize>(),
                "verifies": outputs.iter().map(|o| o.verifies).sum::<usize>(),
                "texts": texts,
            });
            trace::record("gocr", "결과", 0.0, detail);
        }
        let mut outputs = outputs.into_iter();
        Ok(planned
            .into_iter()
            .map(|(_, skew, plan, mut timings)| {
                timings.ocr_ms = ocr_ms;
                let crops = plan
                    .crops
                    .iter()
                    .filter(|c| c.kind != CropKind::Sign)
                    .zip(outputs.by_ref())
                    .map(|(c, output)| CropReading { kind: c.kind, bbox: c.bbox, output })
                    .collect();
                PageReading { skew, plan, crops, timings }
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    #[test]
    fn cutting_keeps_the_box_and_enlarges_on_request() {
        let mut img = RgbImage::from_pixel(100, 80, Rgb([255, 255, 255]));
        img.put_pixel(20, 10, Rgb([0, 0, 0]));
        let piece = cut(&img, (20, 10, 60, 30), 1.0);
        assert_eq!(piece.dimensions(), (40, 20));
        assert_eq!(piece.get_pixel(0, 0), &Rgb([0, 0, 0]));
        assert_eq!(cut(&img, (20, 10, 60, 30), 2.5).dimensions(), (100, 50));
    }

    #[test]
    fn grayscale_matches_the_usual_weights() {
        let img = RgbImage::from_pixel(1, 1, Rgb([255, 0, 0]));
        let v = gray_of(&img).get_pixel(0, 0)[0];
        assert!((75..=78).contains(&v), "{v}");
    }
}
