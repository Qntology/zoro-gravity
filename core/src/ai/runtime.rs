//! 앱이 쓰는 모델 묶음. 판단 · 글쓰기 · 문서 읽기가 상주 관리자를 함께 씁니다 (한 번에 한 모델).

use std::sync::Arc;

use anyhow::Result;
use candle_core::Device;

use super::accel;
use super::document::DocumentReader;
use super::model_store::{model_dir, models_dir};
use super::pointer::PointerModel;
use super::residency::{Budget, Residency};
use super::zoro::{open_engine_models, SharedBackbone};

pub struct Runtime {
    /// 표시용 장치 이름
    pub backend: &'static str,
    pub device: Device,
    pub model: PointerModel,
    pub writer: SharedBackbone,
    pub reader: Arc<DocumentReader>,
}

/// 장치를 고르고 모델을 엽니다. 가중치는 쓸 때 단계별로 올라옵니다.
pub fn open() -> Result<Runtime> {
    let device = accel::select_device()?;
    let backend = accel::backend_name(&device);
    let residency = Residency::shared(device.clone(), Budget::auto(&device));
    let (model, writer) = open_engine_models(&model_dir("zoro"), residency.clone())?;
    let reader = Arc::new(DocumentReader::open(&models_dir(), residency)?);
    Ok(Runtime { backend, device, model, writer, reader })
}
