//! 실제 모델 파일(모델 폴더의 `zoro`, `gocr`, `siglip2`)과 실제 장치로 하는 점검.
//! 모델 파일과 GPU 가 있어야 하고 오래 걸려서 기본 실행에서는 건너뜁니다.
//!
//!   cargo test -p zorogravity-core --test real_models -- --ignored --nocapture --test-threads 1
//!   ZOROGRAVITY_DEVICE=cpu cargo test -p zorogravity-core --test real_models -- --ignored --nocapture --test-threads 1
//!
//! 여러 모델을 한 장치에 올리는 점검이 있어서 병렬로 돌리면 안 됩니다 (`--test-threads 1`).

use std::sync::Arc;

use anyhow::Result;
use candle_core::quantized::ggml_file::qtensor_from_ggml;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{Device, Module};
use chrono::{NaiveDate, TimeZone, Utc};
use zorogravity_core::case::{BankFacts, CaseFile, Category, Excuse, LeaseFacts, Link, Node, Person, Relation, ScannedDoc, Signature};
use zorogravity_core::drafting::{normalize_excuse, summarize_grievance};
use zorogravity_core::policy::AutomationPolicy;
use zorogravity_core::privacy::node_hash;
use zorogravity_core::state_machine::{CaseOutcome, Engine};
use zorogravity_core::verdict::Verdict;
use zorogravity_core::ai::document::DocumentReader;
use zorogravity_core::ai::jev_head::JevHead;
use zorogravity_core::ai::model_store::{model_dir, models_dir};
use zorogravity_core::ai::ocr::{LanguageDecoder, LanguageWeights, OcrEngine, PhaseCtx, VisionEncoder};
use zorogravity_core::ai::pointer::{PointerModel, PointerPrompt};
use zorogravity_core::ai::residency::{Action, Budget, DeviceMemoryShort, Group, Loaded, ModelId, Phase, Residency, SharedResidency};
use zorogravity_core::ai::siglip2::image::prepare as siglip2_prepare;
use zorogravity_core::ai::siglip2::plan::CropKind;
use zorogravity_core::ai::siglip2::Siglip2;
use zorogravity_core::ai::speculative::{BlockDiffusion, Speculation};
use zorogravity_core::ai::zoro::model::StreamMode;
use zorogravity_core::ai::tokenizer::token_covering;
use zorogravity_core::ai::zoro::ZoroBackbone;
use zorogravity_core::ai::{accel, verify};
use zorogravity_core::decision::{DecisionModel, DecisionTask, Fact};
use zorogravity_core::state_machine::gate3_substantive::OPTIONS;
use zorogravity_core::verdict::ALL;

/// `python scripts/gen_sample_forms.py out/forms_test 4 1234` 로 만든 합성 서식
const SAMPLE_FORM: &str = "../out/forms_test/rent_application_0.jpg";

fn residency() -> SharedResidency {
    let device = accel::select_device().expect("장치");
    Residency::shared(device.clone(), Budget::auto(&device))
}

fn gate3_task(path: &str, homonym: &str) -> DecisionTask {
    DecisionTask {
        title: "임차 요건 심사".into(),
        facts: vec![
            Fact::new("적용 규칙", "임차인과 임대인이 2촌 이내 친족이거나 같은 세대·같은 계좌이면 지급 제외"),
            Fact::new("가까운 연결 경로", path),
            Fact::new("동명이인 후보", homonym),
        ],
        options: OPTIONS.to_vec(),
    }
}

fn pointer_model(res: SharedResidency, mode: StreamMode) -> (PointerModel, Arc<ZoroBackbone>) {
    struct Shared(Arc<ZoroBackbone>);
    impl zorogravity_core::ai::pointer::Backbone for Shared {
        fn encode(&self, p: &PointerPrompt) -> Result<zorogravity_core::ai::pointer::PointerHidden> {
            self.0.encode(p)
        }
    }
    let backbone = Arc::new(ZoroBackbone::open_with(&model_dir("zoro"), res, mode).expect("zoro 열기"));
    let head = JevHead::from_safetensors(&model_dir("zoro").join("jev_head.safetensors")).expect("jev_head");
    (PointerModel::new(Box::new(Shared(Arc::clone(&backbone))), head), backbone)
}

#[test]
#[ignore]
fn zoro_continues_a_sentence_sensibly() {
    // 구조에 대한 가정(정규화 방식 · 출력 게이트 · RoPE · 선형 주의)이 맞으면 다음 토큰이 말이 됩니다.
    let backbone = ZoroBackbone::open(&model_dir("zoro"), residency()).expect("zoro 열기");
    let enc = backbone.tokenizer().encode("The capital of France is").unwrap();
    let hidden = backbone.model().hidden_states(&enc.ids).unwrap();
    let last = hidden.narrow(0, enc.ids.len() - 1, 1).unwrap().to_device(&Device::Cpu).unwrap();
    // 출력 투영은 임베딩 표와 공유합니다. 점검에서만 표 전체를 CPU 에 읽습니다.
    let emb = backbone.model().spec().file.read_qtensor("token_embd.weight", &Device::Cpu).unwrap();
    let logits = QMatMul::from_qtensor(emb).unwrap().forward(&last).unwrap().squeeze(0).unwrap().to_vec1::<f32>().unwrap();
    let best = (0..logits.len()).max_by(|a, b| logits[*a].total_cmp(&logits[*b])).unwrap();
    let max = logits[best];
    let prob = 1.0 / logits.iter().map(|l| (l - max).exp()).sum::<f32>();
    let word = backbone.tokenizer().decode(&[best as u32]).unwrap();
    println!("다음 토큰 {word:?} ({:.1}%)", prob * 100.0);
    assert!(word.contains("Paris"), "다음 토큰이 {word:?} 입니다");
    assert!(prob > 0.2, "확률 {prob}");
}

#[test]
#[ignore]
fn prompt_positions_land_on_the_quote_closing_every_option_code() {
    let backbone = ZoroBackbone::open(&model_dir("zoro"), residency()).expect("zoro 열기");
    // 20개 선택지를 모두 넣은 질문
    let task = DecisionTask { title: "전체 선택지".into(), facts: vec![Fact::new("메모", "한글 설명 ✓")], options: ALL.to_vec() };
    let prompt = PointerPrompt::render(&task);
    let enc = backbone.tokenizer().encode(&prompt.text).unwrap();
    let q = token_covering(&enc.offsets, prompt.query_byte).expect("질의 토큰");
    assert_eq!(q, enc.ids.len() - 1, "질의는 프롬프트의 마지막 토큰");
    let upto = |i: usize| backbone.tokenizer().decode(&enc.ids[..=i]).unwrap();
    assert!(upto(q).ends_with("<|im_start|>assistant\n"));
    for (i, v) in ALL.iter().enumerate() {
        let idx = token_covering(&enc.offsets, prompt.option_bytes[i]).expect("선택지 토큰");
        let read = upto(idx);
        let closing = read.rfind('"').expect("닫는 따옴표");
        assert!(read[..closing].ends_with(v.code()), "선택지 {i} ({}): 위치까지 읽은 글이 코드와 닫는 따옴표로 끝나야 합니다", v.code());
    }
}

#[test]
#[ignore]
fn decisions_are_deterministic_follow_the_facts_and_load_only_text_weights() {
    let res = residency();
    let (model, _bb) = pointer_model(res.clone(), StreamMode::from_env().unwrap());
    let linked = gate3_task("있음 (1홉)", "아니오");
    let a = model.probabilities(&linked).unwrap();
    let b = model.probabilities(&linked).unwrap();
    for (x, y) in a.iter().zip(&b) {
        assert!((x - y).abs() < 1e-5, "같은 질문의 확률이 달라졌습니다: {a:?} {b:?}");
    }
    assert!((a.iter().sum::<f32>() - 1.0).abs() < 1e-4);
    // 간접 증거: 연결이 있으면 위반 쪽, 없으면 정상 쪽으로 움직입니다 (위치 · 스케일 약속이 맞다는 증명은 아닙니다).
    let none = model.probabilities(&gate3_task("없음", "아니오")).unwrap();
    println!("연결 있음 {a:?} · 연결 없음 {none:?}");
    assert!(a[2] > 0.9, "연결이 있는데 위반 확률이 {:.3}", a[2]);
    assert!(none[2] < 0.1, "연결이 없는데 위반 확률이 {:.3}", none[2]);

    // 쓰지 않는 가중치는 올라온 적이 없습니다.
    let r = res.lock().unwrap();
    let loaded: Vec<Group> = r.history().iter().filter(|t| t.action == Action::Load).map(|t| t.group).collect();
    assert!(!loaded.contains(&Group::Vision), "비전이 올라왔습니다: {loaded:?}");
    assert!(!loaded.contains(&Group::Embedding), "임베딩 표 전체가 올라왔습니다: {loaded:?}");
    assert_eq!(r.resident_model(), Some(ModelId::Zoro));
}

#[test]
#[ignore]
fn streaming_blocks_in_chunks_gives_the_same_decision_as_keeping_them_resident() {
    let res = residency();
    let task = gate3_task("없음", "예 (같은 이름이 연결망에 있으나 다른 사람으로 판별)");
    let (resident, _a) = pointer_model(res.clone(), StreamMode::Resident);
    let want = resident.probabilities(&task).unwrap();
    drop(resident);
    let (streamed, b) = pointer_model(res.clone(), StreamMode::Chunks(8));
    assert_eq!(b.model().plan().len(), 4, "블록 32 개를 8 개씩 4 구간");
    let got = streamed.probabilities(&task).unwrap();
    for (x, y) in want.iter().zip(&got) {
        assert!((x - y).abs() < 1e-4, "상주 {want:?} / 스트리밍 {got:?}");
    }
    // 스트리밍을 시작한 뒤(8개짜리 첫 구간을 올린 시점부터)의 기록만 봅니다. 두 구간이 함께 올라온 적이 없어야 합니다.
    let r = res.lock().unwrap();
    let history = r.history();
    let first = history
        .iter()
        .position(|t| t.action == Action::Load && t.group == Group::blocks(0..8))
        .expect("스트리밍 첫 구간을 올린 기록");
    // 시작 시점에는 상주 때 올라온 전체 블록이 이미 내려가 있어야 합니다 (기록 순서: 내림 → 올림).
    let mut live = 0u64;
    for t in &history[..first] {
        match t.action {
            Action::Load => live += t.bytes,
            Action::Release => live -= t.bytes,
        }
    }
    assert_eq!(live, 0, "스트리밍 첫 구간을 올리기 전에 이전 상주분이 모두 내려가야 합니다");
    let chunk_max = history[first..].iter().filter(|t| t.action == Action::Load).map(|t| t.bytes).max().unwrap();
    let mut peak = 0u64;
    for t in &history[first..] {
        match t.action {
            Action::Load => live += t.bytes,
            Action::Release => live -= t.bytes,
        }
        peak = peak.max(live);
    }
    println!("구간 하나의 최대 크기 {:.0} MB · 스트리밍 중 최대 상주 {:.0} MB", chunk_max as f64 / 1048576.0, peak as f64 / 1048576.0);
    assert!(peak <= chunk_max + (1 << 20), "스트리밍 중 두 구간 이상이 함께 올라왔습니다 (최대 {peak} > 구간 {chunk_max})");
    assert!(chunk_max < 1200 << 20, "8 블록 구간이 전체 블록보다 훨씬 작아야 합니다: {chunk_max}");
}

#[test]
#[ignore]
fn the_gpu_and_the_cpu_decide_alike() {
    let device = accel::select_device().expect("장치");
    if !accel::is_gpu(&device) {
        println!("GPU 가 아니라서 CPU 와 비교할 것이 없습니다");
        return;
    }
    let task = gate3_task("있음 (2홉)", "아니오");
    let gpu = {
        let res = Residency::shared(device.clone(), Budget::auto(&device));
        let (m, _b) = pointer_model(res.clone(), StreamMode::Auto);
        let p = m.probabilities(&task).unwrap();
        drop(m);
        res.lock().unwrap().release_all().unwrap();
        p
    };
    let cpu = {
        let res = Residency::shared(Device::Cpu, Budget::unlimited());
        let (m, _b) = pointer_model(res, StreamMode::Auto);
        m.probabilities(&task).unwrap()
    };
    println!("GPU {gpu:?} · CPU {cpu:?}");
    // CPU 의 Q8_0 행렬곱은 활성값도 8비트로 줄입니다. 확률이 중간값이면 그 차이가 수 % 포인트로 보입니다.
    for (g, c) in gpu.iter().zip(&cpu) {
        assert!((g - c).abs() < 0.1, "GPU 와 CPU 의 확률이 크게 다릅니다: {gpu:?} / {cpu:?}");
    }
    let best = |p: &[f32]| p.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
    assert_eq!(best(&gpu), best(&cpu));
    let _ = verify::checks_enabled();
}

/// 비전 단계에서 비전 묶음의 크기만 돌려주는 점검용 인코더
struct VisionProbe;
impl VisionEncoder for VisionProbe {
    type Input = ();
    type Features = u64;
    fn encode(&self, ctx: &PhaseCtx<'_>, vision: &Loaded, _: &()) -> Result<u64> {
        assert_eq!(ctx.residency.resident_groups(), vec![Group::Vision]);
        assert!(vision.has("model.visual.patch_embed.proj.weight"));
        Ok(vision.device_bytes())
    }
}

/// 언어 단계에서 언어 묶음의 크기 합을 돌려주는 점검용 디코더
struct LanguageProbe;
impl LanguageDecoder for LanguageProbe {
    type Features = u64;
    type Output = (u64, u64);
    fn decode(&self, ctx: &PhaseCtx<'_>, language: &LanguageWeights, vision_bytes: u64) -> Result<(u64, u64)> {
        assert!(!ctx.residency.resident_groups().contains(&Group::Vision), "디코딩 중에는 비전이 내려가 있어야 합니다");
        assert!(language.blocks.has("blk.15.attn_q.weight"));
        Ok((vision_bytes, language.blocks.device_bytes() + language.output.device_bytes()))
    }
}

#[test]
#[ignore]
fn zoro_and_the_ocr_phases_take_turns_on_one_device() {
    let res = residency();
    let (model, _bb) = pointer_model(res.clone(), StreamMode::Auto);
    let task = gate3_task("있음 (1홉)", "아니오");
    let before = model.probabilities(&task).unwrap();

    let ocr = OcrEngine::open(&model_dir("gocr"), res.clone(), VisionProbe, LanguageProbe).expect("gocr 열기");
    let out = ocr.recognize(&[(), ()]).unwrap();
    println!("비전 {:.0} MB · 언어 {:.0} MB", out[0].0 as f64 / 1048576.0, out[0].1 as f64 / 1048576.0);
    assert_eq!(out.len(), 2);
    assert_eq!(res.lock().unwrap().resident_model(), Some(ModelId::Gocr));

    let after = model.probabilities(&task).unwrap();
    for (x, y) in before.iter().zip(&after) {
        assert!((x - y).abs() < 1e-4, "OCR 를 거친 뒤 판단이 달라졌습니다: {before:?} / {after:?}");
    }
    assert_eq!(res.lock().unwrap().resident_model(), Some(ModelId::Zoro));

    // 모든 시점에서 두 모델이 함께 올라온 적이 없습니다 (올림 · 내림 기록을 되짚음).
    let r = res.lock().unwrap();
    let mut live = std::collections::HashMap::<ModelId, u64>::new();
    for t in r.history() {
        let e = live.entry(t.model).or_default();
        match t.action {
            Action::Load => *e += t.bytes,
            Action::Release => *e -= t.bytes,
        }
        let resident: Vec<_> = live.iter().filter(|(_, b)| **b > 0).map(|(m, _)| *m).collect();
        assert!(resident.len() <= 1, "동시 상주: {resident:?}");
    }
}

// ---- 실제 모델을 엔진에 꽂아서 ----

const SALT: &[u8] = b"test-salt-not-for-production";

#[test]
#[ignore]
fn siglip2_finds_the_document_and_keeps_each_tower_apart() -> Result<()> {
    let res = residency();
    let model = Siglip2::open(&model_dir("siglip2"), res.clone())?;
    let img = image::open(SAMPLE_FORM)?.to_rgb8();
    let input = siglip2_prepare(&img, model.config().patch, 256)?;
    assert!(input.n_patches() <= 256);
    let grid = model.encode_image(&input)?;
    assert_eq!(res.lock().unwrap().resident_groups(), vec![Group::Vision], "문구 앵커 없이 페이지만 읽을 때는 비전 타워만 올라옵니다");
    let texts = ["a korean document".to_string(), "a photo of a cat".to_string()];
    let v = model.encode_phrases(&texts)?;
    assert_eq!(res.lock().unwrap().resident_groups(), vec![Group::Text], "문구를 읽을 때는 텍스트 타워만 올라옵니다");
    let cos = |t: &[f32]| grid.pooled.iter().zip(t).map(|(a, b)| a * b).sum::<f32>();
    assert!(cos(&v[0]) > cos(&v[1]) + 0.05, "서류: {} · 고양이: {}", cos(&v[0]), cos(&v[1]));
    Ok(())
}

#[test]
#[ignore]
fn a_page_is_cropped_and_read_with_one_model_at_a_time() -> Result<()> {
    let res = residency();
    let reader = DocumentReader::open(&models_dir(), res.clone())?;
    let img = image::open(SAMPLE_FORM)?.to_rgb8();
    let page = reader.read(&img)?;
    println!("조각 {} · 면적 {:.2} · OCR {:.1} s", page.crops.len(), page.plan.kept_area(), page.timings.ocr_ms / 1e3);
    assert!(page.plan.kept_area() < 0.7, "여백은 읽지 않습니다: {}", page.plan.kept_area());
    let title = page.crops.iter().find(|c| c.kind == CropKind::Title).expect("제목 조각");
    assert!(title.output.text.contains("신청서"), "제목: {}", title.output.text);
    assert!(page.plan.crops.iter().any(|c| c.kind == CropKind::Sign), "서명 조각이 있어야 합니다");

    // 모델은 SigLIP2 비전 → gocr 비전 → gocr 언어 순서로만 올라오고, 앞의 것은 그 전에 내려갑니다.
    let r = res.lock().unwrap();
    let mut resident: Vec<(ModelId, Group)> = Vec::new();
    let mut order: Vec<(ModelId, Group)> = Vec::new();
    for t in r.history() {
        match t.action {
            Action::Load => {
                assert!(resident.iter().all(|(m, _)| *m == t.model), "모델이 겹쳤습니다: {resident:?} + {}", t.model);
                resident.push((t.model, t.group));
                order.push((t.model, t.group));
            }
            Action::Release => resident.retain(|e| *e != (t.model, t.group)),
        }
    }
    assert_eq!(order[0], (ModelId::Siglip2, Group::Vision));
    assert_eq!(order[1], (ModelId::Gocr, Group::Vision));
    assert!(matches!(order[2], (ModelId::Gocr, Group::Blocks { .. })), "{order:?}");
    assert!(!order.iter().any(|(_, g)| *g == Group::Text), "실행 때 텍스트 타워는 필요 없습니다");
    Ok(())
}

#[test]
#[ignore]
fn pages_read_together_match_pages_read_one_by_one() -> Result<()> {
    // 격자가 같은 조각은 묶이고, 모델은 묶음 전체에서 단계마다 한 번만 올라옵니다.
    let res = residency();
    let reader = DocumentReader::open(&models_dir(), res.clone())?;
    let pages: Vec<image::RgbImage> = [SAMPLE_FORM, "../out/forms_test/rent_application_1.jpg", "../out/forms_test/bankbook_0.jpg"]
        .iter()
        .map(|p| Ok(image::open(p)?.to_rgb8()))
        .collect::<Result<_>>()?;
    let before = res.lock().unwrap().history().len();
    let together = reader.read_pages(&pages)?;
    let loads = res.lock().unwrap().history()[before..].iter().filter(|t| t.action == Action::Load && t.group == Group::Vision).count();
    assert_eq!(loads, 2, "SigLIP2 비전과 gocr 비전이 한 번씩만 올라옵니다");
    let texts = |r: &zorogravity_core::ai::document::PageReading| -> Vec<(CropKind, String)> { r.crops.iter().map(|c| (c.kind, c.output.text.clone())).collect() };
    for (page, got) in pages.iter().zip(&together) {
        assert_eq!(texts(got), texts(&reader.read(page)?));
    }
    Ok(())
}

fn clean_welfare(id: &str, relations: Vec<Relation>) -> CaseFile {
    let doc = |kind: &str, hash: &str| ScannedDoc { kind: kind.into(), content_hash: hash.into(), pages: 1 };
    CaseFile {
        case_id: id.into(),
        category: Category::Welfare,
        docs: vec![doc("신청서", "h1"), doc("임대차계약서", "h2"), doc("이체내역서", "h3"), doc("금융정보동의서", "h4")],
        applicant: Person { name: "김철수".into(), birth_prefix: "990101".into(), identifier: "9901011234567".into() },
        landlord: Some(Person { name: "박영희".into(), birth_prefix: "640312".into(), identifier: "6403122345678".into() }),
        lease: Some(LeaseFacts { applied_rent: 500_000, contract_rent: 500_000, latest_transfer: 500_000 }),
        bank: Some(BankFacts { handwritten_account: "110-123-991023".into(), printed_account: "110-123-991023".into() }),
        signature: Some(Signature { stroke_count: 8, bbox_area_ratio: 0.42, single_line: false }),
        poa_written_on: None,
        today: NaiveDate::from_ymd_opt(2026, 10, 31).unwrap(),
        excuse: None,
        relations,
        supplement_count: 0,
        grievance_text: None,
    }
}

#[test]
#[ignore]
fn the_zoro_vision_tower_lives_in_its_own_file() -> Result<()> {
    let res = residency();
    let zoro = ZoroBackbone::open(&model_dir("zoro"), res.clone())?;
    let spec = zoro.model().spec();
    assert!(spec.file.names_with_prefix("model.visual.").is_empty(), "텍스트 파일에 비전 텐서가 없어야 합니다");
    let mmproj = spec.mmproj.as_ref().expect("mmproj 파일");
    assert_eq!(mmproj.meta_str("general.type")?, "mmproj");
    assert!(mmproj.names_with_prefix("blk.").is_empty(), "mmproj 에 텍스트 텐서가 없어야 합니다");
    let handles = res.lock().unwrap().enter(spec, &Phase::new("zoro:비전", [Group::Vision]))?;
    assert_eq!(handles[0].len(), mmproj.tensor_count());
    Ok(())
}

#[test]
#[ignore]
fn speculation_writes_the_same_text_with_fewer_reads() -> Result<()> {
    let res = residency();
    let zoro = ZoroBackbone::open(&model_dir("zoro"), res.clone())?;
    let plan = Speculation { masks: 2, lookup: 2, width: 3 };
    let file = &zoro.model().spec().file;
    let diffusion = BlockDiffusion::from_file(file)?.expect("텍스트 GGUF 에 블록 디퓨전 설정이 있어야 합니다");
    assert_eq!(diffusion, BlockDiffusion { bd_size: 8, mask_id: 248077 });
    let system = "민원 담당 공무원입니다. 민원 글의 핵심을 한 문장으로 요약합니다.";
    let (mut reads, mut tokens) = (0, 0);
    for user in [
        "우리 동네 앞 도로에 지난달부터 지름 50센티미터 정도의 구멍이 패여 있습니다. 빨리 포장 공사를 해 주십시오.",
        "재산세 고지서를 받았는데 작년보다 12만 원이 더 나왔습니다. 면적이 잘못 적힌 것 같습니다.",
    ] {
        let plain = zoro.write_with(system, user, 48, Speculation::NONE)?;
        let guessed = zoro.write_with(system, user, 48, plan)?;
        assert_eq!(guessed.text, plain.text, "추측은 글을 바꾸지 않습니다");
        assert_eq!(guessed.finished, plain.finished);
        assert!((guessed.min_confidence - plain.min_confidence).abs() < 1e-3);
        assert_eq!(plain.reads, plain.tokens, "추측이 없으면 한 토큰씩 읽습니다");
        (reads, tokens) = (reads + guessed.reads, tokens + guessed.tokens);
    }
    assert!(reads < tokens, "순전파 {reads}번 · 토큰 {tokens}개");
    let groups = res.lock().unwrap().resident_groups();
    assert!(!groups.contains(&Group::Vision), "글쓰기에는 비전 타워를 올리지 않습니다: {groups:?}");
    Ok(())
}

#[test]
#[ignore]
fn zoro_writes_a_checked_summary_and_the_same_words_come_out_twice() -> Result<()> {
    let zoro = ZoroBackbone::open(&model_dir("zoro"), residency())?;
    let text = "재산세 고지서를 받았는데 작년보다 12만 원이 더 나왔습니다. 면적이 잘못 적힌 것 같습니다. 고지 금액을 확인해 주십시오.";
    let s = summarize_grievance(&zoro, text)?;
    println!("{s:?}");
    assert!(s.verified, "{:?}", s.problems);
    assert_eq!(s.department.as_deref(), Some("세무과"));
    assert_eq!(summarize_grievance(&zoro, text)?, s, "같은 입력이면 같은 글이 나옵니다");

    // 명세 Case 2-C: 영수증을 말하지 않은 소명은 영수증을 요구하는 쪽으로 갑니다.
    let excuse = normalize_excuse(&zoro, "지난달에 보일러가 고장 나서 수리비 20만 원을 제가 먼저 내고 월세에서 깠어요")?;
    assert_eq!(excuse, Some(Excuse::RepairDeduction { has_receipt: false }));
    Ok(())
}

#[test]
#[ignore]
fn the_engine_with_the_real_model_never_finalizes_a_violation() {
    let model = zorogravity_core::ai::zoro::open_decision_model(&model_dir("zoro"), residency()).expect("판단 모델");
    let mut e = Engine::new(SALT, AutomationPolicy::default(), Some(Box::new(model)));
    let now = Utc.with_ymd_and_hms(2026, 10, 31, 9, 0, 0).unwrap();
    e.enable_automation("officer-17", "지침 제2조 (테스트용 법령 근거)", now).unwrap();

    // 모친이 임대인 본인인 위장 임차: 규칙이 위반을 세우고, 모델도 위반에 동의해도 불리한 처분은 자동 확정되지 않습니다.
    let link = Relation {
        a: Node { hash: node_hash(SALT, "9901011234567"), name: "김철수".into() },
        b: Node { hash: node_hash(SALT, "6403122345678"), name: "박영희".into() },
        link: Link::Family { degree: 1 },
    };
    match e.run(clean_welfare("real-violation", vec![link]), now).unwrap() {
        CaseOutcome::OfficerReview { judgement: Some(j), .. } => {
            println!("위반 사례: 판정 {:?} · 모델 확신도 {:?}", j.verdict, j.confidence);
            assert_eq!(j.verdict, Verdict::G3ViolationHighRisk);
            assert!(j.confidence.unwrap_or(0.0) > 0.9, "실제 모델이 위반에 동의해야 합니다: {:?}", j.confidence);
        }
        other => panic!("불리한 처분은 공무원 확인으로 남아야 합니다: {other:?}"),
    }
    assert_eq!(e.store.archive().len(), 0);

    // 문제 없는 사례: 모델 확신도가 기준(0.95) 이상이면 자동 확정, 미만이면 공무원 확인입니다. 어느 쪽이든 안전한 경로여야 합니다.
    match e.run(clean_welfare("real-clean", vec![]), now).unwrap() {
        CaseOutcome::AutoFinalized { .. } => println!("정상 사례: 자동 확정"),
        CaseOutcome::OfficerReview { judgement: Some(j), .. } => {
            println!("정상 사례: 공무원 확인 · 판정 {:?} · 모델 확신도 {:?}", j.verdict, j.confidence);
            assert_eq!(j.verdict, Verdict::G3EligibleNormal);
            assert!(j.confidence.is_none_or(|c| c < 0.95), "기준 이상이면 자동 확정됐어야 합니다: {:?}", j.confidence);
        }
        other => panic!("{other:?}"),
    }
}

/// 다른 프로그램이 VRAM 을 차지한 상황을 흉내: 양자화 가중치를 장치에 올려 두어 여유를 `leave_free_mb` 만 남깁니다.
fn hog_vram(device: &Device, leave_free_mb: u64) -> Vec<QTensor> {
    let (rows, cols) = (8192usize, 8192usize);
    let bytes = vec![0u8; rows * cols / 32 * 34];
    let each = (rows * cols / 32 * 36) as u64; // 장치에서는 블록마다 스케일을 F32 로 폅니다
    let free = device.mem_info().unwrap().expect("장치 여유").0 as u64;
    let n = free.saturating_sub(leave_free_mb << 20) / each;
    (0..n).map(|_| qtensor_from_ggml(GgmlDType::Q8_0, &bytes, vec![rows, cols], device).expect("hog")).collect()
}

#[test]
#[ignore]
fn memory_pressure_is_reported_instead_of_silently_changing_the_numbers() {
    // 같은 GPU 를 다른 프로그램이 쓰고 있어 자리가 모자랄 때, 일부 가중치가 조용히 호스트에 남아 수치가 달라지면 안 됩니다.
    // (실제로 겪은 일: 같은 GPU 에서 zoro 프로세스 둘을 동시에 돌렸더니 판단 확률이 실행마다 달랐습니다.)
    let device = accel::select_device().expect("장치");
    if !accel::is_gpu(&device) {
        println!("GPU 가 아니라서 건너뜁니다");
        return;
    }
    let task = gate3_task("없음", "아니오");
    let res = Residency::shared(device.clone(), Budget::unlimited());
    let (model, _bb) = pointer_model(res.clone(), StreamMode::Resident);
    let baseline = model.probabilities(&task).unwrap();
    res.lock().unwrap().release_all().unwrap();

    // 여유를 2 GB 만 남깁니다 (zoro 텍스트 블록은 약 3.8 GB 가 필요).
    let hog = hog_vram(&device, 2048);
    let free = device.mem_info().unwrap().unwrap().0 >> 20;
    println!("점유 후 장치 여유 {free} MB");
    let err = model.probabilities(&task).expect_err("자리가 모자라면 거부해야 합니다");
    let short = err.downcast_ref::<DeviceMemoryShort>().unwrap_or_else(|| panic!("DeviceMemoryShort 여야 합니다: {err:#}"));
    println!("거부됨: {short}");
    assert!(res.lock().unwrap().resident_groups().is_empty(), "거부된 뒤에는 아무것도 올라와 있지 않아야 합니다");

    // 자리를 돌려주면 처음과 같은 확률로 복구됩니다.
    drop(hog);
    device.trim_memory_pool(0).unwrap();
    let again = model.probabilities(&task).unwrap();
    for (a, b) in baseline.iter().zip(&again) {
        assert!((a - b).abs() < 1e-6, "복구 뒤 확률이 처음과 다릅니다: {baseline:?} / {again:?}");
    }
}
