//! zoro-gravity 앱. 창구 PC 한 대에서 민원인 키오스크와 공무원 콘솔 두 창을 띄웁니다.
//!
//! 모델 관리는 logis-center 와 같은 구조입니다. 프론트가 명령을 부르면 다운로드는 Rust 스레드에서 돌고,
//! `download_progress` · `download_complete` · `download_error` 이벤트로 설정 화면에 알립니다.
//! 모델이 없으면 `app_error_alert` 로 설정을 열고 다운로드를 시작하게 합니다.
//!
//! 키오스크에는 내부 정보를 보내지 않습니다. 모델 · 심사 명령은 공무원 창에서만 받고, 이벤트도 공무원 창으로만 보냅니다.
//! 키오스크는 스캔(`scan_documents`)과 접수(`submit_form`)만 부르고, 접수 번호와 교부 문서만 받습니다.
//!
//! 스캐너는 `ZOROGRAVITY_SCANNER_DIR` 폴더의 이미지로 대신합니다. 실제 스캐너 드라이버가 들어갈 자리입니다.

use std::collections::HashMap;
use std::hash::{BuildHasher, Hasher};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use chrono::{DateTime, Local, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{AppHandle, Emitter, Manager, RunEvent, State, Window};
use zorogravity_core::ai::accel;
use zorogravity_core::ai::document::DocumentReader;
use zorogravity_core::case::Category;
use zorogravity_core::drafting::DEPARTMENTS;
use zorogravity_core::extract::{extract, PageText};
use zorogravity_core::forms;
use zorogravity_core::intake::{self, Checklist, CounterInput, DeskAction, DeskError, IntakeItem, IntakeStatus, KioskStatus, SubmitError};
use zorogravity_core::lattice::Lattice;
use zorogravity_core::policy::{AutomationPolicy, OperatingMode};
use zorogravity_core::privacy;
use zorogravity_core::state_machine::gate3_substantive::Justification;
use zorogravity_core::state_machine::Engine;
use zorogravity_models::{app_dir, delete_model as remove_model, download, entry, models_dir, status, CATALOG};

const OFFICER: &str = "officer";
const KIOSK: &str = "kiosk";

/// 창구 하나의 상태. 엔진은 접수된 건의 세션을 들고 있습니다.
struct Desk {
    engine: Mutex<Engine>,
    /// 합성 관계망 · 대장 (docs/synthetic-data.md)
    db: Lattice,
    /// GPU 를 쓰는 일은 한 번에 하나입니다. 스캔 중에도 공무원 화면은 엔진을 읽을 수 있습니다.
    gpu: Mutex<()>,
    reader: Mutex<Option<Arc<DocumentReader>>>,
    status: Mutex<String>,
    loaded: Mutex<bool>,
    device: Mutex<Option<candle_core::Device>>,
    /// 키오스크가 결과를 기다리는 접수 번호
    kiosk_case: Mutex<Option<String>>,
    /// 지금 민원인이 넣은 서류. 읽은 값은 화면을 거치지 않고 접수로 바로 갑니다.
    kiosk_scans: Mutex<Vec<Value>>,
}

/// 키오스크가 받는 값. 내부 정보를 넣지 않습니다.
#[derive(Debug, Serialize)]
struct KioskReply {
    /// "waiting", "finished", "invalid", "sign_again", "excuse"
    state: &'static str,
    receipt_no: Option<String>,
    notice: Option<String>,
    message: Option<String>,
    locked: Vec<String>,
}

impl Default for KioskReply {
    fn default() -> Self {
        Self { state: "waiting", receipt_no: None, notice: None, message: None, locked: Vec::new() }
    }
}

/// 스캔 결과. 민원인이 낸 서류에서 읽은 값뿐입니다.
#[derive(Debug, Serialize)]
struct ScanReply {
    category: Option<&'static str>,
    docs: Vec<Value>,
    prefill: Value,
    unknown_pages: usize,
    unsure: Vec<String>,
}

/// 공무원 콘솔의 접수 목록 한 줄 (JSON 키는 프론트와 같은 이름)
#[derive(Debug, Serialize)]
struct IntakeRowDto {
    receipt_no: String,
    name: Option<String>,
    category: Option<&'static str>,
    /// "awaiting" (확인 대기), "transferred" (이첩 완료), "finalized" (확정 완료)
    state: &'static str,
    received_at: Option<String>,
    suggested_department: Option<String>,
    department: Option<String>,
    officer_id: Option<String>,
    transferred_at: Option<String>,
    archive_id: Option<String>,
    final_code: Option<String>,
    auto: bool,
}

/// 키오스크가 보내는 접수 입력
#[derive(Debug, Deserialize)]
struct FormSubmit {
    category: String,
    values: Value,
}

/// 공무원 확정 입력 (창구 확인과 정산 금액)
#[derive(Debug, Deserialize)]
struct ConfirmInput {
    receipt_no: String,
    officer_id: String,
    docs: Vec<String>,
    identity_checked: bool,
    pending_checked: bool,
    contract_rent: Option<i64>,
    latest_transfer: Option<i64>,
    printed_account: Option<String>,
    landlord_name: Option<String>,
    landlord_birth: Option<String>,
    approved_amount: Option<i64>,
    evidence_sum: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ActionInput {
    receipt_no: String,
    officer_id: String,
    action: String,
    evidence_scans: Option<u32>,
    team_lead_cosign: Option<bool>,
    rule_ref: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AutomationInput {
    on: bool,
    approver_id: String,
    legal_basis: String,
}

#[derive(Debug, Serialize)]
struct ConsoleDto {
    automation: bool,
    engine: String,
    active: usize,
    archived: usize,
    graph_nodes: usize,
    audit_count: usize,
    audit_last: Vec<String>,
}

#[derive(Debug, Serialize)]
struct FormLineDto {
    label: String,
    value: String,
}

#[derive(Debug, Serialize)]
struct DocDto {
    kind: String,
    how: &'static str,
    done: bool,
}

#[derive(Debug, Serialize)]
struct SummaryDto {
    gist: String,
    demand: String,
    department: Option<String>,
    verified: bool,
    problems: Vec<String>,
}

#[derive(Debug, Serialize)]
struct DepartmentDto {
    name: &'static str,
    description: &'static str,
}

#[derive(Debug, Serialize)]
struct IntakeDetailDto {
    row: IntakeRowDto,
    /// "grievance" 이면 이첩, 그 밖에는 확정
    kind: &'static str,
    category_label: &'static str,
    birth_prefix: String,
    contact: Option<String>,
    body: String,
    lines: Vec<FormLineDto>,
    reason: String,
    candidate: Option<String>,
    candidate_class: Option<&'static str>,
    confidence: Option<f32>,
    docs: Vec<DocDto>,
    pending: Vec<String>,
    identity_done: bool,
    relations: Vec<String>,
    claimed_total: Option<i64>,
    verified_sum: Option<i64>,
    needs: Vec<&'static str>,
    summary: Option<SummaryDto>,
    departments: Vec<DepartmentDto>,
}

#[derive(Debug, Deserialize)]
struct ChecklistInput {
    identity_checked: bool,
    contact_checked: bool,
    body_read: bool,
    department_checked: bool,
}

#[derive(Debug, Deserialize)]
struct TransferInput {
    receipt_no: String,
    officer_id: String,
    department: String,
    checklist: ChecklistInput,
}

/// 다운로드 하나의 상태
#[derive(Debug, Clone, Default, Serialize)]
struct Job {
    running: bool,
    percent: u32,
    message: String,
    error: Option<String>,
}

#[derive(Default)]
struct Jobs(Mutex<HashMap<&'static str, Job>>);

impl Jobs {
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<&'static str, Job>> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// 설정 화면의 모델 한 줄
#[derive(Debug, Serialize)]
struct ModelRow {
    name: &'static str,
    label: &'static str,
    license: &'static str,
    repo: Option<&'static str>,
    total_bytes: u64,
    bytes_on_disk: u64,
    installed: bool,
    job: Job,
}

#[derive(Debug, Serialize)]
struct ModelReport {
    dir: String,
    models: Vec<ModelRow>,
}

fn officer_only(window: &Window) -> Result<(), String> {
    if window.label() == OFFICER {
        Ok(())
    } else {
        Err("공무원 창에서만 할 수 있습니다".to_string())
    }
}

fn kiosk_only(window: &Window) -> Result<(), String> {
    if window.label() == KIOSK {
        Ok(())
    } else {
        Err("민원인 창에서만 할 수 있습니다".to_string())
    }
}

/// 접수 번호와 솔트에 쓰는 난수. 표준 라이브러리의 무작위 시드에 시각을 섞습니다.
/// 접수 번호는 비밀이 아닙니다. 겹치면 코어가 다시 뽑습니다.
fn random_u32() -> u32 {
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    hasher.write_u128(nanos);
    hasher.finish() as u32
}

/// 내려받을 수 있는 모델이 모두 있는가. 키오스크는 이 값만 받습니다.
fn engine_ready() -> bool {
    status().iter().all(|s| s.installed || s.entry.repo.is_none())
}

fn notify_kiosk(app: &AppHandle) {
    let loaded = *app.state::<Desk>().loaded.lock().unwrap_or_else(|p| p.into_inner());
    let _ = app.emit_to(KIOSK, "kiosk_state", json!({ "ready": engine_ready() && loaded }));
}

#[tauri::command]
fn check_model_status(window: Window, jobs: State<'_, Jobs>) -> Result<ModelReport, String> {
    officer_only(&window)?;
    let jobs = jobs.lock();
    let models = status()
        .into_iter()
        .map(|s| ModelRow {
            name: s.entry.name,
            label: s.entry.label,
            license: s.entry.license,
            repo: s.entry.repo,
            total_bytes: s.entry.total_bytes(),
            bytes_on_disk: s.bytes_on_disk,
            installed: s.installed,
            job: jobs.get(s.entry.name).cloned().unwrap_or_default(),
        })
        .collect();
    Ok(ModelReport { dir: models_dir().display().to_string(), models })
}

/// 없는 모델마다 설정을 열고 내려받으라고 알립니다. 프론트가 이벤트를 들을 준비를 마친 뒤 부릅니다.
#[tauri::command]
fn ensure_models(window: Window, app: AppHandle) -> Result<(), String> {
    officer_only(&window)?;
    for s in status().iter().filter(|s| !s.installed && s.entry.repo.is_some()) {
        let alert = json!({ "action": "open_settings", "model": s.entry.name, "message": format!("{} 모델을 내려받습니다", s.entry.label) });
        app.emit_to(OFFICER, "app_error_alert", alert).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// 내려받기를 시작하고 바로 돌아옵니다. 같은 모델을 두 번 받지 않습니다.
#[tauri::command]
fn download_model(window: Window, app: AppHandle, jobs: State<'_, Jobs>, model_name: String) -> Result<(), String> {
    officer_only(&window)?;
    let model = entry(&model_name).filter(|e| e.repo.is_some()).ok_or_else(|| format!("내려받을 수 없는 모델입니다: {model_name}"))?;
    {
        let mut jobs = jobs.lock();
        let job = jobs.entry(model.name).or_default();
        if job.running {
            return Ok(());
        }
        *job = Job { running: true, ..Job::default() };
    }
    std::thread::spawn(move || {
        let state = app.state::<Jobs>();
        let result = download(model.name, &mut |p| {
            if let Some(job) = state.lock().get_mut(model.name) {
                job.percent = p.percent;
                job.message = p.message.clone();
            }
            let _ = app.emit_to(OFFICER, "download_progress", json!({ "model": model.name, "percent": p.percent, "message": p.message }));
        });
        let error = result.err().map(|e| format!("{e:#}"));
        if let Some(job) = state.lock().get_mut(model.name) {
            job.running = false;
            job.error = error.clone();
        }
        let _ = match error {
            Some(error) => app.emit_to(OFFICER, "download_error", json!({ "model": model.name, "error": error })),
            None => app.emit_to(OFFICER, "download_complete", json!({ "model": model.name })),
        };
        notify_kiosk(&app);
    });
    Ok(())
}

/// 받는 중인 모델의 폴더는 지우지 않습니다.
fn delete_models(app: &AppHandle, jobs: &Jobs, names: &[&'static str]) -> Result<(), String> {
    let mut jobs = jobs.lock();
    if let Some(busy) = names.iter().find(|n| jobs.get(**n).is_some_and(|j| j.running)) {
        return Err(format!("{busy} 를 내려받는 중입니다"));
    }
    for name in names {
        remove_model(name).map_err(|e| format!("{e:#}"))?;
        jobs.remove(name);
    }
    notify_kiosk(app);
    Ok(())
}

#[tauri::command]
fn delete_model(window: Window, app: AppHandle, jobs: State<'_, Jobs>, model_name: String) -> Result<(), String> {
    officer_only(&window)?;
    let model = entry(&model_name).ok_or_else(|| format!("모르는 모델입니다: {model_name}"))?;
    delete_models(&app, &jobs, &[model.name])
}

#[tauri::command]
fn delete_all_models(window: Window, app: AppHandle, jobs: State<'_, Jobs>) -> Result<(), String> {
    officer_only(&window)?;
    let names: Vec<&'static str> = CATALOG.iter().map(|e| e.name).collect();
    delete_models(&app, &jobs, &names)
}

/// 키오스크가 시작할 때 묻습니다. 준비 여부만 돌려줍니다.
#[tauri::command]
fn kiosk_ready(desk: State<'_, Desk>) -> bool {
    engine_ready() && *desk.loaded.lock().unwrap_or_else(|p| p.into_inner())
}

fn category_of(name: &str) -> Result<Category, String> {
    match name {
        "welfare" => Ok(Category::Welfare),
        "subsidy" => Ok(Category::Subsidy),
        "permit" => Ok(Category::Permit),
        "cert" => Ok(Category::Cert),
        "grievance" => Ok(Category::Grievance),
        _ => Err("지원하지 않는 민원입니다.".to_string()),
    }
}

fn category_name(c: Category) -> &'static str {
    match c {
        Category::Welfare => "welfare",
        Category::Subsidy => "subsidy",
        Category::Permit => "permit",
        Category::Cert => "cert",
        Category::Grievance => "grievance",
    }
}

/// 스캐너가 이미지를 떨어뜨리는 폴더. 실제 스캐너 드라이버가 들어갈 자리입니다.
fn scanner_dir() -> PathBuf {
    std::env::var_os("ZOROGRAVITY_SCANNER_DIR").map(PathBuf::from).unwrap_or_else(|| app_dir().join("scanner"))
}

/// 스캐너 폴더의 이미지를 읽어 서류 값을 돌려줍니다 (민원인 창 전용). 읽은 이미지는 바로 지웁니다.
#[tauri::command(async)]
fn scan_documents(window: Window, app: AppHandle, desk: State<'_, Desk>) -> Result<ScanReply, String> {
    kiosk_only(&window)?;
    let reader = desk.reader.lock().unwrap_or_else(|p| p.into_inner()).clone().ok_or("창구를 준비하고 있습니다. 잠시만 기다려 주세요.")?;
    let dir = scanner_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| matches!(x.to_string_lossy().to_lowercase().as_str(), "jpg" | "jpeg" | "png"))).collect())
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        return Err("스캐너에 서류가 없습니다. 서류나 신분증을 넣어 주세요.".to_string());
    }
    let total = files.len();
    let _ = app.emit_to(KIOSK, "scan_progress", json!({ "total": total, "stage": "읽는 중" }));
    let mut images = Vec::with_capacity(total);
    let mut hashes = Vec::with_capacity(total);
    for f in &files {
        let bytes = std::fs::read(f).map_err(|e| e.to_string())?;
        hashes.push(privacy::node_hash(b"scan", &String::from_utf8_lossy(&bytes)));
        images.push(image::load_from_memory(&bytes).map_err(|e| format!("이미지를 읽을 수 없습니다: {e}"))?.to_rgb8());
    }
    let pages = {
        let _gpu = desk.gpu.lock().unwrap_or_else(|p| p.into_inner());
        reader.read_pages(&images).map_err(|e| {
            eprintln!("[스캔] 읽기 실패: {e:#}");
            "서류를 읽지 못했습니다. 창구 직원에게 알려 주세요.".to_string()
        })?
    };
    let texts: Vec<PageText> = pages.iter().zip(&hashes).map(|(p, h)| PageText::from_reading(h, p)).collect();
    let found = extract(&texts, &desk.db);
    for f in &files {
        let _ = std::fs::remove_file(f);
    }
    // 화면에는 서류 이름과 입력 칸 값만 보냅니다. 민원인의 값이 바뀌지 않도록 읽은 값은 여기 둡니다.
    let docs = found.docs.iter().map(|d| json!({ "kind": d["kind"], "signed": d["signed"] })).collect();
    desk.kiosk_scans.lock().unwrap_or_else(|p| p.into_inner()).extend(found.docs);
    Ok(ScanReply { category: found.category.map(category_name), docs, prefill: found.prefill, unknown_pages: found.unknown_pages, unsure: found.unsure })
}

/// 민원을 접수합니다 (민원인 창 전용). 돌아오는 것은 접수 번호와 민원인에게 보여도 되는 안내뿐입니다.
#[tauri::command(async)]
fn submit_form(window: Window, app: AppHandle, desk: State<'_, Desk>, input: FormSubmit) -> Result<KioskReply, String> {
    kiosk_only(&window)?;
    if !engine_ready() {
        return Err("창구를 준비하고 있습니다. 잠시만 기다려 주세요.".to_string());
    }
    let category = category_of(&input.category)?;
    let mut values = input.values;
    if let Value::Object(map) = &mut values {
        map.insert("scanned".to_string(), Value::Array(desk.kiosk_scans.lock().unwrap_or_else(|p| p.into_inner()).clone()));
    }
    let result = {
        let _gpu = desk.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let mut engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
        // 접수 번호의 날짜와 심사 기준일은 창구의 현지 날짜입니다.
        intake::submit_form(&mut engine, &desk.db, category, &values, Utc::now(), Local::now().date_naive(), random_u32())
    };
    let reply = match result {
        Ok(receipt) => {
            desk.kiosk_scans.lock().unwrap_or_else(|p| p.into_inner()).clear();
            notify_intake(&app);
            match receipt.status {
                KioskStatus::Waiting => {
                    *desk.kiosk_case.lock().unwrap_or_else(|p| p.into_inner()) = Some(receipt.receipt_no.clone());
                    KioskReply { state: "waiting", receipt_no: Some(receipt.receipt_no), ..KioskReply::default() }
                }
                KioskStatus::Finished { notice } => KioskReply { state: "finished", receipt_no: Some(receipt.receipt_no), notice: Some(notice), ..KioskReply::default() },
            }
        }
        Err(SubmitError::Form(e)) => KioskReply { state: "invalid", message: Some(e.to_string()), ..KioskReply::default() },
        Err(SubmitError::Retry(message)) => KioskReply { state: "sign_again", message: Some(message), ..KioskReply::default() },
        Err(SubmitError::NeedExcuse { message, locked }) => KioskReply { state: "excuse", message: Some(message), locked, ..KioskReply::default() },
        Err(SubmitError::Internal(e)) => {
            eprintln!("[접수] 접수 실패: {e:#}");
            return Err("접수하지 못했습니다. 창구 직원에게 알려 주세요.".to_string());
        }
    };
    Ok(reply)
}

/// 키오스크가 처음 화면으로 돌아갑니다. 진행 중이던 건의 안내를 더 받지 않습니다.
#[tauri::command]
fn kiosk_reset(window: Window, desk: State<'_, Desk>) -> Result<(), String> {
    kiosk_only(&window)?;
    *desk.kiosk_case.lock().unwrap_or_else(|p| p.into_inner()) = None;
    desk.kiosk_scans.lock().unwrap_or_else(|p| p.into_inner()).clear();
    Ok(())
}

/// 접수 목록이 바뀌었음을 공무원 창에 알립니다. 목록은 공무원 창이 다시 받습니다.
fn notify_intake(app: &AppHandle) {
    let _ = app.emit_to(OFFICER, "intake_changed", json!({}));
}

/// 키오스크가 기다리는 건이 끝났으면 교부 문서를 보냅니다.
fn notify_kiosk_case(app: &AppHandle, desk: &Desk, receipt: &str) {
    let mut current = desk.kiosk_case.lock().unwrap_or_else(|p| p.into_inner());
    if current.as_deref() != Some(receipt) {
        return;
    }
    let notice = {
        let engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
        engine.store.archive().iter().find(|a| a.case_id == receipt).map(|a| a.citizen_doc.clone())
    };
    let notice = notice.unwrap_or_else(|| "접수하신 민원이 담당 부서로 전달되었습니다.".to_string());
    let _ = app.emit_to(KIOSK, "case_update", json!({ "receipt_no": receipt, "notice": notice }));
    *current = None;
}

fn local_time(at: DateTime<Utc>) -> String {
    at.with_timezone(&Local).format("%Y-%m-%d %H:%M").to_string()
}

fn intake_row(item: IntakeItem) -> IntakeRowDto {
    let mut row = IntakeRowDto {
        receipt_no: item.receipt_no,
        name: item.name,
        category: item.category.map(forms::label),
        state: "awaiting",
        received_at: None,
        suggested_department: None,
        department: None,
        officer_id: None,
        transferred_at: None,
        archive_id: None,
        final_code: None,
        auto: false,
    };
    match item.status {
        IntakeStatus::Awaiting { received_at, suggested } => {
            row.received_at = Some(local_time(received_at));
            row.suggested_department = suggested;
        }
        IntakeStatus::Transferred(t) => {
            row.state = "transferred";
            row.department = Some(t.department);
            row.officer_id = Some(t.officer_id);
            row.transferred_at = Some(local_time(t.at));
        }
        IntakeStatus::Finalized { archive_id, final_code, auto } => {
            row.state = "finalized";
            row.archive_id = Some(archive_id);
            row.final_code = Some(final_code);
            row.auto = auto;
        }
    }
    row
}

/// 공무원 콘솔의 접수 목록 (공무원 창 전용)
#[tauri::command(async)]
fn intake_list(window: Window, desk: State<'_, Desk>) -> Result<Vec<IntakeRowDto>, String> {
    officer_only(&window)?;
    let engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
    Ok(intake::list_intake(&engine).into_iter().map(intake_row).collect())
}

/// 접수 상세 (공무원 창 전용). 처리된 건은 목록의 기록만 남습니다.
#[tauri::command(async)]
fn intake_detail(window: Window, desk: State<'_, Desk>, receipt_no: String) -> Result<IntakeDetailDto, String> {
    officer_only(&window)?;
    let engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
    let d = intake::intake_detail(&engine, &receipt_no).map_err(|e| e.to_string())?;
    let kind = if d.item.category == Some(Category::Grievance) { "grievance" } else { "confirm" };
    let category_label = d.item.category.map(forms::label).unwrap_or("");
    Ok(IntakeDetailDto {
        row: intake_row(d.item),
        kind,
        category_label,
        birth_prefix: d.birth_prefix,
        contact: d.contact,
        body: d.body,
        lines: d.lines.into_iter().map(|(label, value)| FormLineDto { label, value }).collect(),
        reason: d.reason,
        candidate: d.candidate,
        candidate_class: d.candidate_class,
        confidence: d.confidence,
        docs: d.docs.into_iter().map(|x| DocDto { kind: x.kind, how: x.how, done: x.done }).collect(),
        pending: d.pending,
        identity_done: d.identity_done,
        relations: d.relations,
        claimed_total: d.claimed_total,
        verified_sum: d.verified_sum,
        needs: d.needs,
        summary: d.summary.map(|s| SummaryDto { gist: s.gist, demand: s.demand, department: s.department, verified: s.verified, problems: s.problems }),
        departments: DEPARTMENTS.iter().map(|(name, description)| DepartmentDto { name, description }).collect(),
    })
}

fn desk_message(e: DeskError) -> String {
    match e {
        DeskError::Internal(err) => {
            eprintln!("[창구] 처리 실패: {err:#}");
            "처리하지 못했습니다. 앱 로그를 확인해 주세요.".to_string()
        }
        other => other.to_string(),
    }
}

/// 고충민원을 담당 부서로 이첩합니다 (공무원 창 전용). 실패하면 아무것도 바뀌지 않습니다.
#[tauri::command(async)]
fn transfer_intake(window: Window, app: AppHandle, desk: State<'_, Desk>, input: TransferInput) -> Result<IntakeRowDto, String> {
    officer_only(&window)?;
    let c = input.checklist;
    let checklist = Checklist { identity_checked: c.identity_checked, contact_checked: c.contact_checked, body_read: c.body_read, department_checked: c.department_checked };
    let row = {
        let mut engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
        intake::transfer_grievance(&mut engine, &input.receipt_no, &input.officer_id, &input.department, checklist, Utc::now()).map_err(desk_message)?
    };
    notify_intake(&app);
    notify_kiosk_case(&app, &desk, &input.receipt_no);
    Ok(intake_row(IntakeItem { receipt_no: input.receipt_no, name: None, category: Some(Category::Grievance), status: IntakeStatus::Transferred(row) }))
}

impl ConfirmInput {
    fn counter(&self) -> CounterInput {
        CounterInput {
            docs: self.docs.clone(),
            identity_checked: self.identity_checked,
            pending_checked: self.pending_checked,
            contract_rent: self.contract_rent,
            latest_transfer: self.latest_transfer,
            printed_account: self.printed_account.clone().filter(|s| !s.trim().is_empty()),
            landlord: self.landlord_name.clone().filter(|s| !s.trim().is_empty()).zip(self.landlord_birth.clone()),
            approved_amount: self.approved_amount,
            evidence_sum: self.evidence_sum,
        }
    }
}

/// 공무원이 창구 확인을 반영하고 확정합니다 (공무원 창 전용). 후보가 바뀌면 멈추고 알립니다.
#[tauri::command(async)]
fn confirm_form(window: Window, app: AppHandle, desk: State<'_, Desk>, input: ConfirmInput) -> Result<String, String> {
    officer_only(&window)?;
    let result = {
        let _gpu = desk.gpu.lock().unwrap_or_else(|p| p.into_inner());
        let mut engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
        intake::confirm_form(&mut engine, &desk.db, &input.receipt_no, &input.officer_id, &input.counter(), Utc::now())
    };
    notify_intake(&app);
    let archive_id = result.map_err(desk_message)?;
    notify_kiosk_case(&app, &desk, &input.receipt_no);
    Ok(archive_id)
}

/// 반려 · 보완요구 · 강제 진입 · 고위험 승인 (공무원 창 전용). 규칙이 막으면 사유를 돌려줍니다.
#[tauri::command(async)]
fn officer_action(window: Window, app: AppHandle, desk: State<'_, Desk>, input: ActionInput) -> Result<String, String> {
    officer_only(&window)?;
    let action = match input.action.as_str() {
        "reject" => DeskAction::Reject,
        "supplement" => DeskAction::Supplement,
        "force_pass" => DeskAction::ForcePass,
        "approve_cancel" => DeskAction::ApproveHighRisk(Justification::CancelMistake),
        "approve_evidence" => DeskAction::ApproveHighRisk(Justification::ProveFalsePositive { evidence_scans: input.evidence_scans.unwrap_or(0) }),
        "approve_exception" => DeskAction::ApproveHighRisk(Justification::ApplyLawException { team_lead_cosign: input.team_lead_cosign.unwrap_or(false), rule_ref: input.rule_ref.clone().unwrap_or_default() }),
        other => return Err(format!("모르는 처리입니다: {other}")),
    };
    let result = {
        let mut engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
        intake::officer_action(&mut engine, &input.receipt_no, &input.officer_id, action, Utc::now())
    };
    notify_intake(&app);
    let archive_id = result.map_err(desk_message)?;
    notify_kiosk_case(&app, &desk, &input.receipt_no);
    Ok(archive_id)
}

/// 제3자 실명 열람 요청 (공무원 창 전용). 허용과 거부 모두 감사 기록에 남습니다.
#[tauri::command(async)]
fn request_unmask(window: Window, desk: State<'_, Desk>, receipt_no: String, officer_id: String) -> Result<bool, String> {
    officer_only(&window)?;
    let mut engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
    intake::request_unmask(&mut engine, &receipt_no, &officer_id, Utc::now()).map_err(desk_message)
}

/// 자동화 모드 켜기 · 끄기 (공무원 창 전용). 승인자와 법령 근거가 있어야 켜지고 감사 기록에 남습니다.
#[tauri::command(async)]
fn set_automation(window: Window, app: AppHandle, desk: State<'_, Desk>, input: AutomationInput) -> Result<(), String> {
    officer_only(&window)?;
    let mut engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
    if input.on {
        engine.enable_automation(&input.approver_id, &input.legal_basis, Utc::now()).map_err(|e| format!("{e:#}"))?;
    } else {
        engine.disable_automation(&input.approver_id, Utc::now());
    }
    drop(engine);
    notify_intake(&app);
    Ok(())
}

/// 콘솔 머리와 바닥에 보이는 상태 (공무원 창 전용)
#[tauri::command(async)]
fn console_status(window: Window, desk: State<'_, Desk>) -> Result<ConsoleDto, String> {
    officer_only(&window)?;
    let engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
    let audit = engine.store.audit().events();
    Ok(ConsoleDto {
        automation: engine.policy.mode() == OperatingMode::Auto,
        engine: desk.status.lock().unwrap_or_else(|p| p.into_inner()).clone(),
        active: engine.store.active_count(),
        archived: engine.store.archive().len(),
        graph_nodes: desk.db.node_count(),
        audit_count: audit.len(),
        audit_last: audit.iter().rev().take(4).map(|e| format!("{} {:?} · {}", e.at.with_timezone(&Local).format("%H:%M:%S"), e.actor, e.detail)).collect(),
    })
}

/// 모델을 뒤에서 엽니다. 가중치가 받아질 때까지 기다리고, 열리면 엔진에 꽂습니다.
fn load_models(app: AppHandle) {
    let desk = app.state::<Desk>();
    let gpu_build = cfg!(any(feature = "vulkan", feature = "rocm", feature = "cuda", feature = "metal"));
    if !gpu_build || std::env::var("ZOROGRAVITY_ENGINE").is_ok_and(|v| v == "rules") {
        *desk.status.lock().unwrap_or_else(|p| p.into_inner()) = "규칙만 (모델 없음)".to_string();
        *desk.loaded.lock().unwrap_or_else(|p| p.into_inner()) = true;
        notify_kiosk(&app);
        return;
    }
    while !engine_ready() {
        std::thread::sleep(std::time::Duration::from_secs(2));
    }
    match zorogravity_core::ai::runtime::open() {
        Ok(rt) => {
            {
                let mut engine = desk.engine.lock().unwrap_or_else(|p| p.into_inner());
                engine.model = Some(Box::new(rt.model));
                engine.writer = Some(Box::new(rt.writer));
            }
            *desk.reader.lock().unwrap_or_else(|p| p.into_inner()) = Some(rt.reader);
            *desk.status.lock().unwrap_or_else(|p| p.into_inner()) = format!("Zoro-1 · {}", rt.backend);
            *desk.device.lock().unwrap_or_else(|p| p.into_inner()) = Some(rt.device);
        }
        Err(e) => {
            eprintln!("[엔진] 모델을 열지 못했습니다: {e:#}");
            *desk.status.lock().unwrap_or_else(|p| p.into_inner()) = "규칙만 (모델 열기 실패)".to_string();
        }
    }
    *desk.loaded.lock().unwrap_or_else(|p| p.into_inner()) = true;
    notify_kiosk(&app);
}

pub fn run() {
    let salt: Vec<u8> = (0..8).flat_map(|_| random_u32().to_le_bytes()).collect();
    let desk = Desk {
        engine: Mutex::new(Engine::new(&salt, AutomationPolicy::default(), None)),
        db: Lattice::embedded().expect("합성 DB 를 읽을 수 없습니다"),
        gpu: Mutex::new(()),
        reader: Mutex::new(None),
        status: Mutex::new("모델 여는 중".to_string()),
        loaded: Mutex::new(false),
        device: Mutex::new(None),
        kiosk_case: Mutex::new(None),
        kiosk_scans: Mutex::new(Vec::new()),
    };
    tauri::Builder::default()
        .manage(Jobs::default())
        .manage(desk)
        .setup(|app| {
            let handle = app.handle().clone();
            std::thread::spawn(move || load_models(handle));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            check_model_status,
            ensure_models,
            download_model,
            delete_model,
            delete_all_models,
            kiosk_ready,
            kiosk_reset,
            scan_documents,
            submit_form,
            intake_list,
            intake_detail,
            transfer_intake,
            confirm_form,
            officer_action,
            request_unmask,
            set_automation,
            console_status
        ])
        .build(tauri::generate_context!())
        .expect("앱을 시작할 수 없습니다")
        .run(|app, event| {
            // ROCm 프로세스는 DLL 을 내리다 멈출 수 있습니다. GPU 를 비우고 바로 끝냅니다.
            if let RunEvent::Exit = event {
                let desk = app.state::<Desk>();
                let device = desk.device.lock().unwrap_or_else(|p| p.into_inner()).clone();
                if let Some(device) = device {
                    accel::finish_process(&device, 0);
                }
            }
        });
}
