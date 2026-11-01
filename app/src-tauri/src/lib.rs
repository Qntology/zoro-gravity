//! zoro-gravity 앱. 창구 PC 한 대에서 민원인 키오스크와 공무원 콘솔 두 창을 띄웁니다.
//!
//! 모델 관리는 logis-center 와 같은 구조입니다. 프론트가 명령을 부르면 다운로드는 Rust 스레드에서 돌고,
//! `download_progress` · `download_complete` · `download_error` 이벤트로 설정 화면에 알립니다.
//! 모델이 없으면 `app_error_alert` 로 설정을 열고 다운로드를 시작하게 합니다.
//!
//! 키오스크에는 내부 정보를 보내지 않습니다. 모델 명령은 공무원 창에서만 받고, 다운로드 이벤트도 공무원 창으로만 보냅니다.

use std::collections::HashMap;
use std::sync::Mutex;

use serde::Serialize;
use serde_json::json;
use tauri::{AppHandle, Emitter, Manager, State, Window};
use zorogravity_models::{delete_model as remove_model, download, entry, models_dir, status, CATALOG};

const OFFICER: &str = "officer";
const KIOSK: &str = "kiosk";

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

/// 내려받을 수 있는 모델이 모두 있는가. 키오스크는 이 값만 받습니다.
fn engine_ready() -> bool {
    status().iter().all(|s| s.installed || s.entry.repo.is_none())
}

fn notify_kiosk(app: &AppHandle) {
    let _ = app.emit_to(KIOSK, "kiosk_state", json!({ "ready": engine_ready() }));
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
fn kiosk_ready() -> bool {
    engine_ready()
}

pub fn run() {
    tauri::Builder::default()
        .manage(Jobs::default())
        .invoke_handler(tauri::generate_handler![check_model_status, ensure_models, download_model, delete_model, delete_all_models, kiosk_ready])
        .run(tauri::generate_context!())
        .expect("앱을 시작할 수 없습니다");
}
