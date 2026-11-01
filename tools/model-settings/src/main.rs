//! 모델 관리 화면. 이 PC 에서만 열리는 작은 서버로 설정 화면을 띄웁니다.
//!
//! 앱 본체(Tauri)가 생기면 같은 화면을 logis-center 처럼 설정 탭에 붙이고, 서버 대신 `invoke` 로 같은 함수를 부르면 됩니다.
//!
//!   cargo run --release -p zorogravity-models
//!   cargo run --release -p zorogravity-models -- --no-open   (브라우저를 열지 않음)

use std::collections::HashMap;
use std::io::Read;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};
use tiny_http::{Header, Method, Request, Response, Server};
use zorogravity_models::{delete_all, delete_model, download, entry, models_dir, status, CATALOG};

const PAGE: &str = include_str!("settings.html");
const PORT: u16 = 17860;

/// 다운로드 하나의 상태
#[derive(Debug, Clone, Default)]
struct Job {
    running: bool,
    percent: u32,
    message: String,
    error: Option<String>,
}

type Jobs = Arc<Mutex<HashMap<&'static str, Job>>>;

fn lock(jobs: &Jobs) -> std::sync::MutexGuard<'_, HashMap<&'static str, Job>> {
    jobs.lock().unwrap_or_else(|p| p.into_inner())
}

fn models_json(jobs: &Jobs) -> Value {
    let jobs = lock(jobs);
    let models: Vec<Value> = status()
        .iter()
        .map(|s| {
            let job = jobs.get(s.entry.name).cloned().unwrap_or_default();
            json!({
                "name": s.entry.name,
                "label": s.entry.label,
                "license": s.entry.license,
                "repo": s.entry.repo,
                "total_bytes": s.entry.total_bytes(),
                "bytes_on_disk": s.bytes_on_disk,
                "installed": s.installed,
                "running": job.running,
                "percent": job.percent,
                "message": job.message,
                "error": job.error,
            })
        })
        .collect();
    json!({ "dir": models_dir().display().to_string(), "models": models })
}

/// 내려받기를 시작합니다. 같은 모델을 두 번 받지 않습니다.
fn start_download(name: &str, jobs: &Jobs) -> Result<()> {
    let entry = entry(name).ok_or_else(|| anyhow!("모르는 모델입니다: {name}"))?;
    if entry.repo.is_none() {
        bail!("{name} 은 내려받을 곳이 없습니다");
    }
    {
        let mut all = lock(jobs);
        let job = all.entry(entry.name).or_default();
        if job.running {
            return Ok(());
        }
        *job = Job { running: true, ..Job::default() };
    }
    let jobs = Arc::clone(jobs);
    std::thread::spawn(move || {
        let result = download(entry.name, &mut |p| {
            if let Some(job) = lock(&jobs).get_mut(entry.name) {
                job.percent = p.percent;
                job.message = p.message;
            }
        });
        if let Some(job) = lock(&jobs).get_mut(entry.name) {
            job.running = false;
            job.error = result.err().map(|e| format!("{e:#}"));
        }
    });
    Ok(())
}

/// 받는 중인 모델의 폴더는 지우지 않습니다.
fn start_delete(names: &[&'static str], jobs: &Jobs) -> Result<()> {
    let mut all = lock(jobs);
    if let Some(busy) = names.iter().find(|n| all.get(**n).is_some_and(|j| j.running)) {
        bail!("{busy} 를 내려받는 중입니다");
    }
    for name in names {
        delete_model(name)?;
        all.remove(name);
    }
    Ok(())
}

/// 이 PC 의 이 화면에서 온 요청만 받습니다. 다른 사이트가 브라우저를 통해 모델을 지우지 못하게 막습니다.
fn same_origin(request: &Request, port: u16) -> bool {
    let header = |name: &'static str| request.headers().iter().find(|h| h.field.equiv(name)).map(|h| h.value.as_str().to_string());
    let hosts = [format!("127.0.0.1:{port}"), format!("localhost:{port}")];
    let host_ok = header("Host").is_some_and(|h| hosts.contains(&h));
    let origin_ok = header("Origin").is_none_or(|o| hosts.iter().any(|h| o == format!("http://{h}")));
    host_ok && origin_ok
}

fn reply(request: Request, status: u16, content_type: &'static str, body: String) {
    let header = Header::from_bytes("Content-Type", content_type).expect("헤더");
    let _ = request.respond(Response::from_string(body).with_status_code(status).with_header(header));
}

fn reply_json(request: Request, result: Result<Value>) {
    match result {
        Ok(v) => reply(request, 200, "application/json; charset=utf-8", v.to_string()),
        Err(e) => reply(request, 400, "application/json; charset=utf-8", json!({ "error": format!("{e:#}") }).to_string()),
    }
}

fn handle(mut request: Request, jobs: &Jobs, port: u16) {
    if !same_origin(&request, port) {
        reply(request, 403, "text/plain; charset=utf-8", "다른 곳에서 온 요청입니다".to_string());
        return;
    }
    // 본문은 쓰지 않습니다.
    let _ = std::io::copy(&mut request.as_reader().take(1024), &mut std::io::sink());
    let url = request.url().to_string();
    let parts: Vec<&str> = url.trim_matches('/').split('/').collect();
    match (request.method(), parts.as_slice()) {
        (Method::Get, [""]) => reply(request, 200, "text/html; charset=utf-8", PAGE.to_string()),
        (Method::Get, ["api", "models"]) => {
            let body = models_json(jobs);
            reply_json(request, Ok(body));
        }
        (Method::Post, ["api", "download", name]) => {
            let done = start_download(name, jobs).map(|()| models_json(jobs));
            reply_json(request, done);
        }
        (Method::Post, ["api", "delete", name]) => {
            let done = match entry(name) {
                Some(e) => start_delete(&[e.name], jobs).map(|()| models_json(jobs)),
                None => Err(anyhow!("모르는 모델입니다: {name}")),
            };
            reply_json(request, done);
        }
        (Method::Post, ["api", "delete-all"]) => {
            let names: Vec<&'static str> = CATALOG.iter().map(|e| e.name).collect();
            let done = start_delete(&names, jobs).and_then(|()| delete_all()).map(|()| models_json(jobs));
            reply_json(request, done);
        }
        _ => reply(request, 404, "text/plain; charset=utf-8", "없는 주소입니다".to_string()),
    }
}

fn open_browser(url: &str) {
    #[cfg(windows)]
    let opened = std::process::Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(target_os = "macos")]
    let opened = std::process::Command::new("open").arg(url).spawn();
    #[cfg(not(any(windows, target_os = "macos")))]
    let opened = std::process::Command::new("xdg-open").arg(url).spawn();
    if let Err(e) = opened {
        eprintln!("브라우저를 열 수 없습니다 ({e}). 위 주소를 직접 여세요.");
    }
}

fn main() -> Result<()> {
    // 정해 둔 포트가 쓰이고 있으면 OS 가 고른 포트로 엽니다.
    let server = Server::http(("127.0.0.1", PORT)).or_else(|_| Server::http(("127.0.0.1", 0))).map_err(|e| anyhow!("서버를 열 수 없습니다: {e}"))?;
    let port = server.server_addr().to_ip().map_or(PORT, |a| a.port());
    let url = format!("http://127.0.0.1:{port}/");
    println!("모델 관리 화면: {url}");
    println!("모델 폴더: {}", models_dir().display());
    println!("끝내려면 Ctrl+C 를 누르세요.");
    if !std::env::args().any(|a| a == "--no-open") {
        open_browser(&url);
    }
    let jobs: Jobs = Arc::default();
    for request in server.incoming_requests() {
        handle(request, &jobs, port);
    }
    Ok(())
}
