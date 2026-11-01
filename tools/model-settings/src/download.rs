//! 모델 다운로드. logis-center 의 `download_model` 과 같이 `.tmp` 로 받은 뒤 이름을 바꾸고, 진행률을 이벤트로 알립니다.
//! 수 GB 파일이라 끊긴 곳부터 이어 받고, 서버가 알려 준 크기만큼 받았는지 확인합니다.

use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use ureq::Agent;

use crate::{entry, model_dir, ModelEntry};

/// 내려받는 곳. 기본은 Hugging Face 이고, 망이 분리된 곳은 같은 경로 구조의 미러를 지정합니다.
const HUB: &str = "https://huggingface.co";
/// 미러 주소를 정하는 환경 변수
pub const MIRROR_ENV: &str = "ZOROGRAVITY_MODEL_MIRROR";

const CHUNK: usize = 1 << 20;

/// 진행률 이벤트 (logis-center 의 `download_progress` 와 같은 내용)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Progress {
    pub model: &'static str,
    /// 모델 전체의 진행률 (0 ~ 100)
    pub percent: u32,
    pub message: String,
}

fn url(repo: &str, file: &str) -> String {
    let base = std::env::var(MIRROR_ENV).ok().filter(|m| !m.trim().is_empty()).unwrap_or_else(|| HUB.to_string());
    format!("{}/{repo}/resolve/main/{file}", base.trim_end_matches('/'))
}

/// `bytes 100-199/200` → 200
fn content_range_total(value: &str) -> Option<u64> {
    value.rsplit('/').next()?.trim().parse().ok()
}

/// 파일 하나를 `.tmp` 로 받습니다. 끊기면 `.tmp` 를 남겨서 다음에 이어 받습니다. `on_bytes` 는 받은 총량과 서버가 알려 준 크기를 받습니다.
fn fetch(agent: &Agent, url: &str, tmp: &Path, on_bytes: &mut dyn FnMut(u64, Option<u64>)) -> Result<()> {
    let have = fs::metadata(tmp).map_or(0, |m| m.len());
    let mut request = agent.get(url);
    if have > 0 {
        request = request.header("Range", format!("bytes={have}-"));
    }
    let response = match request.call() {
        Err(ureq::Error::StatusCode(416)) => {
            // 이어 받을 자리가 파일 끝을 넘었습니다. 파일이 바뀐 것이므로 처음부터 받습니다.
            fs::remove_file(tmp)?;
            return fetch(agent, url, tmp, on_bytes);
        }
        other => other.with_context(|| format!("다운로드 실패: {url}"))?,
    };
    // 서버가 범위 요청을 무시하면 처음부터 다시 보냅니다.
    let resumed = response.status() == 206;
    let total = if resumed {
        response.headers().get("content-range").and_then(|v| v.to_str().ok()).and_then(content_range_total)
    } else {
        response.body().content_length()
    };
    let mut out = OpenOptions::new().create(true).write(true).append(resumed).truncate(!resumed).open(tmp)?;
    let mut written = if resumed { have } else { 0 };
    on_bytes(written, total);
    let mut body = response.into_body().into_reader();
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = body.read(&mut buf).with_context(|| format!("연결이 끊겼습니다: {url}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n])?;
        written += n as u64;
        on_bytes(written, total);
    }
    out.flush()?;
    match total {
        Some(total) if total != written => bail!("크기가 맞지 않습니다: {url} ({written} / {total} 바이트)"),
        _ => Ok(()),
    }
}

fn run(entry: &'static ModelEntry, repo: &str, on: &mut dyn FnMut(Progress)) -> Result<()> {
    let dir = model_dir(entry.name);
    fs::create_dir_all(&dir).with_context(|| format!("모델 폴더를 만들 수 없습니다: {}", dir.display()))?;
    // 파일을 받는 데 몇 분이 걸려서 연결을 재사용할 이득이 없습니다. 쉬던 연결은 서버가 닫았을 수 있습니다.
    let agent: Agent = Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .max_idle_connections(0)
        .max_idle_connections_per_host(0)
        .build()
        .into();
    let total = entry.total_bytes().max(1);
    let percent_at = |bytes: u64| ((bytes as f64 / total as f64) * 100.0).min(100.0) as u32;
    let mut done = 0u64;
    for file in entry.files {
        let path = dir.join(file.name);
        if fs::metadata(&path).is_ok_and(|m| m.len() > 0) {
            done += file.bytes;
            on(Progress { model: entry.name, percent: percent_at(done), message: format!("이미 있음: {}", file.name) });
            continue;
        }
        let link = url(repo, file.name);
        on(Progress { model: entry.name, percent: percent_at(done), message: format!("다운로드 중: {} ({link})", file.name) });
        let tmp = dir.join(format!("{}.tmp", file.name));
        let mut last = u32::MAX;
        fetch(&agent, &link, &tmp, &mut |bytes, size| {
            // 저장소의 파일이 바뀌어도 어긋나지 않게, 이 파일의 몫은 서버가 알려 준 크기로 나눕니다.
            let part = match size {
                Some(size) if size > 0 => (bytes as f64 / size as f64).min(1.0),
                _ => 0.0,
            };
            let percent = percent_at(done + (file.bytes as f64 * part) as u64);
            // 1 MiB 마다 알리면 너무 잦아서 숫자가 바뀔 때만 알립니다.
            if percent != last {
                last = percent;
                on(Progress { model: entry.name, percent, message: format!("다운로드 중: {}", file.name) });
            }
        })?;
        fs::rename(&tmp, &path).with_context(|| format!("파일을 옮길 수 없습니다: {}", path.display()))?;
        done += file.bytes;
    }
    on(Progress { model: entry.name, percent: 100, message: "완료".to_string() });
    Ok(())
}

/// 모델을 내려받습니다. 끝날 때까지 돌아오지 않으므로 화면은 별도 스레드에서 부릅니다.
pub fn download(name: &str, on: &mut dyn FnMut(Progress)) -> Result<()> {
    let entry = entry(name).ok_or_else(|| anyhow!("모르는 모델입니다: {name}"))?;
    let Some(repo) = entry.repo else {
        bail!("{name} 은 내려받을 곳이 없습니다. 직접 변환해서 {} 에 넣으세요", model_dir(name).display());
    };
    run(entry, repo, on)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempModels;
    use crate::{status, CATALOG};
    use std::collections::HashMap;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};

    /// 받은 요청의 (경로, 이어 받을 자리)
    type Seen = Arc<Mutex<Vec<(String, Option<String>)>>>;

    /// 범위 요청을 지원하는 작은 서버. 파일은 `/<저장소>/resolve/main/<이름>` 으로 내줍니다.
    struct Mirror {
        files: Arc<Mutex<HashMap<String, Vec<u8>>>>,
        /// 범위 요청을 무시할지, 응답을 몇 바이트에서 끊을지
        ignore_range: Arc<Mutex<bool>>,
        cut_at: Arc<Mutex<Option<usize>>>,
        /// 알리지 않고 연결을 닫을지 (쉬는 연결을 서버가 먼저 닫는 경우)
        silent_close: Arc<Mutex<bool>>,
        requests: Seen,
        base: String,
    }

    impl Mirror {
        fn start() -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let base = format!("http://{}", listener.local_addr().unwrap());
            let m = Self {
                files: Arc::default(),
                ignore_range: Arc::default(),
                cut_at: Arc::default(),
                silent_close: Arc::default(),
                requests: Arc::default(),
                base,
            };
            let (files, ignore, cut, seen) = (m.files.clone(), m.ignore_range.clone(), m.cut_at.clone(), m.requests.clone());
            let silent = m.silent_close.clone();
            std::thread::spawn(move || {
                for stream in listener.incoming().flatten() {
                    let mut lines = BufReader::new(stream.try_clone().unwrap()).lines().map_while(|l| l.ok());
                    let path = lines.next().unwrap_or_default().split(' ').nth(1).unwrap_or("").to_string();
                    let mut range = None;
                    for line in lines.by_ref() {
                        if line.is_empty() {
                            break;
                        }
                        if let Some(v) = line.to_ascii_lowercase().strip_prefix("range: bytes=") {
                            range = v.trim_end_matches('-').parse::<usize>().ok();
                        }
                    }
                    seen.lock().unwrap().push((path.clone(), range.map(|r| r.to_string())));
                    let mut s = stream;
                    let body = files.lock().unwrap().get(&path).cloned();
                    let Some(body) = body else {
                        let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        continue;
                    };
                    let from = range.filter(|_| !*ignore.lock().unwrap());
                    if from.is_some_and(|f| f > body.len()) {
                        let _ = s.write_all(b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        continue;
                    }
                    let start = from.unwrap_or(0);
                    let head = match from {
                        Some(_) => format!(
                            "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len() - 1,
                            body.len(),
                            body.len() - start
                        ),
                        None => format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()),
                    };
                    let head = if *silent.lock().unwrap() { head.replace("Connection: close\r\n", "") } else { head };
                    let _ = s.write_all(head.as_bytes());
                    let end = cut.lock().unwrap().take().map_or(body.len(), |c| c.min(body.len()));
                    let _ = s.write_all(&body[start..end.max(start)]);
                    if *silent.lock().unwrap() {
                        // 클라이언트가 이 연결로 다음 요청을 보낸 뒤에 닫습니다.
                        std::thread::sleep(Duration::from_millis(50));
                    }
                }
            });
            std::env::set_var(MIRROR_ENV, &m.base);
            m
        }

        fn add_model(&self, name: &str, size: usize) {
            let e = entry(name).unwrap();
            let mut files = self.files.lock().unwrap();
            for (i, f) in e.files.iter().enumerate() {
                let body: Vec<u8> = (0..size + i).map(|b| (b % 251) as u8).collect();
                files.insert(format!("/{}/resolve/main/{}", e.repo.unwrap(), f.name), body);
            }
        }
    }

    impl Drop for Mirror {
        fn drop(&mut self) {
            std::env::remove_var(MIRROR_ENV);
        }
    }

    fn collect(name: &str) -> (Result<()>, Vec<Progress>) {
        let mut seen = Vec::new();
        let result = download(name, &mut |p| seen.push(p));
        (result, seen)
    }

    #[test]
    fn links_point_at_the_hub_or_the_mirror() {
        let _t = TempModels::new("url");
        assert_eq!(url("Qntology/zoro-1-gguf", "active.json"), "https://huggingface.co/Qntology/zoro-1-gguf/resolve/main/active.json");
        std::env::set_var(MIRROR_ENV, "http://mirror.local/hf/");
        assert_eq!(url("a/b", "c.gguf"), "http://mirror.local/hf/a/b/resolve/main/c.gguf");
        std::env::remove_var(MIRROR_ENV);
    }

    #[test]
    fn reads_the_total_from_a_content_range() {
        assert_eq!(content_range_total("bytes 100-199/200"), Some(200));
        assert_eq!(content_range_total("bytes 0-1/*"), None);
    }

    #[test]
    fn downloads_every_file_and_reports_rising_progress() {
        let t = TempModels::new("ok");
        let mirror = Mirror::start();
        mirror.add_model("gocr", 2048);
        let (result, seen) = collect("gocr");
        result.unwrap();
        let dir = t.path().join("gocr");
        for (i, f) in entry("gocr").unwrap().files.iter().enumerate() {
            assert_eq!(fs::metadata(dir.join(f.name)).unwrap().len(), 2048 + i as u64, "{}", f.name);
            assert!(!dir.join(format!("{}.tmp", f.name)).exists());
        }
        assert_eq!(seen.last().unwrap().percent, 100);
        assert!(seen.windows(2).all(|w| w[0].percent <= w[1].percent), "진행률이 뒤로 가면 안 됩니다");
        assert!(status().iter().find(|s| s.entry.name == "gocr").unwrap().installed);
    }

    /// 큰 파일을 받는 동안 쉬던 연결을 서버가 닫아도 다음 파일을 받을 수 있어야 합니다.
    #[test]
    fn a_connection_closed_by_the_server_is_not_reused_for_the_next_file() {
        let _t = TempModels::new("close");
        let mirror = Mirror::start();
        mirror.add_model("gocr", 1024);
        *mirror.silent_close.lock().unwrap() = true;
        collect("gocr").0.unwrap();
    }

    #[test]
    fn files_already_there_are_not_fetched_again() {
        let t = TempModels::new("skip");
        let mirror = Mirror::start();
        mirror.add_model("gocr", 16);
        fs::create_dir_all(t.path().join("gocr")).unwrap();
        fs::write(t.path().join("gocr/gocr-q8.gguf"), b"kept").unwrap();
        collect("gocr").0.unwrap();
        assert_eq!(fs::read(t.path().join("gocr/gocr-q8.gguf")).unwrap(), b"kept");
        assert!(!mirror.requests.lock().unwrap().iter().any(|(p, _)| p.ends_with("gocr-q8.gguf")));
    }

    #[test]
    fn a_broken_connection_keeps_the_partial_file_and_the_next_run_resumes() {
        let t = TempModels::new("resume");
        let mirror = Mirror::start();
        mirror.add_model("gocr", 4096);
        *mirror.cut_at.lock().unwrap() = Some(1000);
        let (result, _) = collect("gocr");
        assert!(result.is_err(), "받다 만 파일을 완성본으로 두면 안 됩니다");
        let dir = t.path().join("gocr");
        assert!(!dir.join("gocr-q8.gguf").exists());
        assert_eq!(fs::metadata(dir.join("gocr-q8.gguf.tmp")).unwrap().len(), 1000);
        assert!(!status()[1].installed);

        collect("gocr").0.unwrap();
        let want: Vec<u8> = (0..4096).map(|b| (b % 251) as u8).collect();
        assert_eq!(fs::read(dir.join("gocr-q8.gguf")).unwrap(), want, "이어 받은 파일이 원본과 같아야 합니다");
        let seen = mirror.requests.lock().unwrap();
        assert!(seen.iter().any(|(p, r)| p.ends_with("gocr-q8.gguf") && r.as_deref() == Some("1000")));
    }

    #[test]
    fn a_server_that_ignores_ranges_restarts_the_file() {
        let t = TempModels::new("restart");
        let mirror = Mirror::start();
        mirror.add_model("gocr", 512);
        *mirror.ignore_range.lock().unwrap() = true;
        let dir = t.path().join("gocr");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("gocr-q8.gguf.tmp"), vec![0xAAu8; 100]).unwrap();
        collect("gocr").0.unwrap();
        let want: Vec<u8> = (0..512).map(|b| (b % 251) as u8).collect();
        assert_eq!(fs::read(dir.join("gocr-q8.gguf")).unwrap(), want);
    }

    #[test]
    fn a_partial_file_longer_than_the_source_restarts_from_zero() {
        let t = TempModels::new("416");
        let mirror = Mirror::start();
        mirror.add_model("gocr", 64);
        let dir = t.path().join("gocr");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("gocr-q8.gguf.tmp"), vec![0u8; 500]).unwrap();
        collect("gocr").0.unwrap();
        assert_eq!(fs::metadata(dir.join("gocr-q8.gguf")).unwrap().len(), 64);
    }

    #[test]
    fn a_missing_file_is_an_error_and_leaves_nothing_behind() {
        let t = TempModels::new("404");
        let _mirror = Mirror::start();
        let err = collect("gocr").0.unwrap_err();
        assert!(format!("{err:#}").contains("gocr-q8.gguf"), "{err:#}");
        assert!(!t.path().join("gocr/gocr-q8.gguf").exists());
    }

    #[test]
    fn models_without_a_link_and_unknown_names_are_refused() {
        let _t = TempModels::new("refuse");
        assert!(CATALOG.iter().any(|e| e.repo.is_none()));
        assert!(collect("siglip2").0.is_err());
        assert!(collect("모르는모델").0.is_err());
    }
}
