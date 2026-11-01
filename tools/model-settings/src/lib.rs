//! 모델 폴더 · 모델 목록 · 상태 · 삭제. logis-center 의 `utils/paths.rs`, `check_model_status`, `delete_all_models` 와 같은 방식입니다.
//!
//! 모델은 저장소가 아니라 OS 의 앱 데이터 폴더에 둡니다.
//!
//! | OS | 위치 |
//! | --- | --- |
//! | Windows | `%LOCALAPPDATA%\zorogravity\models` (`C:\Users\<사용자>\AppData\Local\...`) |
//! | macOS | `~/Library/Application Support/zorogravity/models` |
//! | Linux | `$XDG_DATA_HOME/zorogravity/models` (없으면 `~/.local/share/zorogravity/models`) |
//!
//! 다운로드는 `download` 기능에 있습니다. 코어 엔진은 이 크레이트에서 폴더 위치만 쓰므로 네트워크 코드가 들어가지 않습니다.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

#[cfg(feature = "download")]
mod download;
#[cfg(feature = "download")]
pub use download::{download, Progress, MIRROR_ENV};

/// 앱 데이터 폴더 이름
pub const APP_DIR: &str = "zorogravity";
/// 모델 폴더를 직접 정하는 환경 변수 (시험 · 공용 저장소용)
pub const MODELS_DIR_ENV: &str = "ZOROGRAVITY_MODELS_DIR";

/// 내려받을 파일 하나. `bytes` 는 진행률과 화면 표시에 쓰는 크기입니다 (실제 완료 판단은 서버가 알려 준 크기로 합니다).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelFile {
    pub name: &'static str,
    pub bytes: u64,
}

const fn file(name: &'static str, bytes: u64) -> ModelFile {
    ModelFile { name, bytes }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelEntry {
    /// 폴더 이름. `load_active` 와 예제가 이 이름을 씁니다.
    pub name: &'static str,
    pub label: &'static str,
    pub license: &'static str,
    /// Hugging Face 저장소. 없으면 내려받지 못하고 직접 변환해서 넣습니다.
    pub repo: Option<&'static str>,
    /// `active.json` 을 맨 뒤에 둡니다. 끝까지 받기 전에는 모델이 열리지 않습니다.
    pub files: &'static [ModelFile],
}

impl ModelEntry {
    pub fn total_bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
}

pub const CATALOG: [ModelEntry; 3] = [
    ModelEntry {
        name: "zoro",
        label: "Zoro-1 판단 · 글쓰기",
        license: "Apache-2.0",
        repo: Some("Qntology/zoro-1-gguf"),
        files: &[
            file("zoro-q8.gguf", 4_482_356_320),
            file("mmproj-zoro-bf16.gguf", 667_052_960),
            file("tokenizer.json", 19_989_510),
            file("jev_head.safetensors", 5_245_232),
            file("block_diffusion.json", 34),
            file("active.json", 216),
        ],
    },
    ModelEntry {
        name: "gocr",
        label: "GravityOCR 문서 인식",
        license: "MIT",
        repo: Some("Qntology/GravityOCR-KO-gguf"),
        files: &[
            file("gocr-q8.gguf", 715_315_392),
            file("mmproj-gocr-bf16.gguf", 868_268_128),
            file("tokenizer.json", 6_838_959),
            file("block_diffusion.json", 237),
            file("active.json", 216),
        ],
    },
    ModelEntry { name: "siglip2", label: "SigLIP2 문서 크롭", license: "Apache-2.0", repo: None, files: &[] },
];

pub fn entry(name: &str) -> Option<&'static ModelEntry> {
    CATALOG.iter().find(|e| e.name == name)
}

/// 앱 데이터 폴더. OS 가 위치를 알려 주지 않으면 실행 폴더 아래에 둡니다.
pub fn app_dir() -> PathBuf {
    match dirs::data_local_dir() {
        Some(dir) => dir.join(APP_DIR),
        None => PathBuf::from("zorogravity-data"),
    }
}

/// 모델 폴더
pub fn models_dir() -> PathBuf {
    match std::env::var_os(MODELS_DIR_ENV) {
        Some(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => app_dir().join("models"),
    }
}

/// 모델 하나의 폴더 (`zoro`, `gocr`, `siglip2`)
pub fn model_dir(name: &str) -> PathBuf {
    models_dir().join(name)
}

/// 화면에 보여 줄 모델 상태
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelStatus {
    pub entry: &'static ModelEntry,
    /// 필요한 파일이 모두 있는가
    pub installed: bool,
    /// 폴더가 차지하는 바이트 수 (받다가 멈춘 파일 포함)
    pub bytes_on_disk: u64,
}

/// `.tmp` 를 끝까지 받은 뒤에 이름을 바꾸므로, 파일이 있으면 완전한 파일입니다.
fn installed(dir: &Path, entry: &ModelEntry) -> bool {
    let has = |name: &str| fs::metadata(dir.join(name)).is_ok_and(|m| m.len() > 0);
    if entry.files.is_empty() {
        return has("active.json");
    }
    entry.files.iter().all(|f| has(f.name))
}

fn dir_bytes(dir: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(dir) else { return 0 };
    entries.flatten().filter_map(|e| e.metadata().ok()).filter(|m| m.is_file()).map(|m| m.len()).sum()
}

pub fn status() -> Vec<ModelStatus> {
    CATALOG
        .iter()
        .map(|entry| {
            let dir = model_dir(entry.name);
            ModelStatus { entry, installed: installed(&dir, entry), bytes_on_disk: dir_bytes(&dir) }
        })
        .collect()
}

/// 모델 하나의 폴더를 지웁니다. 목록에 없는 이름은 지우지 않습니다.
pub fn delete_model(name: &str) -> Result<()> {
    let Some(entry) = entry(name) else { bail!("모르는 모델입니다: {name}") };
    let dir = model_dir(entry.name);
    if dir.exists() {
        fs::remove_dir_all(&dir).with_context(|| format!("모델 폴더를 지울 수 없습니다: {}", dir.display()))?;
    }
    Ok(())
}

/// 목록에 있는 모델을 모두 지웁니다.
pub fn delete_all() -> Result<()> {
    CATALOG.iter().try_for_each(|e| delete_model(e.name))
}

#[cfg(test)]
pub(crate) mod testing {
    use std::path::{Path, PathBuf};
    use std::sync::{Mutex, MutexGuard};

    static ENV: Mutex<()> = Mutex::new(());

    /// 임시 모델 폴더. 환경 변수를 바꾸므로 시험을 한 번에 하나씩 돌립니다.
    pub struct TempModels {
        pub dir: PathBuf,
        _guard: MutexGuard<'static, ()>,
    }

    impl TempModels {
        pub fn new(tag: &str) -> Self {
            let guard = ENV.lock().unwrap_or_else(|p| p.into_inner());
            let dir = std::env::temp_dir().join(format!("zorogravity-models-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).unwrap();
            std::env::set_var(super::MODELS_DIR_ENV, &dir);
            Self { dir, _guard: guard }
        }

        pub fn path(&self) -> &Path {
            &self.dir
        }
    }

    impl Drop for TempModels {
        fn drop(&mut self) {
            std::env::remove_var(super::MODELS_DIR_ENV);
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testing::TempModels;
    use super::*;

    fn fill(dir: &Path, names: &[&str]) {
        fs::create_dir_all(dir).unwrap();
        for n in names {
            fs::write(dir.join(n), b"x").unwrap();
        }
    }

    #[test]
    fn models_live_in_the_os_app_data_folder() {
        let dir = app_dir();
        assert_eq!(dir.file_name().unwrap(), APP_DIR);
        if let Some(base) = dirs::data_local_dir() {
            assert!(dir.starts_with(base));
        }
        #[cfg(windows)]
        assert!(dir.to_string_lossy().to_ascii_lowercase().contains("appdata"));
        #[cfg(target_os = "macos")]
        assert!(dir.to_string_lossy().contains("Application Support"));
    }

    #[test]
    fn the_environment_can_move_the_models_folder() {
        let t = TempModels::new("env");
        assert_eq!(models_dir(), t.path());
        assert_eq!(model_dir("zoro"), t.path().join("zoro"));
    }

    #[test]
    fn the_catalog_uses_the_given_repositories_and_ends_with_the_active_record() {
        assert_eq!(entry("zoro").unwrap().repo, Some("Qntology/zoro-1-gguf"));
        assert_eq!(entry("gocr").unwrap().repo, Some("Qntology/GravityOCR-KO-gguf"));
        assert!(entry("없음").is_none());
        for e in CATALOG.iter().filter(|e| e.repo.is_some()) {
            assert_eq!(e.files.last().unwrap().name, "active.json", "{}", e.name);
            assert!(e.total_bytes() > 1_000_000_000);
        }
    }

    #[test]
    fn a_model_counts_as_installed_only_with_every_file() {
        let t = TempModels::new("status");
        let zoro = t.path().join("zoro");
        fill(&zoro, &["zoro-q8.gguf", "tokenizer.json"]);
        assert!(!status()[0].installed);
        assert_eq!(status()[0].bytes_on_disk, 2);
        let names: Vec<&str> = entry("zoro").unwrap().files.iter().map(|f| f.name).collect();
        fill(&zoro, &names);
        assert!(status()[0].installed);
        // 내려받을 수 없는 모델은 직접 넣은 선택 기록을 봅니다.
        assert!(!status()[2].installed);
        fill(&t.path().join("siglip2"), &["active.json"]);
        assert!(status()[2].installed);
    }

    #[test]
    fn deleting_removes_only_known_model_folders() {
        let t = TempModels::new("delete");
        fill(&t.path().join("zoro"), &["a.gguf"]);
        fill(&t.path().join("gocr"), &["b.gguf"]);
        fill(&t.path().join("my-notes"), &["keep.txt"]);
        assert!(delete_model("..").is_err());
        assert!(delete_model("my-notes").is_err());
        delete_model("zoro").unwrap();
        assert!(!t.path().join("zoro").exists());
        assert!(t.path().join("gocr").exists());
        delete_model("zoro").unwrap();
        delete_all().unwrap();
        assert!(!t.path().join("gocr").exists());
        assert!(t.path().join("my-notes/keep.txt").exists());
    }
}
