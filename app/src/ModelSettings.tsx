import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useState } from "react";

interface Job {
  running: boolean;
  percent: number;
  message: string;
  error: string | null;
}

interface ModelRow {
  name: string;
  label: string;
  license: string;
  repo: string | null;
  total_bytes: number;
  bytes_on_disk: number;
  installed: boolean;
  job: Job;
}

interface ModelReport {
  dir: string;
  models: ModelRow[];
}

interface Alert {
  action: string;
  model?: string;
  message: string;
}

const size = (b: number) => (b >= 1e9 ? `${(b / 1e9).toFixed(1)} GB` : `${Math.round(b / 1e6)} MB`);

function note(m: ModelRow): string {
  if (m.job.running) return m.job.message;
  if (m.installed) return `${size(m.bytes_on_disk)} 사용 중`;
  if (m.bytes_on_disk > 0) return `받다 멈춤 (${size(m.bytes_on_disk)}). 다시 누르면 이어 받습니다.`;
  return m.repo ? `huggingface.co/${m.repo}` : "내려받을 곳이 없습니다. 직접 변환해서 모델 폴더에 넣으세요.";
}

/** 모델 다운로드 · 삭제 설정. logis-center 처럼 모델이 없으면 Rust 의 알림을 받아 설정을 열고 내려받습니다. */
export function ModelSettings({ open, onOpen }: { open: boolean; onOpen: () => void }) {
  const [report, setReport] = useState<ModelReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);

  const refresh = useCallback(() => invoke<ModelReport>("check_model_status").then(setReport).catch((e) => setError(String(e))), []);

  const run = useCallback(
    (command: string, args?: Record<string, string>) => {
      setError(null);
      setConfirming(null);
      invoke(command, args).catch((e) => setError(String(e))).finally(refresh);
    },
    [refresh],
  );

  useEffect(() => {
    const stops = [
      listen<Alert>("app_error_alert", (e) => {
        if (e.payload.action !== "open_settings") return setError(e.payload.message);
        onOpen();
        if (e.payload.model) run("download_model", { modelName: e.payload.model });
      }),
      listen<{ model: string; percent: number; message: string }>("download_progress", (e) => {
        const { model, percent, message } = e.payload;
        setReport((r) => r && { ...r, models: r.models.map((m) => (m.name === model ? { ...m, job: { ...m.job, running: true, percent, message } } : m)) });
      }),
      listen("download_complete", refresh),
      listen("download_error", refresh),
    ];
    // 듣기 시작한 뒤에 물어야 알림을 놓치지 않습니다.
    Promise.all(stops).then(() => invoke("ensure_models")).catch((e) => setError(String(e))).finally(refresh);
    return () => void stops.forEach((s) => s.then((f) => f()));
  }, [onOpen, refresh, run]);

  if (!open) return null;
  const models = report?.models ?? [];
  const missing = models.filter((m) => m.repo && !m.installed && !m.job.running);
  const busy = models.some((m) => m.job.running);

  return (
    <aside className="settings">
      <h2>모델 관리</h2>
      <p className="muted path">모델 폴더: {report?.dir}</p>
      <div className="settings-actions">
        <button className="teal" disabled={missing.length === 0} onClick={() => missing.forEach((m) => run("download_model", { modelName: m.name }))}>
          없는 모델 모두 받기
        </button>
        <button
          className="danger"
          disabled={busy || !models.some((m) => m.bytes_on_disk > 0)}
          onClick={() => (confirming === "all" ? run("delete_all_models") : setConfirming("all"))}
        >
          {confirming === "all" ? "한 번 더 누르면 모두 삭제" : "모델 모두 삭제"}
        </button>
      </div>
      {error && <p className="error">{error}</p>}
      {models.map((m) => (
        <article className="model" key={m.name}>
          <div className="model-top">
            <strong>
              {m.label} / {m.license}
            </strong>
            <span className="buttons">
              <button
                className={m.repo && !m.installed && !m.job.running ? "green" : "grey"}
                disabled={!m.repo || m.installed || m.job.running}
                onClick={() => run("download_model", { modelName: m.name })}
              >
                {m.job.running ? `받는 중 (${m.job.percent}%)` : m.installed ? "받음" : m.repo ? `다운로드 (${size(m.total_bytes)})` : "링크 없음"}
              </button>
              {m.bytes_on_disk > 0 && !m.job.running && (
                <button className="danger" onClick={() => (confirming === m.name ? run("delete_model", { modelName: m.name }) : setConfirming(m.name))}>
                  {confirming === m.name ? "한 번 더 누르면 삭제" : "삭제"}
                </button>
              )}
            </span>
          </div>
          {m.job.running && (
            <div className="bar" role="progressbar" aria-valuenow={m.job.percent}>
              <div style={{ width: `${m.job.percent}%` }} />
            </div>
          )}
          <p className={m.job.error ? "error" : "muted"}>{m.job.error ? `오류: ${m.job.error}` : note(m)}</p>
        </article>
      ))}
    </aside>
  );
}
