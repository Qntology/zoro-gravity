import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useState } from "react";
import { IntakeView } from "./IntakeView";
import { ModelSettings } from "./ModelSettings";

interface Console {
  automation: boolean;
  engine: string;
  active: number;
  archived: number;
  graph_nodes: number;
  audit_count: number;
  audit_last: string[];
}

/** 공무원 콘솔. 사번은 머리에서 한 번만 적고, 자동화 모드는 승인자와 법령 근거가 있어야 켜집니다. */
export function OfficerView() {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [officerId, setOfficerId] = useState("");
  const [status, setStatus] = useState<Console | null>(null);
  const [opening, setOpening] = useState(false);
  const [basis, setBasis] = useState("");
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    invoke<Console>("console_status")
      .then(setStatus)
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    refresh();
    const stop = listen("intake_changed", () => refresh());
    return () => void stop.then((f) => f());
  }, [refresh]);

  async function toggle(on: boolean) {
    setError(null);
    try {
      await invoke("set_automation", { input: { on, approver_id: officerId, legal_basis: basis } });
      setOpening(false);
      setBasis("");
      refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <main className="officer">
      <header className="officer-header">
        <strong>zorogravity 심사 콘솔</strong>
        <span className="grow" />
        <label className="inline">
          담당자 사번
          <input id="officer-id" value={officerId} placeholder="예: A-1024" onChange={(e) => setOfficerId(e.target.value)} />
        </label>
        {status?.automation ? (
          <button id="auto-off" className="danger" onClick={() => toggle(false)}>
            자동화 끄기
          </button>
        ) : (
          <button id="auto-open" className="plain" disabled={!officerId.trim()} onClick={() => setOpening((o) => !o)}>
            자동화 켜기
          </button>
        )}
        <em className={`tag ${status?.automation ? "auto" : ""}`}>{status?.automation ? "자동화 모드" : "보조 모드"}</em>
        <button className="plain" onClick={() => setSettingsOpen((open) => !open)}>
          {settingsOpen ? "설정 닫기" : "설정"}
        </button>
      </header>
      {opening && (
        <div className="auto-form">
          <label className="inline wide">
            법령 근거
            <input id="auto-basis" value={basis} placeholder="자동화 처분의 근거 조항" onChange={(e) => setBasis(e.target.value)} />
          </label>
          <button id="auto-on" className="green" disabled={!basis.trim()} onClick={() => toggle(true)}>
            켜기
          </button>
        </div>
      )}
      {error && <p className="error bar-error">{error}</p>}
      <div className="officer-body">
        <IntakeView officerId={officerId} onCounts={refresh} />
        {/* 설정이 닫혀 있어도 다운로드 이벤트를 받도록 항상 둡니다. */}
        <ModelSettings open={settingsOpen} onOpen={() => setSettingsOpen(true)} />
      </div>
      <footer className="tables">
        <span>엔진 {status?.engine ?? "-"}</span>
        <span>확인 대기 {status?.active ?? 0}건</span>
        <span>보관 문서 {status?.archived ?? 0}건</span>
        <span>합성 관계망 {status?.graph_nodes ?? 0}노드</span>
        <span>감사 기록 {status?.audit_count ?? 0}건</span>
        {status?.audit_last[0] && <span className="audit-tail">{status.audit_last[0]}</span>}
      </footer>
    </main>
  );
}
