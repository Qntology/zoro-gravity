import { useState } from "react";
import { ModelSettings } from "./ModelSettings";

const GATES = ["Gate 0 분류", "Gate 1 형식심사", "Gate 2 정합성", "Gate 3 실질심사", "Gate 4 공문서", "Gate 5 이관"];
const TABLES = ["tb_active_session", "tb_official_archive", "tb_graph_entities", "tb_audit_log"];

/** 공무원 콘솔. 심사 화면은 다음 단계에서 이 틀에 붙입니다. */
export function OfficerView() {
  const [settingsOpen, setSettingsOpen] = useState(false);

  return (
    <main className="officer">
      <header className="officer-header">
        <strong>zoro-gravity 심사 콘솔</strong>
        <button className="plain" onClick={() => setSettingsOpen((open) => !open)}>
          {settingsOpen ? "설정 닫기" : "설정"}
        </button>
      </header>
      <ol className="gates">
        {GATES.map((gate) => (
          <li key={gate}>{gate}</li>
        ))}
      </ol>
      <div className="officer-body">
        <section className="empty">진행 중인 민원이 없습니다.</section>
        {/* 설정이 닫혀 있어도 다운로드 이벤트를 받도록 항상 둡니다. */}
        <ModelSettings open={settingsOpen} onOpen={() => setSettingsOpen(true)} />
      </div>
      <footer className="tables">
        {TABLES.map((table) => (
          <span key={table}>{table}: 대기</span>
        ))}
      </footer>
    </main>
  );
}
