import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useEffect, useState } from "react";

const STEPS = ["서류 및 서명 확인", "내용 대조", "완료"];

/** 민원인 화면. 준비 여부 말고는 내부 정보를 받지 않습니다. */
export function KioskView() {
  const [ready, setReady] = useState<boolean | null>(null);

  useEffect(() => {
    invoke<boolean>("kiosk_ready").then(setReady);
    const stop = listen<{ ready: boolean }>("kiosk_state", (e) => setReady(e.payload.ready));
    return () => void stop.then((f) => f());
  }, []);

  return (
    <main className="kiosk">
      <header className="kiosk-header">스마트 민원 창구</header>
      <ol className="steps">
        {STEPS.map((step, i) => (
          <li key={step}>
            STEP {i + 1}. {step}
          </li>
        ))}
      </ol>
      <section className="bot-card">
        <h1>bot</h1>
        <p>{ready === false ? "창구를 준비하고 있습니다. 잠시만 기다려 주세요." : "서류를 스캐너에 넣어 주세요."}</p>
      </section>
    </main>
  );
}
