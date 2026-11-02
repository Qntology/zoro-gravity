import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useEffect, useRef, useState } from "react";
import { CategoryKey, CategorySchema, EXCUSES, Field, SCHEMAS, schemaOf } from "./formSchemas";
import { SignaturePad, SignatureStats } from "./SignaturePad";

const STEPS = ["서류 넣기", "내용 확인", "결과"];
const RESET_SECONDS = 30;
/** 기다리다 자리를 뜬 민원인의 정보가 남지 않도록 처음 화면으로 돌아갑니다 */
const WAITING_RESET_SECONDS = 300;
/** 입력 중에 자리를 뜨면 앞 사람의 값이 화면에 남지 않게 합니다 */
const IDLE_RESET_SECONDS = 120;

type Phase = "start" | "scanning" | "pick" | "form" | "excuse" | "waiting" | "finished";
type Values = Record<string, string>;

interface ScanDoc {
  kind: string;
  signed: boolean;
}

interface ScanReply {
  category: CategoryKey | null;
  docs: ScanDoc[];
  prefill: Values;
  unknown_pages: number;
  unsure: string[];
}

interface KioskReply {
  state: "waiting" | "finished" | "invalid" | "sign_again" | "excuse";
  receipt_no: string | null;
  notice: string | null;
  message: string | null;
  locked: string[];
}

const STEP_OF: Record<Phase, number> = { start: 0, scanning: 0, pick: 0, form: 1, excuse: 1, waiting: 2, finished: 2 };

/** 민원인 화면. 서류를 넣으면 내용을 채워 주고, 돌아오는 것은 접수 번호와 교부 문서뿐입니다. 내부 정보는 받지 않습니다. */
export function KioskView() {
  const [ready, setReady] = useState<boolean | null>(null);
  const [phase, setPhase] = useState<Phase>("start");
  const [schema, setSchema] = useState<CategorySchema | null>(null);
  const [docs, setDocs] = useState<ScanDoc[]>([]);
  const [prefill, setPrefill] = useState<Values>({});
  const [values, setValues] = useState<Values>({});
  const [unknownPages, setUnknownPages] = useState(0);
  const [consent, setConsent] = useState(false);
  const [sig, setSig] = useState<SignatureStats | null>(null);
  const [padKey, setPadKey] = useState(0);
  const [signFails, setSignFails] = useState(0);
  const [stamp, setStamp] = useState(false);
  const [excuse, setExcuse] = useState("");
  const [excuseText, setExcuseText] = useState("");
  const [excuseMessage, setExcuseMessage] = useState("");
  const [locked, setLocked] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [receipt, setReceipt] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const receiptRef = useRef<string | null>(null);

  useEffect(() => {
    invoke<boolean>("kiosk_ready").then(setReady);
    const stopState = listen<{ ready: boolean }>("kiosk_state", (e) => setReady(e.payload.ready));
    // 공무원이 끝내면 교부 문서가 옵니다.
    const stopCase = listen<{ receipt_no: string; notice: string }>("case_update", (e) => {
      if (e.payload.receipt_no !== receiptRef.current) return;
      setNotice(e.payload.notice);
      setPhase("finished");
    });
    return () => {
      void stopState.then((f) => f());
      void stopCase.then((f) => f());
    };
  }, []);

  // 끝나면 잠시 뒤 처음 화면으로 돌아갑니다. 앞 민원인의 정보가 남지 않게 합니다.
  useEffect(() => {
    if (phase !== "finished" && phase !== "waiting") return;
    const timer = window.setTimeout(reset, (phase === "finished" ? RESET_SECONDS : WAITING_RESET_SECONDS) * 1000);
    return () => window.clearTimeout(timer);
  }, [phase]);

  useEffect(() => {
    if (phase !== "pick" && phase !== "form" && phase !== "excuse") return;
    let timer = window.setTimeout(reset, IDLE_RESET_SECONDS * 1000);
    const touched = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(reset, IDLE_RESET_SECONDS * 1000);
    };
    window.addEventListener("pointerdown", touched);
    window.addEventListener("keydown", touched);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener("pointerdown", touched);
      window.removeEventListener("keydown", touched);
    };
  }, [phase]);

  function reset() {
    void invoke("kiosk_reset");
    receiptRef.current = null;
    setPhase("start");
    setSchema(null);
    setDocs([]);
    setPrefill({});
    setValues({});
    setUnknownPages(0);
    setConsent(false);
    setSig(null);
    setSignFails(0);
    setStamp(false);
    setExcuse("");
    setExcuseText("");
    setLocked([]);
    setError(null);
    setReceipt(null);
    setNotice(null);
  }

  async function scan() {
    setPhase("scanning");
    setError(null);
    try {
      const reply = await invoke<ScanReply>("scan_documents");
      setDocs((d) => [...d, ...reply.docs]);
      setPrefill((p) => ({ ...p, ...reply.prefill }));
      setValues((v) => ({ ...reply.prefill, ...v }));
      setUnknownPages((n) => n + reply.unknown_pages);
      if (schema) {
        setPhase("form");
      } else if (reply.category) {
        setSchema(schemaOf(reply.category));
        setPhase("form");
      } else {
        setPhase("pick");
      }
    } catch (e) {
      setError(String(e));
      setPhase(schema ? "form" : "start");
    }
  }

  const paperConsent = docs.some((d) => d.kind === "금융정보동의서" && d.signed);
  const needsConsent = !!schema?.consent && !paperConsent;
  const needsSign = !!schema?.sign && !stamp && !(schema.key === "welfare" && paperConsent);

  async function submit(chosenExcuse: string | null) {
    if (!schema) return;
    setBusy(true);
    setError(null);
    const edited = Object.keys(prefill).filter((k) => prefill[k] !== undefined && String(prefill[k]) !== String(values[k] ?? ""));
    const payload: Record<string, unknown> = {
      ...values,
      scanned: docs,
      edited,
      consent: consent || paperConsent,
      stamp_requested: stamp,
    };
    if (sig) payload.signature = sig;
    if (chosenExcuse) {
      payload.excuse = chosenExcuse;
      payload.excuse_text = excuseText;
    }
    try {
      const reply = await invoke<KioskReply>("submit_form", { input: { category: schema.key, values: payload } });
      switch (reply.state) {
        case "waiting":
          receiptRef.current = reply.receipt_no;
          setReceipt(reply.receipt_no);
          setPhase("waiting");
          break;
        case "finished":
          receiptRef.current = reply.receipt_no;
          setReceipt(reply.receipt_no);
          setNotice(reply.notice);
          setPhase("finished");
          break;
        case "excuse":
          setExcuseMessage(reply.message ?? "");
          setLocked(reply.locked);
          setExcuse("");
          setPhase("excuse");
          break;
        case "sign_again":
          // 서명이 기준에 못 미치면 패드를 비우고 다시 받습니다.
          setError(reply.message);
          setSig(null);
          setPadKey((k) => k + 1);
          setSignFails((n) => n + 1);
          setPhase("form");
          break;
        default:
          setError(reply.message);
          setPhase("form");
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  const missing = schema ? firstMissing(schema, values, needsConsent && !consent, needsSign && !sig) : null;

  return (
    <main className="kiosk">
      <header className="kiosk-header">스마트 민원 창구</header>
      <ol className="steps">
        {STEPS.map((step, i) => (
          <li key={step} className={ready && i === STEP_OF[phase] ? "active" : undefined}>
            {i + 1}. {step}
          </li>
        ))}
      </ol>

      {ready !== true ? (
        <section className="bot-card">
          <h1>안내</h1>
          <p>창구를 준비하고 있습니다. 잠시만 기다려 주세요.</p>
        </section>
      ) : phase === "start" ? (
        <section className="kiosk-card start">
          <h2>민원 서류나 신분증을 스캐너에 넣어 주세요</h2>
          <p className="lead">서류를 읽어 내용을 채워 드립니다. 서류가 없어도 신청할 수 있습니다.</p>
          {error && <p className="error">{error}</p>}
          <div className="start-actions">
            <button id="scan" className="big green" onClick={scan}>
              스캔 시작
            </button>
            <button id="no-docs" className="big plain" onClick={() => setPhase("pick")}>
              서류 없이 신청
            </button>
          </div>
        </section>
      ) : phase === "scanning" ? (
        <section className="bot-card">
          <div className="spinner" />
          <p>서류를 읽고 있습니다</p>
          <h1>한 장에 10초쯤 걸립니다</h1>
        </section>
      ) : phase === "pick" ? (
        <section className="kiosk-card">
          <h2>어떤 민원인가요?</h2>
          <div className="choices">
            {SCHEMAS.map((c) => (
              <button
                key={c.key}
                id={`pick-${c.key}`}
                className="choice"
                onClick={() => {
                  setSchema(c);
                  setPhase("form");
                }}
              >
                <strong>{c.label}</strong>
                <span>{c.blurb}</span>
              </button>
            ))}
          </div>
          <div className="actions">
            <button className="plain" onClick={reset}>
              처음으로
            </button>
          </div>
        </section>
      ) : phase === "form" && schema ? (
        <section className="kiosk-card">
          <h2>{schema.label} 내용을 확인해 주세요</h2>
          {docs.length > 0 && (
            <div className="chips">
              {docs.map((d, i) => (
                <span key={`${d.kind}-${i}`} className="chip">
                  {d.kind}
                </span>
              ))}
            </div>
          )}
          {unknownPages > 0 && <p className="muted">알아보지 못한 서류 {unknownPages}장은 창구에서 확인합니다.</p>}
          <div className="form-grid">
            {schema.fields.map((f) => (
              <Input key={f.key} field={f} value={values[f.key] ?? ""} scanned={prefill[f.key] !== undefined} onChange={(v) => setValues((prev) => ({ ...prev, [f.key]: v }))} />
            ))}
          </div>
          {needsConsent && (
            <label className="check">
              <input id="consent" type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              금융정보 조회에 동의합니다
            </label>
          )}
          {needsSign && (
            <div className="signature-field">
              <span className="field-label">서명 (성함을 알아볼 수 있게)</span>
              <SignaturePad key={padKey} onChange={setSig} />
            </div>
          )}
          {signFails >= 2 && !stamp && (
            <button id="stamp" className="plain" onClick={() => setStamp(true)}>
              서명이 어려우면 창구에서 날인하기
            </button>
          )}
          {stamp && <p className="muted">접수 뒤 창구에서 직원과 함께 날인합니다.</p>}
          {error && <p className="error">{error}</p>}
          <div className="actions">
            <button className="plain" disabled={busy} onClick={reset}>
              처음으로
            </button>
            <button id="scan-more" className="plain" disabled={busy} onClick={scan}>
              서류 더 넣기
            </button>
            <button id="submit" className="green" disabled={busy || missing !== null} onClick={() => submit(null)}>
              {busy ? "접수 중입니다" : missing ? `${missing} 칸을 채워 주세요` : "접수하기"}
            </button>
          </div>
        </section>
      ) : phase === "excuse" ? (
        <section className="kiosk-card">
          <h2>확인이 필요합니다</h2>
          <p className="lead">{excuseMessage}</p>
          <div className="excuses">
            {EXCUSES.filter((e) => !locked.includes(e.value)).map((e) => (
              <label key={e.value} className="check">
                <input type="radio" name="excuse" value={e.value} checked={excuse === e.value} onChange={() => setExcuse(e.value)} />
                {e.label}
              </label>
            ))}
          </div>
          {excuse === "free_text" && <textarea rows={3} maxLength={300} value={excuseText} placeholder="내용을 적어 주세요" onChange={(e) => setExcuseText(e.target.value)} />}
          {error && <p className="error">{error}</p>}
          <div className="actions">
            <button className="plain" disabled={busy} onClick={() => setPhase("form")}>
              이전
            </button>
            <button id="excuse-submit" className="green" disabled={busy || !excuse || (excuse === "free_text" && !excuseText.trim())} onClick={() => submit(excuse)}>
              {busy ? "접수 중입니다" : "접수하기"}
            </button>
          </div>
        </section>
      ) : phase === "waiting" ? (
        <section className="kiosk-card done">
          <h2>접수되었습니다</h2>
          <p className="receipt">
            접수 번호 <strong>{receipt}</strong>
          </p>
          <p className="lead">담당 주무관이 확인하고 있습니다. 끝나면 이 화면에 결과가 나옵니다.</p>
          <div className="spinner" />
          <div className="actions">
            <button className="plain" onClick={reset}>
              처음으로
            </button>
          </div>
        </section>
      ) : (
        <section className="kiosk-card done">
          <h2>처리가 끝났습니다</h2>
          <p className="receipt">
            접수 번호 <strong>{receipt}</strong>
          </p>
          <pre className="notice-doc">{notice}</pre>
          <p className="muted">{RESET_SECONDS}초 뒤 처음 화면으로 돌아갑니다.</p>
          <div className="actions">
            <button id="finish" className="green" onClick={reset}>
              처음으로
            </button>
          </div>
        </section>
      )}
    </main>
  );
}

/** 아직 적지 않은 항목의 이름. 서버 검사가 최종이고, 이 검사는 빠진 칸을 먼저 알려 주는 용도입니다. */
function firstMissing(schema: CategorySchema, values: Values, needsConsent: boolean, needsSign: boolean): string | null {
  for (const f of schema.fields) {
    if (!f.optional && !String(values[f.key] ?? "").trim()) return f.label;
  }
  if (needsConsent) return "금융정보 동의";
  if (needsSign) return "서명";
  return null;
}

function Input({ field: f, value, scanned, onChange }: { field: Field; value: string; scanned: boolean; onChange: (v: string) => void }) {
  const id = `f-${f.key}`;
  const label = (
    <span>
      {f.label}
      {scanned && <em className="from-scan">서류에서 읽음</em>}
    </span>
  );
  switch (f.kind) {
    case "birth":
      return (
        <label>
          {label}
          <input id={id} inputMode="numeric" value={value} maxLength={6} placeholder="예: 990101" onChange={(e) => onChange(e.target.value.replace(/\D/g, ""))} />
        </label>
      );
    case "money":
    case "count":
      return (
        <label>
          {label}
          <input id={id} inputMode="numeric" value={value} placeholder="숫자만 적어 주세요" onChange={(e) => onChange(e.target.value.replace(/[^\d,]/g, ""))} />
        </label>
      );
    case "date":
      return (
        <label>
          {label}
          <input id={id} type="date" value={value} onChange={(e) => onChange(e.target.value)} />
        </label>
      );
    case "choice":
      return (
        <label>
          {label}
          <select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
            <option value="">고르세요</option>
            {f.options?.map((o) => (
              <option key={o} value={o}>
                {o}
              </option>
            ))}
          </select>
        </label>
      );
    case "long":
      return (
        <label className="wide">
          {label}
          <textarea id={id} value={value} maxLength={f.max} rows={7} onChange={(e) => onChange(e.target.value)} />
        </label>
      );
    default:
      return (
        <label>
          {label}
          <input id={id} inputMode={f.kind === "tel" ? "tel" : undefined} value={value} maxLength={f.max ?? 30} placeholder={f.hint} onChange={(e) => onChange(e.target.value)} />
        </label>
      );
  }
}
