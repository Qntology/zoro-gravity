import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useRef, useState } from "react";

interface IntakeRow {
  receipt_no: string;
  name: string | null;
  category: string | null;
  state: "awaiting" | "transferred" | "finalized";
  received_at: string | null;
  suggested_department: string | null;
  department: string | null;
  officer_id: string | null;
  transferred_at: string | null;
  archive_id: string | null;
  final_code: string | null;
  auto: boolean;
}

interface Summary {
  gist: string;
  demand: string;
  department: string | null;
  verified: boolean;
  problems: string[];
}

interface Department {
  name: string;
  description: string;
}

interface DocRow {
  kind: string;
  how: string;
  done: boolean;
}

interface IntakeDetail {
  row: IntakeRow;
  kind: "grievance" | "confirm";
  category_label: string;
  birth_prefix: string;
  contact: string | null;
  body: string;
  lines: { label: string; value: string }[];
  reason: string;
  candidate: string | null;
  candidate_class: "favorable" | "adverse" | "procedural" | null;
  confidence: number | null;
  docs: DocRow[];
  pending: string[];
  identity_done: boolean;
  relations: string[];
  claimed_total: number | null;
  verified_sum: number | null;
  needs: string[];
  summary: Summary | null;
  departments: Department[];
}

const CANDIDATE_LABEL: Record<string, string> = {
  G3_ELIGIBLE_NORMAL: "지급 · 승인 요건 충족",
  G3_ELIGIBLE_HOMONYM_CLEAR: "동명이인 확인됨, 요건 충족",
  G3_VIOLATION_HIGH_RISK: "위반 의심 (고위험)",
  G3_VIOLATION_RELATED_VENDOR_SPLIT: "특수관계 거래 · 분할 결제 의심",
  G3_HOLD_SIMPLE_RECEIPT_MISSING: "증빙 부족 (보완 대상)",
  G1_EXPIRED_MANDATORY_REJECT: "위임장 유효기간 초과",
  G2_REQUIRE_REPAIR_RECEIPT: "수리비 영수증 필요",
};

/** 처리가 끝난 건의 결과 이름. 처분 코드는 감사 기록에 그대로 남습니다. */
const RESULT_LABEL: Record<string, string> = {
  G3_ELIGIBLE_NORMAL: "승인",
  G3_ELIGIBLE_HOMONYM_CLEAR: "승인 (동명이인 확인)",
  G3_VIOLATION_HIGH_RISK: "반려",
  G3_VIOLATION_RELATED_VENDOR_SPLIT: "반려",
  G1_EXPIRED_MANDATORY_REJECT: "반려 (위임장 기간 초과)",
  SUPPLEMENT_7DAYS: "보완요구 7일",
};

const won = (n: number | null | undefined) => (n == null ? "-" : `${n.toLocaleString()}원`);
const digits = (s: string) => s.replace(/[^\d]/g, "");
const toNumber = (s: string) => (digits(s) ? Number(digits(s)) : null);

/** 공무원 콘솔의 접수 목록과 상세. 사유를 먼저 보이고, 남은 확인만 받고, 후보에 맞는 처리만 보입니다. */
export function IntakeView({ officerId, onCounts }: { officerId: string; onCounts: () => void }) {
  const [rows, setRows] = useState<IntakeRow[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<IntakeDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const asked = useRef(0);

  const load = useCallback(() => {
    invoke<IntakeRow[]>("intake_list")
      .then(setRows)
      .catch((e) => setError(String(e)));
    onCounts();
  }, [onCounts]);

  useEffect(() => {
    load();
    const stop = listen("intake_changed", () => load());
    return () => void stop.then((f) => f());
  }, [load]);

  /** 늦게 도착한 앞선 응답이 새 선택을 덮지 않도록 요청 번호를 맞춰 봅니다. */
  const open = useCallback((receipt: string | null) => {
    const ticket = ++asked.current;
    setSelected(receipt);
    setDetail(null);
    setError(null);
    if (!receipt) return;
    invoke<IntakeDetail>("intake_detail", { receiptNo: receipt })
      .then((d) => {
        if (ticket === asked.current) setDetail(d);
      })
      .catch((e) => {
        if (ticket === asked.current) setError(String(e));
      });
  }, []);

  const finished = (message: string) => {
    setNotice(message);
    open(null);
    load();
  };

  const awaiting = rows.filter((r) => r.state === "awaiting");
  const closed = rows.filter((r) => r.state !== "awaiting");

  return (
    <section className="intake">
      <aside className="intake-list">
        {notice && <p className="notice">{notice}</p>}
        <h3>확인 대기 ({awaiting.length})</h3>
        {awaiting.length === 0 && <p className="muted">확인 대기 건이 없습니다.</p>}
        {awaiting.map((r) => (
          <button key={r.receipt_no} className={`intake-item${selected === r.receipt_no ? " selected" : ""}`} onClick={() => open(r.receipt_no)}>
            <strong>{r.name}</strong>
            <span>
              {r.category} · {r.receipt_no}
            </span>
            {r.suggested_department && <span>부서 후보: {r.suggested_department}</span>}
          </button>
        ))}
        <h3>처리 완료 ({closed.length})</h3>
        {closed.map((r) => (
          <div key={r.receipt_no} className="intake-item done">
            <strong>
              {r.receipt_no} {r.auto && <em className="tag auto">자동 확정</em>}
            </strong>
            <span>{r.state === "transferred" ? `이첩 · ${r.department}` : `${RESULT_LABEL[r.final_code ?? ""] ?? r.final_code} · ${r.archive_id}`}</span>
          </div>
        ))}
      </aside>

      <article className="intake-detail">
        {!selected && <p className="muted">왼쪽에서 접수를 고르세요.</p>}
        {error && !detail && <p className="error">{error}</p>}
        {selected && !detail && !error && <p className="muted">불러오는 중입니다.</p>}
        {detail &&
          (detail.kind === "grievance" ? (
            <GrievancePanel key={detail.row.receipt_no} detail={detail} officerId={officerId} busy={busy} setBusy={setBusy} onDone={finished} />
          ) : (
            <CasePanel key={detail.row.receipt_no} detail={detail} officerId={officerId} busy={busy} setBusy={setBusy} onDone={finished} onChanged={(text) => { setNotice(text); open(detail.row.receipt_no); }} />
          ))}
      </article>
    </section>
  );
}

interface PanelProps {
  detail: IntakeDetail;
  officerId: string;
  busy: boolean;
  setBusy: (b: boolean) => void;
  onDone: (message: string) => void;
}

function Head({ detail }: { detail: IntakeDetail }) {
  return (
    <>
      <h2>
        {detail.row.name} <span className="muted">· {detail.category_label} · {detail.row.receipt_no}</span>
      </h2>
      <p className="muted">
        생년월일 {detail.birth_prefix} · 접수 {detail.row.received_at}
        {detail.contact ? ` · 연락처 ${detail.contact}` : ""}
      </p>
    </>
  );
}

/** 고충민원: 본문을 읽고 부서를 정해 이첩합니다. 이미 끝난 확인은 묻지 않습니다. */
function GrievancePanel({ detail, officerId, busy, setBusy, onDone }: PanelProps) {
  const [read, setRead] = useState(false);
  const [identity, setIdentity] = useState(false);
  const [department, setDepartment] = useState(detail.row.suggested_department ?? "");
  const [error, setError] = useState<string | null>(null);
  const identityDone = detail.identity_done || identity;
  const missing = !officerId.trim() ? "담당자 사번" : !identityDone ? "본인 확인" : !read ? "원문 확인" : !department ? "담당 부서" : null;

  async function transfer() {
    setBusy(true);
    setError(null);
    try {
      const checklist = { identity_checked: true, contact_checked: true, body_read: read, department_checked: true };
      const row = await invoke<IntakeRow>("transfer_intake", { input: { receipt_no: detail.row.receipt_no, officer_id: officerId, department, checklist } });
      onDone(`${row.receipt_no} 이첩 완료: ${row.department}`);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <Head detail={detail} />
      {detail.summary && (
        <section className="callout">
          <strong>{detail.summary.gist}</strong>
          <span>요구: {detail.summary.demand}</span>
          {!detail.summary.verified && <span className="error">요약 확인 필요: {detail.summary.problems.join(", ")}</span>}
        </section>
      )}
      <section>
        <h4>민원 원문</h4>
        <div className="intake-body">{detail.body}</div>
      </section>
      <section className="todo">
        {!detail.identity_done && (
          <label className="check">
            <input id="check-identity" type="checkbox" checked={identity} onChange={(e) => setIdentity(e.target.checked)} />
            신분증으로 본인을 확인했습니다
          </label>
        )}
        <label className="check">
          <input id="check-body" type="checkbox" checked={read} onChange={(e) => setRead(e.target.checked)} />
          원문을 읽고 요약의 내용을 맞춰 봤습니다
        </label>
        <label className="field">
          담당 부서
          <select id="department" value={department} onChange={(e) => setDepartment(e.target.value)}>
            <option value="">고르세요</option>
            {detail.departments.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name} ({d.description})
              </option>
            ))}
          </select>
        </label>
      </section>
      {error && <p className="error">{error}</p>}
      <div className="actions">
        <button id="primary" className="green" disabled={busy || missing !== null} onClick={transfer}>
          {busy ? "처리 중입니다" : missing ? `${missing} 필요` : "담당 부서로 이첩"}
        </button>
      </div>
    </>
  );
}

/** 복지 · 지원금 · 허가 · 증명: 사유 → 남은 확인 → 값 입력 → 후보에 맞는 처리. */
function CasePanel({ detail, officerId, busy, setBusy, onDone, onChanged }: PanelProps & { onChanged: (text: string) => void }) {
  const [identity, setIdentity] = useState(false);
  const [docsSeen, setDocsSeen] = useState<Record<string, boolean>>({});
  const [pendingSeen, setPendingSeen] = useState(false);
  const [contract, setContract] = useState("");
  const [transfer, setTransfer] = useState("");
  const [account, setAccount] = useState("");
  const [landlordName, setLandlordName] = useState("");
  const [landlordBirth, setLandlordBirth] = useState("");
  const [evidence, setEvidence] = useState("");
  const [approved, setApproved] = useState(detail.verified_sum != null ? String(detail.verified_sum) : "");
  const [risk, setRisk] = useState("");
  const [scans, setScans] = useState("");
  const [cosign, setCosign] = useState(false);
  const [ruleRef, setRuleRef] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const undone = detail.docs.filter((d) => !d.done);
  const adverse = detail.candidate_class === "adverse";
  const procedural = detail.candidate_class === "procedural";
  const short = detail.candidate === "G3_HOLD_SIMPLE_RECEIPT_MISSING";
  const ownerId = officerId.trim();

  const missing = (() => {
    if (!ownerId) return "담당자 사번";
    if (!detail.identity_done && !identity) return "본인 확인";
    if (undone.some((d) => !docsSeen[d.kind])) return "서류 원본 확인";
    if (detail.pending.length > 0 && !pendingSeen) return "확인할 일";
    if (detail.needs.includes("lease") && (!contract || !transfer)) return "월세 · 이체액";
    if (detail.needs.includes("bank") && !account.trim()) return "통장 계좌";
    if (detail.needs.includes("landlord") && (!landlordName.trim() || landlordBirth.length !== 6)) return "임대인";
    return null;
  })();

  const counter = (extra: Record<string, unknown> = {}) => ({
    receipt_no: detail.row.receipt_no,
    officer_id: ownerId,
    docs: undone.filter((d) => docsSeen[d.kind]).map((d) => d.kind),
    identity_checked: identity,
    pending_checked: pendingSeen,
    contract_rent: toNumber(contract),
    latest_transfer: toNumber(transfer),
    printed_account: account || null,
    landlord_name: landlordName || null,
    landlord_birth: landlordBirth || null,
    approved_amount: null,
    evidence_sum: toNumber(evidence),
    ...extra,
  });

  async function run(task: () => Promise<string>) {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      onDone(await task());
    } catch (e) {
      const text = String(e);
      if (text.includes("후보 처분이 바뀌었습니다")) {
        onChanged(text);
      } else {
        setError(text);
      }
    } finally {
      setBusy(false);
    }
  }

  const confirm = (extra?: Record<string, unknown>) =>
    run(async () => {
      const archive = await invoke<string>("confirm_form", { input: counter(extra) });
      return `${detail.row.receipt_no} 처리 완료 (문서 ${archive})`;
    });

  const action = (name: string, extra: Record<string, unknown> = {}) =>
    run(async () => {
      const archive = await invoke<string>("officer_action", { input: { receipt_no: detail.row.receipt_no, officer_id: ownerId, action: name, ...extra } });
      return `${detail.row.receipt_no} 처리 완료 (문서 ${archive})`;
    });

  async function unmask() {
    setError(null);
    try {
      const allowed = await invoke<boolean>("request_unmask", { receiptNo: detail.row.receipt_no, officerId: ownerId });
      setMessage(allowed ? "열람이 허용되어 감사 기록에 남았습니다." : "위험도가 낮아 열람이 거부되었고 감사 기록에 남았습니다.");
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <>
      <Head detail={detail} />
      <section className={`callout ${detail.candidate_class ?? "none"}`}>
        <strong>
          {detail.candidate ? (CANDIDATE_LABEL[detail.candidate] ?? detail.candidate) : "규칙 판단 전"}
          {detail.confidence != null && <em className="tag">확신도 {detail.confidence.toFixed(2)}</em>}
        </strong>
        <span>{detail.reason}</span>
      </section>
      {message && <p className="notice">{message}</p>}

      <section>
        <h4>서류</h4>
        <div className="chips">
          {detail.docs.map((d) => (
            <span key={d.kind} className={`chip ${d.done ? "" : "open"}`}>
              {d.kind} · {d.how}
            </span>
          ))}
        </div>
      </section>

      {(!detail.identity_done || undone.length > 0 || detail.pending.length > 0) && (
        <section className="todo">
          <h4>확인할 일</h4>
          {!detail.identity_done && (
            <label className="check">
              <input id="check-identity" type="checkbox" checked={identity} onChange={(e) => setIdentity(e.target.checked)} />
              신분증으로 본인을 확인했습니다
            </label>
          )}
          {undone.map((d) => (
            <label key={d.kind} className="check">
              <input type="checkbox" className="check-doc" checked={!!docsSeen[d.kind]} onChange={(e) => setDocsSeen((s) => ({ ...s, [d.kind]: e.target.checked }))} />
              {d.kind} 원본을 확인했습니다
            </label>
          ))}
          {detail.pending.length > 0 && (
            <>
              <ul className="pending">
                {detail.pending.map((p) => (
                  <li key={p}>{p}</li>
                ))}
              </ul>
              <label className="check">
                <input id="check-pending" type="checkbox" checked={pendingSeen} onChange={(e) => setPendingSeen(e.target.checked)} />
                위 내용을 원본과 맞춰 봤습니다
              </label>
            </>
          )}
        </section>
      )}

      {detail.needs.length > 0 && (
        <section className="intake-form">
          {detail.needs.includes("lease") && (
            <>
              <label>
                계약서 월세 (원)
                <input id="in-contract" inputMode="numeric" value={contract} onChange={(e) => setContract(e.target.value)} />
              </label>
              <label>
                최근 이체액 (원)
                <input id="in-transfer" inputMode="numeric" value={transfer} onChange={(e) => setTransfer(e.target.value)} />
              </label>
            </>
          )}
          {detail.needs.includes("bank") && (
            <label>
              통장 계좌
              <input id="in-account" value={account} onChange={(e) => setAccount(e.target.value)} />
            </label>
          )}
          {detail.needs.includes("landlord") && (
            <>
              <label>
                임대인 성명
                <input id="in-landlord" value={landlordName} onChange={(e) => setLandlordName(e.target.value)} />
              </label>
              <label>
                임대인 생년월일 앞 6자리
                <input id="in-landlord-birth" inputMode="numeric" maxLength={6} value={landlordBirth} onChange={(e) => setLandlordBirth(digits(e.target.value))} />
              </label>
            </>
          )}
        </section>
      )}

      {detail.claimed_total != null && (
        <section>
          <h4>정산</h4>
          <p>
            신청 {won(detail.claimed_total)} · 인정 합계 {won(detail.verified_sum)}
          </p>
        </section>
      )}

      {detail.relations.length > 0 && (
        <section>
          <h4>관계망 연결 (가림 처리)</h4>
          <ul className="pending">
            {detail.relations.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
          <div>
            <button id="unmask" className="plain" disabled={!ownerId} onClick={unmask}>
              실명 열람 요청
            </button>
          </div>
        </section>
      )}

      <details className="lines">
        <summary>접수 내용 전체</summary>
        <dl className="intake-dl">
          {detail.lines.map((l, i) => (
            <div key={`${l.label}-${i}`} className="intake-line">
              <dt>{l.label}</dt>
              <dd className="pre">{l.value}</dd>
            </div>
          ))}
        </dl>
      </details>

      {short && (
        <section className="intake-form">
          <label>
            인정 금액 (원)
            <input id="in-approved" inputMode="numeric" value={approved} onChange={(e) => setApproved(e.target.value)} />
          </label>
          <label>
            창구에서 추가 확인한 증빙 (원)
            <input id="in-evidence" inputMode="numeric" value={evidence} onChange={(e) => setEvidence(e.target.value)} />
          </label>
        </section>
      )}

      {adverse && (
        <section className="intake-form">
          <label>
            승인 사유 (승인할 때만)
            <select id="risk" value={risk} onChange={(e) => setRisk(e.target.value)}>
              <option value="">고르세요</option>
              <option value="approve_evidence">오판 입증 (원본 증빙 첨부)</option>
              <option value="approve_exception">조례 예외 적용 (팀장 공동 결재)</option>
              <option value="approve_cancel">승인 취소 (반려 유지)</option>
            </select>
          </label>
          {risk === "approve_evidence" && (
            <label>
              첨부한 원본 증빙 장수
              <input id="in-scans" inputMode="numeric" value={scans} onChange={(e) => setScans(digits(e.target.value))} />
            </label>
          )}
          {risk === "approve_exception" && (
            <>
              <label>
                근거 조항
                <input id="in-rule" value={ruleRef} onChange={(e) => setRuleRef(e.target.value)} />
              </label>
              <label className="check">
                <input id="check-cosign" type="checkbox" checked={cosign} onChange={(e) => setCosign(e.target.checked)} />
                팀장 공동 결재를 받았습니다
              </label>
            </>
          )}
        </section>
      )}

      {error && <p className="error">{error}</p>}
      <div className="actions">
        {adverse && (
          <button id="approve" className="plain" disabled={busy || !risk || !ownerId} onClick={() => action(risk, { evidence_scans: toNumber(scans), team_lead_cosign: cosign, rule_ref: ruleRef })}>
            고위험 승인
          </button>
        )}
        {(procedural || undone.length > 0) && (
          <button id="supplement" className="plain" disabled={busy || !ownerId} onClick={() => action("supplement")}>
            보완요구
          </button>
        )}
        {short ? (
          <button id="primary" className="green" disabled={busy || missing !== null || !approved} onClick={() => confirm({ approved_amount: toNumber(approved) })}>
            {busy ? "처리 중입니다" : missing ? `${missing} 필요` : "감액 정산"}
          </button>
        ) : (
          <button id="primary" className={adverse ? "danger-fill" : "green"} disabled={busy || missing !== null} onClick={() => confirm()}>
            {busy ? "처리 중입니다" : missing ? `${missing} 필요` : adverse ? "반려 확정" : "확정"}
          </button>
        )}
      </div>
    </>
  );
}
