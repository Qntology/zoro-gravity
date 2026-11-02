// 실제 앱의 민원인 · 공무원 창을 끝까지 돌립니다. --record <폴더> 면 두 창을 함께 녹화합니다.
//
//   npm i --no-save playwright-core@1.55.0
//   node l4.js [--record rec] [--only A1,O1] [--pace 1] [--out 결과.json]
//
// 앱은 먼저 띄워 둡니다. WebView2 에 디버그 포트를 열고, 스캐너 폴더를 같은 값으로 줍니다.
//   WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222 --disable-features=CalculateNativeWinOcclusion --disable-backgrounding-occluded-windows --disable-renderer-backgrounding
//   ZOROGRAVITY_SCANNER_DIR=<폴더>
// 서류 이미지는 `python scripts/gen_case_docs.py out/cases 11` 로 만듭니다. 결과는 out/l4-results.json 에 씁니다.
const { chromium } = require("playwright-core");
const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "../..");
const SCAN_DIR = process.env.ZOROGRAVITY_SCANNER_DIR;
const args = process.argv.slice(2);
const flag = (name) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : null;
};
const REC = flag("--record");
const ONLY = (flag("--only") || "").split(",").filter(Boolean);
const PACE = Number(flag("--pace") || (REC ? 1 : 0));
const OUT = flag("--out") || path.join(ROOT, "out", "l4-results.json");

const scenarios = JSON.parse(fs.readFileSync(`${ROOT}/core/fixtures/scenarios.json`, "utf8")).scenarios;
const scenario = (id) => scenarios.find((s) => s.id === id);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pause = (ms) => (PACE ? sleep(ms * PACE) : Promise.resolve());

const results = [];
const timeline = [];
// 흐름마다 사람이 한 동작 수를 셉니다 (보고서의 절차 비교).
let KIOSK = null;
let actions = { kiosk: 0, officer: 0 };
const count = (page) => (page === KIOSK ? actions.kiosk++ : actions.officer++);
const t0 = Date.now();
let currentFlow = "";
const mark = (track, screen, label, extra = {}) => timeline.push({ t: (Date.now() - t0) / 1000, flow: currentFlow, track, screen, label, ...extra });

const FORBIDDEN = ["모친", "관계망", "G1_", "G2_", "G3_", "G4_", "위반 의심", "확신도", "촌", "동일 세대"];

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

async function connect(deadlineMs = 300000) {
  const start = Date.now();
  while (Date.now() - start < deadlineMs) {
    try {
      return await chromium.connectOverCDP("http://127.0.0.1:9222");
    } catch {
      await sleep(2000);
    }
  }
  throw new Error("CDP 연결 시간 초과");
}

async function findPage(browser, marker, deadlineMs = 300000) {
  const start = Date.now();
  while (Date.now() - start < deadlineMs) {
    for (const ctx of browser.contexts()) {
      for (const p of ctx.pages()) {
        const text = await p.evaluate(() => document.body?.innerText || "").catch(() => "");
        if (text.includes(marker)) return p;
      }
    }
    await sleep(1500);
  }
  throw new Error(`${marker} 창을 찾지 못했습니다`);
}

const text = (page) => page.evaluate(() => document.body.innerText);

async function waitText(page, wanted, ms = 60000) {
  const list = Array.isArray(wanted) ? wanted : [wanted];
  const start = Date.now();
  while (Date.now() - start < ms) {
    const body = await text(page);
    const hit = list.find((w) => body.includes(w));
    if (hit) return hit;
    await sleep(200);
  }
  throw new Error(`기다린 글자가 없습니다: ${list.join(" | ")}\n${(await text(page)).slice(0, 400)}`);
}

// 터치 지점을 보이게 합니다 (녹화용). 민원 처리와 상관없는 표시입니다.
async function pointer(page) {
  if (!REC) return;
  await page.evaluate(() => {
    if (document.getElementById("rec-dot")) return;
    const d = document.createElement("div");
    d.id = "rec-dot";
    d.style.cssText = "position:fixed;left:-100px;top:-100px;width:26px;height:26px;border-radius:50%;background:rgba(0,123,255,.35);border:2px solid rgba(0,123,255,.85);z-index:99999;pointer-events:none;transform:translate(-50%,-50%);transition:transform .08s";
    document.body.appendChild(d);
    window.addEventListener("mousemove", (e) => {
      d.style.left = e.clientX + "px";
      d.style.top = e.clientY + "px";
    }, true);
    window.addEventListener("mousedown", () => (d.style.transform = "translate(-50%,-50%) scale(.7)"), true);
    window.addEventListener("mouseup", () => (d.style.transform = "translate(-50%,-50%) scale(1)"), true);
  });
}

async function moveTo(page, locator) {
  if (!REC) return;
  const box = await locator.boundingBox();
  if (box) await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, { steps: 14 });
  await sleep(120);
}

async function tap(page, selector) {
  count(page);
  const l = page.locator(selector).first();
  await l.waitFor({ timeout: 30000 });
  await moveTo(page, l);
  await l.click();
  await pause(250);
}

async function type(page, selector, value) {
  count(page);
  const l = page.locator(selector).first();
  await moveTo(page, l);
  await l.click();
  if (REC) {
    await l.fill("");
    await l.pressSequentially(String(value), { delay: 22 });
  } else {
    await l.fill(String(value));
  }
}

async function choose(page, selector, value) {
  count(page);
  const l = page.locator(selector).first();
  await moveTo(page, l);
  await l.selectOption(value);
  await pause(200);
}

async function sign(page) {
  count(page);
  const canvas = page.locator(".sign-canvas").first();
  await canvas.scrollIntoViewIfNeeded();
  const box = await canvas.boundingBox();
  const at = (x, y) => [box.x + box.width * x, box.y + box.height * y];
  const strokes = [
    [[0.15, 0.25], [0.28, 0.2], [0.2, 0.55], [0.34, 0.5]],
    [[0.36, 0.2], [0.38, 0.75]],
    [[0.46, 0.3], [0.58, 0.28], [0.5, 0.7], [0.62, 0.66]],
    [[0.66, 0.2], [0.8, 0.22], [0.7, 0.72], [0.85, 0.7]],
    [[0.2, 0.82], [0.8, 0.86]],
  ];
  for (const s of strokes) {
    await page.mouse.move(...at(...s[0]));
    await page.mouse.down();
    for (const p of s.slice(1)) await page.mouse.move(...at(...p), { steps: REC ? 12 : 4 });
    await page.mouse.up();
  }
}

/** 스캐너에 서류를 넣습니다. 앱은 폴더의 이미지를 읽고 지웁니다. */
function feed(s, mode) {
  const names = Object.keys(s.docs).sort();
  const keep = mode === "full" ? names : mode === "partial" ? names.slice(0, Math.ceil(names.length / 2)) : [];
  const list = mode === "none" ? [] : [...keep, "신분증"];
  fs.mkdirSync(SCAN_DIR, { recursive: true });
  list.forEach((name, i) => fs.copyFileSync(`${ROOT}/out/cases/${s.id}/${name}.jpg`, path.join(SCAN_DIR, `${String(i).padStart(2, "0")}-${name}.jpg`)));
  return list;
}

const fieldValue = (page, key) => page.locator(`#f-${key}`).first().inputValue().catch(() => null);

/** 민원인 쪽: 서류를 넣거나 민원을 고르고, 빈 칸을 채워 접수합니다. 돌아오는 상태를 알려 줍니다. */
async function citizen(kiosk, track, s, mode, opts = {}) {
  const stats = { pages: 0, prefilled: 0, typed: 0, fixed: 0 };
  await waitText(kiosk, "스캔 시작");
  await pause(500);
  mark(track, "kiosk", opts.opening ?? "민원인이 창구에 왔습니다");
  if (mode === "none") {
    await tap(kiosk, "#no-docs");
    await pause(300);
    await tap(kiosk, `#pick-${s.category}`);
  } else {
    stats.pages = feed(s, mode).length;
    mark(track, "kiosk", `스캐너에 서류 ${stats.pages}장을 넣고 스캔합니다`, { papers: s.id, wait: true });
    await tap(kiosk, "#scan");
    await waitText(kiosk, ["내용을 확인해 주세요", "어떤 민원인가요"], 300000);
    if ((await text(kiosk)).includes("어떤 민원인가요")) await tap(kiosk, `#pick-${s.category}`);
  }
  await waitText(kiosk, "내용을 확인해 주세요");
  mark(track, "kiosk", mode === "none" ? "서류 없이 꼭 필요한 값만 적습니다" : "서류에서 읽은 값이 채워져 있습니다");
  await pause(1200);

  const want = { ...s.form, applicant_name: s.applicant.name, applicant_birth: s.applicant.birth };
  const inputs = await kiosk.locator("[id^='f-']").evaluateAll((els) => els.map((e) => e.id.slice(2)));
  for (const key of inputs) {
    const target = want[key];
    if (target === undefined) continue;
    const now = await fieldValue(kiosk, key);
    const same = String(now ?? "").replace(/[, ]/g, "") === String(target).replace(/[, ]/g, "");
    if (same) {
      stats.prefilled++;
      continue;
    }
    if (now) stats.fixed++;
    else stats.typed++;
    const tag = await kiosk.locator(`#f-${key}`).first().evaluate((e) => e.tagName);
    if (tag === "SELECT") await choose(kiosk, `#f-${key}`, target);
    else await type(kiosk, `#f-${key}`, target);
  }
  if (await kiosk.locator("#consent").count()) await tap(kiosk, "#consent");
  if (await kiosk.locator(".sign-canvas").count()) {
    mark(track, "kiosk", "서명합니다");
    if (opts.weakSign) {
      const box = await kiosk.locator(".sign-canvas").first().boundingBox();
      await kiosk.mouse.move(box.x + 30, box.y + 40);
      await kiosk.mouse.down();
      await kiosk.mouse.move(box.x + 60, box.y + 40, { steps: 3 });
      await kiosk.mouse.up();
    } else {
      await sign(kiosk);
    }
    await pause(400);
  }
  mark(track, "kiosk", "접수합니다", { wait: true });
  await tap(kiosk, "#submit");
  if (opts.weakSign) {
    await waitText(kiosk, "서명", 60000);
    await waitText(kiosk, "칸을 채워 주세요", 120000).catch(() => {});
    stats.retried = true;
    await sign(kiosk);
    await tap(kiosk, "#submit");
  }

  let state = await waitText(kiosk, ["처리가 끝났습니다", "접수되었습니다", "확인이 필요합니다"], 300000);
  const excuses = s.excuses || [];
  let next = 0;
  while (state === "확인이 필요합니다") {
    const pick = excuses[next++] ?? "cash";
    mark(track, "kiosk", "차액을 물어 소명을 받습니다. 대장 값은 보이지 않습니다", { wait: true });
    await pause(1500);
    const body = await text(kiosk);
    FORBIDDEN.forEach((w) => assert(!body.includes(w), `키오스크에 내부 정보: ${w}`));
    await tap(kiosk, `input[value='${pick}']`);
    await pause(500);
    await tap(kiosk, "#excuse-submit");
    await sleep(300);
    // 답이 오기 전에는 같은 소명 화면이 그대로입니다.
    while ((await text(kiosk)).includes("접수 중입니다")) await sleep(200);
    await sleep(300);
    state = await waitText(kiosk, ["처리가 끝났습니다", "접수되었습니다", "확인이 필요합니다"], 300000);
  }
  const receipt = (await text(kiosk)).match(/[WSPCG]\d{8}-[0-9A-F]{6}/)?.[0];
  const body = await text(kiosk);
  FORBIDDEN.forEach((w) => assert(!body.includes(w), `키오스크에 내부 정보: ${w}`));
  return { state, receipt, stats };
}

async function openCase(officer, receipt) {
  count(officer);
  await waitText(officer, receipt, 30000);
  const row = officer.locator("button.intake-item", { hasText: receipt }).first();
  await row.waitFor({ timeout: 30000 });
  await moveTo(officer, row);
  await row.click();
  await waitText(officer, receipt);
  await officer.locator(".callout").first().waitFor({ timeout: 30000 });
  await pause(1200);
}

/** 공무원에게 남은 확인을 모두 합니다. 화면에 나온 것만 있습니다. */
async function clearChecks(officer) {
  for (const box of ["#check-identity", "#check-pending"]) {
    if (await officer.locator(box).count()) await tap(officer, box);
  }
  const docs = officer.locator("input.check-doc");
  for (let i = 0; i < (await docs.count()); i++) {
    count(officer);
    await moveTo(officer, docs.nth(i));
    await docs.nth(i).check();
  }
}

async function finishKiosk(kiosk, track, label) {
  await waitText(kiosk, "처리가 끝났습니다", 60000);
  mark(track, "kiosk", label);
  await pause(2200);
  const body = await text(kiosk);
  FORBIDDEN.forEach((w) => assert(!body.includes(w), `키오스크에 내부 정보: ${w}`));
  await tap(kiosk, "#finish");
  return body;
}

async function endWaiting(kiosk) {
  if ((await text(kiosk)).includes("스캔 시작")) return;
  if ((await text(kiosk)).includes("이전")) await kiosk.getByRole("button", { name: "이전" }).click({ timeout: 3000 }).catch(() => {});
  await sleep(300);
  await kiosk.getByRole("button", { name: "처음으로" }).click({ timeout: 3000 }).catch(() => {});
  await sleep(300);
}

const FLOWS = {
  // ---- 자동화 ----
  async A1(k, o) {
    const s = scenario("W1-clean");
    const r = await citizen(k, "auto", s, "full", { opening: "청년월세 신청: 서류 4장과 신분증" });
    assert(r.state === "처리가 끝났습니다", `자동 확정이 아닙니다: ${r.state}`);
    mark("auto", "kiosk", "사람 손을 거치지 않고 바로 지급 승인이 나옵니다");
    await pause(2500);
    const row = o.locator(".intake-item.done", { hasText: r.receipt }).first();
    await row.waitFor({ timeout: 30000 });
    mark("auto", "officer", "공무원 화면: 자동 확정 기록과 감사 기록만 남습니다");
    await pause(3000);
    assert((await row.innerText()).includes("자동 확정"), "자동 확정 표시가 없습니다");
    await tap(k, "#finish");
    return { ...r };
  },
  async A2(k, o) {
    const s = scenario("S1-clean");
    const r = await citizen(k, "auto", s, "full", { opening: "지원금 정산: 영수증철과 정산표" });
    assert(r.state === "처리가 끝났습니다", `자동 확정이 아닙니다: ${r.state}`);
    mark("auto", "kiosk", "영수증과 카드 원장이 맞아 정산이 승인됩니다");
    await pause(2200);
    await tap(k, "#finish");
    return r;
  },
  async A3(k, o) {
    const s = scenario("P1-clean");
    const r = await citizen(k, "auto", s, "id_only", { opening: "영업신고: 서류 없이 신분증만" });
    assert(r.state === "처리가 끝났습니다", `자동 확정이 아닙니다: ${r.state}`);
    mark("auto", "kiosk", "건축물대장과 필증은 대장에서 확인해 수리됩니다");
    await pause(2200);
    await tap(k, "#finish");
    return r;
  },
  async A4(k, o) {
    const s = scenario("C1-clean");
    const r = await citizen(k, "auto", s, "full", { opening: "인감증명 대리 발급: 위임장과 양쪽 신분증" });
    assert(r.state === "처리가 끝났습니다", `자동 확정이 아닙니다: ${r.state}`);
    mark("auto", "kiosk", "위임장 기간과 서명을 확인해 발급이 승인됩니다");
    await pause(2200);
    await tap(k, "#finish");
    return r;
  },
  // ---- 공무원 개입 ----
  async O1(k, o) {
    const s = scenario("W2-mother-gap");
    const r = await citizen(k, "officer", s, "full", { opening: "청년월세 신청: 이체액이 계약 월세보다 적은 건" });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    mark("officer", "officer", "불리한 후보는 자동으로 끝나지 않습니다. 공무원이 사유부터 봅니다");
    await openCase(o, r.receipt);
    const body = await text(o);
    assert(body.includes("위반 의심"), "위반 후보가 보이지 않습니다");
    assert(/\*|○/.test(body), "관계가 가려져 있지 않습니다");
    mark("officer", "officer", "관계망 연결은 이름을 가린 채 보입니다");
    await pause(2500);
    await tap(o, "#unmask");
    await waitText(o, "감사 기록에 남았습니다", 30000);
    mark("officer", "officer", "실명 열람도 감사 기록에 남습니다");
    await pause(2000);
    mark("officer", "officer", "소명 없이 승인하면 규칙이 막습니다");
    await choose(o, "#risk", "approve_evidence");
    await tap(o, "#approve");
    await sleep(500);
    await waitText(o, "입증", 30000).catch(() => {});
    await pause(2200);
    assert(await o.locator(".intake-detail .error").count(), "입증 없는 승인이 막히지 않았습니다");
    await clearChecks(o);
    mark("officer", "officer", "공무원이 반려를 확정합니다", { wait: true });
    await tap(o, "#primary");
    await finishKiosk(k, "officer", "민원인 화면에 이의신청 안내가 함께 나옵니다");
    return r;
  },
  async O2(k, o) {
    const s = scenario("W1-clean");
    const r = await citizen(k, "officer", s, "none", { opening: "서류를 하나도 못 가져온 민원인" });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    mark("officer", "officer", "신분증이 없어 본인 확인만 공무원이 합니다");
    await openCase(o, r.receipt);
    await clearChecks(o);
    await pause(800);
    mark("officer", "officer", "대장에서 서류를 확인했으므로 확인할 일은 하나뿐입니다", { wait: true });
    await tap(o, "#primary");
    await finishKiosk(k, "officer", "민원인은 대기 화면에서 결과를 받습니다");
    return r;
  },
  async O3(k, o) {
    const s = scenario("S3-short-receipts");
    const r = await citizen(k, "officer", s, "full", { opening: "지원금 정산: 영수증 합계가 모자라는 건" });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    mark("officer", "officer", "신청액과 인정 합계의 차이를 공무원이 봅니다");
    await openCase(o, r.receipt);
    await pause(1500);
    await clearChecks(o);
    mark("officer", "officer", "인정 금액으로 감액 정산합니다", { wait: true });
    await tap(o, "#primary");
    await finishKiosk(k, "officer", "감액된 금액으로 정산이 끝납니다");
    return r;
  },
  async O4(k, o) {
    const s = scenario("P2-illegal-building");
    const r = await citizen(k, "officer", s, "id_only", { opening: "영업신고: 위반건축물인 건" });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    await openCase(o, r.receipt);
    mark("officer", "officer", "위반건축물 표기가 사유로 올라옵니다", { wait: true });
    await pause(1500);
    await choose(o, "#risk", "approve_evidence");
    await tap(o, "#approve");
    await sleep(800);
    assert(await o.locator(".intake-detail .error").count(), "입증 없는 승인이 막히지 않았습니다");
    await pause(1500);
    await clearChecks(o);
    await tap(o, "#primary");
    await finishKiosk(k, "officer", "반려 안내가 나옵니다");
    return r;
  },
  async O5(k, o) {
    const s = scenario("C3-unsigned-poa");
    const r = await citizen(k, "officer", s, "full", { opening: "대리 발급: 위임자 서명이 없는 위임장" });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    await openCase(o, r.receipt);
    const body = await text(o);
    assert(body.includes("서명"), "서명 사유가 보이지 않습니다");
    mark("officer", "officer", "서명이 없는 위임장은 서류로 치지 않아 보완을 요구합니다", { wait: true });
    await pause(2000);
    await tap(o, "#supplement");
    await finishKiosk(k, "officer", "보완 기한이 안내됩니다");
    return r;
  },
  async O6(k, o) {
    const s = scenario("G1-noise");
    const r = await citizen(k, "officer", s, "full", { opening: "고충민원: 손으로 쓴 진정서" });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    mark("officer", "officer", "요약과 담당 부서 후보를 원문과 함께 봅니다");
    await openCase(o, r.receipt);
    await pause(1800);
    await tap(o, "#check-body");
    await clearChecks(o);
    const dep = await o.locator("#department").inputValue();
    if (!dep) await choose(o, "#department", s.expect.department);
    mark("officer", "officer", "담당 부서로 이첩합니다", { wait: true });
    await tap(o, "#primary");
    await finishKiosk(k, "officer", "이첩 안내가 민원인에게 갑니다");
    return { ...r, department: dep };
  },
  async O7(k, o) {
    const s = scenario("W7-no-registry");
    const r = await citizen(k, "officer", s, "full", { opening: "대장에 계약이 없는 민원인 (서류는 있음)" });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    await openCase(o, r.receipt);
    await clearChecks(o);
    await tap(o, "#primary");
    await finishKiosk(k, "officer", "서류 원본을 확인해 처리합니다");
    return r;
  },
  async O8(k, o) {
    const s = scenario("W1-clean");
    const r = await citizen(k, "officer", s, "none", { opening: "약한 서명은 다시 받습니다", weakSign: true });
    assert(r.state === "접수되었습니다", `대기가 아닙니다: ${r.state}`);
    await openCase(o, r.receipt);
    await clearChecks(o);
    await tap(o, "#primary");
    await finishKiosk(k, "officer", "다시 서명한 뒤 정상 처리됩니다");
    return r;
  },
};

(async () => {
  const browser = await connect();
  const kiosk = await findPage(browser, "스마트 민원 창구");
  KIOSK = kiosk;
  const officer = await findPage(browser, "심사 콘솔");
  await kiosk.bringToFront().catch(() => {});
  await endWaiting(kiosk);
  await waitText(kiosk, "스캔 시작", 600000);
  await pointer(kiosk);
  await pointer(officer);

  let stopRec = async () => {};
  if (REC) stopRec = await startRecording(browser, kiosk, officer);

  await type(officer, "#officer-id", "A-1024");
  if (!(await officer.locator("#auto-off").count())) {
    mark("setup", "officer", "자동화 모드 켜기: 승인자와 법령 근거가 있어야 켜집니다");
    await tap(officer, "#auto-open");
    await type(officer, "#auto-basis", "합성 시연용 근거 조항");
    await tap(officer, "#auto-on");
    await waitText(officer, "자동화 모드");
    await pause(1500);
  }

  const order = ONLY.length ? ONLY : Object.keys(FLOWS);
  for (const name of order) {
    const started = Date.now();
    actions = { kiosk: 0, officer: 0 };
    currentFlow = name;
    try {
      const r = await FLOWS[name](kiosk, officer);
      results.push({ name, ok: true, seconds: (Date.now() - started) / 1000, actions, ...r });
      console.log(`PASS ${name} ${((Date.now() - started) / 1000).toFixed(1)} s ${JSON.stringify(r.stats ?? {})} 동작 ${JSON.stringify(actions)}`);
    } catch (e) {
      results.push({ name, ok: false, seconds: (Date.now() - started) / 1000, error: String(e).slice(0, 600) });
      console.log(`FAIL ${name} ${String(e).slice(0, 400)}`);
      await endWaiting(kiosk);
    }
    await sleep(400);
  }

  const status = await officer.evaluate(() => document.querySelector("footer")?.innerText ?? "");
  console.log(status);
  await stopRec();
  fs.writeFileSync(OUT, JSON.stringify({ footer: status, t0_epoch_ms: t0, results, timeline }, null, 1));
  const failed = results.filter((r) => !r.ok).length;
  console.log(`결과 ${results.length - failed}/${results.length}`);
  await browser.close().catch(() => {});
  process.exit(failed ? 1 : 0);
})().catch((e) => {
  console.error(e);
  process.exit(2);
});

/** 두 창을 CDP 화면 전송으로 받습니다. 프레임마다 시각을 적어 편집기가 맞춥니다. */
async function startRecording(browser, kiosk, officer) {
  fs.mkdirSync(`${REC}/kiosk`, { recursive: true });
  fs.mkdirSync(`${REC}/officer`, { recursive: true });
  const logs = { kiosk: [], officer: [] };
  const sessions = [];
  for (const [name, page] of [["kiosk", kiosk], ["officer", officer]]) {
    const cdp = await page.context().newCDPSession(page);
    let n = 0;
    cdp.on("Page.screencastFrame", async (f) => {
      const file = `${String(++n).padStart(6, "0")}.jpg`;
      // 받은 시각이 아니라 화면이 그려진 시각을 씁니다.
      const shot = f.metadata?.timestamp ? f.metadata.timestamp * 1000 : Date.now();
      logs[name].push({ t: (shot - t0) / 1000, file });
      await cdp.send("Page.screencastFrameAck", { sessionId: f.sessionId }).catch(() => {});
      fs.writeFileSync(`${REC}/${name}/${file}`, Buffer.from(f.data, "base64"));
    });
    await cdp.send("Page.startScreencast", { format: "jpeg", quality: 80, everyNthFrame: 1 });
    sessions.push(cdp);
  }
  return async () => {
    for (const cdp of sessions) await cdp.send("Page.stopScreencast").catch(() => {});
    fs.writeFileSync(`${REC}/frames.json`, JSON.stringify(logs));
  };
}
