"""두 창 녹화를 교차 편집합니다. 모델이 쓰이는 순간에는 그 모델이 무엇을 했는지 장면을 끼워 넣습니다.

    FFMPEG=<ffmpeg 경로> CASES=out/cases SKIP=O4,O7,O8 python edit_video.py <녹화 폴더> <l4 결과.json> <trace.jsonl> <출력.mp4> [끝 화면.json]

녹화와 시각표는 `l4.js --record` 가 만듭니다. 모델 기록은 앱을 `ZOROGRAVITY_TRACE`, `ZOROGRAVITY_TRACE_TEXT=1` 로 띄워서 얻습니다.
화면의 숫자와 글은 모두 그 실행의 기록에서 옵니다. 끝 화면 json 은 {"lines": ["..."]} 입니다.
`STILLS=<폴더>` 를 주면 영상 대신 장면마다 한 장씩 저장합니다. 한글 글꼴은 Windows 의 맑은 고딕을 씁니다.
"""
import bisect
import json
import os
import re
import statistics
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
FPS = 30
TOP, BOTTOM = 64, 104
FADE = 0.4
MAX_PLAIN = 6.0  # 이보다 짧은 장면은 실제 속도
MAX_SHOWN = 5.0  # 긴 장면을 줄인 길이
MIN_SHOWN = 2.6  # 자막을 읽을 시간. 짧은 장면은 마지막 프레임에서 멈춰 둡니다
SIDE_W = 372
AREA = (16, TOP + 12, 16 + W - SIDE_W - 48, H - BOTTOM - 12)
RELABEL = {"소명 없이 승인하면 규칙이 막습니다": "입증 없이 승인하면 규칙이 막습니다"}

BG = (18, 22, 28)
PANEL = (28, 33, 41)
PANEL2 = (36, 42, 52)
TEXT = (240, 243, 247)
MUTED = (160, 170, 182)
DARK = (15, 18, 22)
GREEN = (72, 187, 120)
BLUE = (88, 166, 255)
ORANGE = (240, 160, 70)
RED = (240, 96, 96)
TEAL = (64, 196, 180)
PURPLE = (199, 146, 234)

# 오른쪽 열의 모델 표: 기록의 (모델, 단계) → 줄
ROWS = [
    ("siglip2", "SigLIP2", "어디를 읽을지", BLUE),
    ("gocr_v", "gocr 비전", "조각 → 글자 특징", TEAL),
    ("gocr_l", "gocr 언어", "특징 → 글자", GREEN),
    ("zoro_j", "Zoro-1 판단", "사실 → 확률", ORANGE),
    ("zoro_w", "Zoro-1 글쓰기", "사실 → 초안", PURPLE),
]
STEP_KEY = {("siglip2", "패치 특징"): "siglip2", ("gocr", "비전"): "gocr_v", ("gocr", "언어"): "gocr_l", ("zoro", "판단"): "zoro_j", ("zoro", "글쓰기"): "zoro_w"}
ROW_OF = {r[0]: r for r in ROWS}
MODEL_NAME = {"siglip2": "SigLIP2", "gocr": "gocr", "zoro": "Zoro-1"}
KIND_COLOR = {"제목": BLUE, "입력란": GREEN, "표": ORANGE, "서명": RED}
OPTION_LABEL = {
    "CAT_01_WELFARE": "복지", "CAT_02_SUBSIDY": "지원금", "CAT_03_PERMIT": "허가", "CAT_04_CERT": "증명", "CAT_05_GRIEVANCE": "고충",
    "G3_ELIGIBLE_NORMAL": "요건 충족", "G3_ELIGIBLE_HOMONYM_CLEAR": "동명이인 확인", "G3_VIOLATION_HIGH_RISK": "위반 (고위험)",
    "G3_HOLD_SIMPLE_RECEIPT_MISSING": "증빙 부족", "G3_VIOLATION_RELATED_VENDOR_SPLIT": "특수관계 · 분할 결제",
}
# 흐름별로 끼워 넣을 모델 장면
FLOW_CUTS = {
    "A1": {"read": True, "decide": True, "write": True, "gantt": True},
    "A2": {"decide": True},
    "A3": {"decide": True},
    "A4": {"sign": "위임장", "decide": True},
    "O1": {"decide": True},
    "O3": {"decide": True},
    "O5": {"sign": "위임장"},
    "O6": {"language": True, "write": True},
}
FLOW_SCENARIO = {"A1": "W1-clean", "A2": "S1-clean", "A3": "P1-clean", "A4": "C1-clean", "O1": "W2-mother-gap", "O3": "S3-short-receipts"}
FLOW_NOTE = {
    "O2": "서류가 없으니 SigLIP2 와 gocr 는 올라오지 않습니다. Zoro-1 만 씁니다",
    "A3": "신분증 한 장만 읽습니다. 나머지 사실은 합성 대장에서 옵니다",
}

REG = "C:/Windows/Fonts/malgun.ttf"
BOLD = "C:/Windows/Fonts/malgunbd.ttf"
FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(BOLD if bold else REG, size)
    return FONTS[key]


def centered(d, cx, y, text, f, fill):
    d.text((cx - d.textlength(text, font=f) / 2, y), text, font=f, fill=fill)


def wrap_px(d, text, f, width):
    """픽셀 폭에 맞춰 글을 나눕니다. 공백이 없는 긴 글은 글자 단위로 끊습니다."""
    lines, now = [], ""
    for word in text.split(" "):
        probe = f"{now} {word}".strip()
        if d.textlength(probe, font=f) <= width:
            now = probe
            continue
        if now:
            lines.append(now)
        now = word
        while d.textlength(now, font=f) > width:
            cut = len(now) - 1
            while cut > 1 and d.textlength(now[:cut], font=f) > width:
                cut -= 1
            lines.append(now[:cut])
            now = now[cut:]
    if now:
        lines.append(now)
    return lines


def block(d, x, y, text, f, fill, width, gap=8):
    for line in wrap_px(d, text, f, width):
        d.text((x, y), line, font=f, fill=fill)
        y += f.size + gap
    return y


def bullets(d, x, y, items, width, size=25, fill=TEXT):
    f = font(size)
    for item in items:
        d.ellipse((x, y + 12, x + 8, y + 20), fill=MUTED)
        y = block(d, x + 22, y, item, f, fill, width - 22) + 8
    return y


def secs(ms):
    return f"{ms / 1000:.1f}초"


class Screen:
    """한쪽 창의 녹화. 시각을 주면 그때의 프레임을 줍니다."""

    def __init__(self, folder, name, frames):
        self.dir = os.path.join(folder, name)
        self.times = [f["t"] for f in frames]
        self.files = [f["file"] for f in frames]
        self.cache = {}

    def at(self, t):
        i = max(0, bisect.bisect_right(self.times, t) - 1)
        if i not in self.cache:
            if len(self.cache) > 12:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = Image.open(os.path.join(self.dir, self.files[i])).convert("RGB")
        return self.cache[i]


class Trace:
    """앱이 남긴 모델 사용 기록. 시각은 녹화 시작 기준 초입니다."""

    def __init__(self, path, t0_ms):
        self.events = []
        with open(path, encoding="utf8") as f:
            for line in f:
                e = json.loads(line)
                e["end"] = (e["t"] - t0_ms) / 1000
                e["start"] = e["end"] - e["ms"] / 1000
                e["key"] = STEP_KEY.get((e["model"], e["step"]))
                self.events.append(e)
        self.events.sort(key=lambda e: e["end"])
        self.work = [e for e in self.events if e["key"]]
        self.moves = [e for e in self.events if e["step"] in ("올림", "내림")]

    def active(self, t):
        for e in self.work:
            if e["start"] <= t <= e["end"]:
                return e
        return None

    def resident(self, t):
        """시각 t 에 GPU 에 올라와 있는 가중치: {(모델, 묶음): 바이트}"""
        now = {}
        for e in self.moves:
            if e["end"] > t:
                break
            key = (e["model"], e["detail"]["group"])
            if e["step"] == "올림":
                now[key] = e["detail"]["bytes"]
            else:
                now.pop(key, None)
        return now

    def between(self, a, b, key=None, step=None):
        return [e for e in self.events if a <= e["end"] <= b and (key is None or e["key"] == key) and (step is None or e["step"] == step)]


def fit(img, box_w, box_h):
    r = min(box_w / img.width, box_h / img.height)
    return img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.BILINEAR)


def plain_text(text):
    """표 태그를 지워 사람이 읽는 글로 바꿉니다."""
    t = re.sub(r"</td>\s*<td[^>]*>", " | ", text)
    t = re.sub(r"</tr>\s*<tr>", " / ", t)
    t = re.sub(r"<[^>]+>", "", t).replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip(" |/")


class Papers:
    """서류 묶음: 페이지 이미지, 계획된 조각, 읽은 글자. 기록의 순서는 스캐너 폴더 순서입니다."""

    def __init__(self, cases_root, scenario, names, plan_event, result_event):
        self.pages = []
        crops_of_text = iter(result_event["detail"]["texts"]) if result_event else iter(())
        for name, page in zip(names, plan_event["detail"]["pages"]):
            img = Image.open(os.path.join(cases_root, scenario["id"], f"{name}.jpg")).convert("RGB")
            # 조각 좌표는 곧게 편 페이지 기준입니다. `deskew` 와 같은 방향으로 돌립니다.
            img = img.rotate(page["skew"], resample=Image.BILINEAR, fillcolor=(255, 255, 255))
            crops = []
            for c in page["crops"]:
                item = {"kind": c["kind"], "bbox": c["bbox"], "text": "", "conf": None}
                if c["kind"] != "서명":
                    read = next(crops_of_text, None)
                    if read:
                        item["text"], item["conf"] = plain_text(read["text"]), read["conf"]
                crops.append(item)
            self.pages.append({"name": name, "img": img, "crops": crops})

    def tiles(self):
        return [(p, c) for p in self.pages for c in p["crops"] if c["kind"] != "서명"]

    @staticmethod
    def cut(page, crop):
        return page["img"].crop(tuple(crop["bbox"]))


def page_with_boxes(page, height, show=1.0, grid=0.0):
    """페이지를 줄여 조각 상자를 그립니다. `grid` 는 SigLIP2 패치 격자의 진하기입니다."""
    img = page["img"]
    r = height / img.height
    out = img.resize((int(img.width * r), height), Image.BILINEAR).convert("RGBA")
    layer = Image.new("RGBA", out.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if grid > 0:
        # 페이지당 최대 256 패치. A4 는 19 × 13 격자입니다.
        cols, rows = 13, 19
        alpha = int(90 * grid)
        for i in range(1, cols):
            x = out.width * i / cols
            d.line((x, 0, x, out.height), fill=(60, 120, 255, alpha), width=1)
        for i in range(1, rows):
            y = out.height * i / rows
            d.line((0, y, out.width, y), fill=(60, 120, 255, alpha), width=1)
    n = int(len(page["crops"]) * show + 0.999) if show > 0 else 0
    # 그림층에서는 나중에 그린 상자가 앞 상자를 덮으므로 서명을 맨 뒤에 그립니다.
    for c in sorted(page["crops"][:n], key=lambda c: c["kind"] == "서명"):
        color = KIND_COLOR.get(c["kind"], MUTED)
        box = [v * r for v in c["bbox"]]
        d.rectangle(box, fill=(*color, 46), outline=(*color, 255), width=3 if c["kind"] == "서명" else 2)
    return Image.alpha_composite(out, layer).convert("RGB")


def model_head(d, name, role, color):
    x, y = AREA[0] + 20, AREA[1] + 14
    tw = d.textlength(name, font=font(28, True)) + 30
    d.rounded_rectangle((x, y, x + tw, y + 48), 10, fill=color)
    d.text((x + 15, y + 5), name, font=font(28, True), fill=DARK)
    d.text((x + tw + 18, y + 7), role, font=font(27), fill=TEXT)


def legend(d, x, y):
    for kind, color in KIND_COLOR.items():
        d.rectangle((x, y + 6, x + 22, y + 26), outline=color, width=3)
        d.text((x + 30, y), kind, font=font(22), fill=TEXT)
        x += 46 + d.textlength(kind, font=font(22)) + 18
    return y + 40


def draw_siglip2(canvas, d, s, p):
    data = s["papers_data"]
    model_head(d, "SigLIP2", "서류 이미지에서 어디를 읽을지 고릅니다", BLUE)
    pages = data.pages
    # 페이지를 차례로 크게 보입니다.
    at = min(len(pages) - 1, int(p * len(pages)))
    local = p * len(pages) - at
    page = pages[at]
    big = page_with_boxes(page, 690, show=max(0.0, (local - 0.25) / 0.6), grid=max(0.0, 1 - local * 1.6))
    x0, y0 = AREA[0] + 24, AREA[1] + 78
    canvas.paste(big, (x0, y0))
    d.rectangle((x0 - 1, y0 - 1, x0 + big.width, y0 + big.height), outline=(90, 100, 112))
    tx = x0 + big.width + 36
    width = AREA[2] - tx - 20
    # 다른 페이지 작은 그림
    strip_h = 150
    x = tx
    for i, q in enumerate(pages):
        th = q.setdefault("thumb", page_with_boxes(q, strip_h))
        canvas.paste(th, (x, y0))
        d.rectangle((x - 2, y0 - 2, x + th.width + 1, y0 + th.height + 1), outline=BLUE if i == at else (90, 100, 112), width=3 if i == at else 1)
        x += th.width + 12
    y = y0 + strip_h + 18
    d.text((tx, y), f"{at + 1} / {len(pages)}  {page['name']}", font=font(24, True), fill=TEXT)
    y = legend(d, tx, y + 40) + 8
    counts = {}
    for q in pages:
        for c in q["crops"]:
            counts[c["kind"]] = counts.get(c["kind"], 0) + 1
    e = s["event"]
    y = bullets(d, tx, y, [
        "먼저 규칙이 기울기를 잡고 잉크로 조각을 나눕니다 (모델 아님)",
        "SigLIP2 비전 타워만 올려 페이지를 패치로 나눠 특징으로 바꿉니다",
        "시각 앵커로 칸마다 제목 · 입력란 · 표 · 안내문 · 서명을 겨룹니다",
        "고정 안내문은 버리고, 서명 · 날인은 글자로 읽지 않고 위치만 남깁니다",
    ], width)
    d.text((tx, y + 6), f"실측  {secs(e['ms'])} · {e['detail']['pages']}장 · 패치 {e['detail']['patches']:,}개", font=font(23, True), fill=BLUE)
    d.text((tx, y + 40), "조각 " + " · ".join(f"{k} {v}" for k, v in counts.items()), font=font(23), fill=MUTED)


def draw_sign(canvas, d, s, p):
    data = s["papers_data"]
    page = next(q for q in data.pages if q["name"] == s["sign_page"])
    model_head(d, "SigLIP2", "서명 · 날인은 글자로 읽지 않고 위치만 찾습니다", BLUE)
    big = page_with_boxes(page, 690, show=min(1.0, p * 1.5))
    x0, y0 = AREA[0] + 24, AREA[1] + 78
    canvas.paste(big, (x0, y0))
    d.rectangle((x0 - 1, y0 - 1, x0 + big.width, y0 + big.height), outline=(90, 100, 112))
    tx = x0 + big.width + 36
    width = AREA[2] - tx - 20
    d.text((tx, y0), page["name"], font=font(30, True), fill=TEXT)
    y = legend(d, tx, y0 + 52) + 12
    signs = [c for c in page["crops"] if c["kind"] == "서명"]
    y = bullets(d, tx, y, [
        "서명 · 날인 자리는 SigLIP2 의 시각 앵커와 잉크 분석으로 찾습니다",
        "gocr 로 읽지 않습니다. 있고 없음만 사건 사실이 됩니다",
    ], width)
    if signs:
        y = block(d, tx, y + 14, f"이 위임장: 서명 조각 {len(signs)}개 있음 → 위임자 서명 확인", font(28, True), GREEN, width)
    else:
        y = block(d, tx, y + 14, "이 위임장: 서명 조각 없음 → 서류로 치지 않고 공무원에게 보냅니다", font(28, True), RED, width)


def draw_gocr_vision(canvas, d, s, p):
    data = s["papers_data"]
    e = s["event"]
    model_head(d, "gocr 비전", "고른 조각을 글자 특징으로 바꿉니다", TEAL)
    tiles = data.tiles()
    shown = tiles[:24]
    cols, tile_w, tile_h = 3, 300, 76
    x0, y0 = AREA[0] + 24, AREA[1] + 82
    done = int(p * len(shown) + 0.5)
    cache = s.setdefault("tile_cache", {})
    for i, (page, crop) in enumerate(shown):
        x = x0 + (i % cols) * (tile_w + 12)
        y = y0 + (i // cols) * (tile_h + 12)
        if i not in cache:
            cache[i] = fit(Papers.cut(page, crop), tile_w, tile_h)
        d.rectangle((x, y, x + tile_w, y + tile_h), fill=(250, 250, 250))
        canvas.paste(cache[i], (x + (tile_w - cache[i].width) // 2, y + (tile_h - cache[i].height) // 2))
        d.rectangle((x - 2, y - 2, x + tile_w + 2, y + tile_h + 2), outline=TEAL if i < done else (80, 90, 100), width=3 if i < done else 1)
    tx = x0 + cols * (tile_w + 12) + 26
    width = AREA[2] - tx - 20
    res = s.get("res_bytes", 0)
    y = bullets(d, tx, y0, [
        "SigLIP2 를 내린 뒤 gocr 비전 타워만 올립니다",
        "조각을 패치로 나눠 이미지 토큰으로 바꿉니다",
        "특징만 남기고 비전 타워를 내립니다. 언어 가중치는 아직 없습니다",
    ], width)
    r = s["result"]["detail"]
    d.text((tx, y + 10), f"실측  {secs(e['ms'])}", font=font(25, True), fill=TEAL)
    d.text((tx, y + 48), f"조각 {e['detail']['inputs']}개", font=font(23), fill=MUTED)
    d.text((tx, y + 82), f"이미지 토큰 {r['image_tokens']:,}개", font=font(23), fill=MUTED)
    if res:
        d.text((tx, y + 116), f"GPU {res / 1e9:.2f} GB 사용", font=font(23), fill=MUTED)
    if len(tiles) > len(shown):
        d.text((x0, AREA[3] - 40), f"화면에는 {len(shown)}개만 보입니다. 전체 {len(tiles)}개", font=font(20), fill=MUTED)


def draw_gocr_language(canvas, d, s, p):
    data = s["papers_data"]
    e = s["event"]
    r = s["result"]["detail"]
    model_head(d, "gocr 언어", "특징을 보고 글자를 씁니다", GREEN)
    pick = [t for t in data.tiles() if t[1]["text"] and (not s.get("only_page") or t[0]["name"] == s["only_page"])]
    if not s.get("only_page"):
        pick = pick[:: max(1, len(pick) // 12)][:12]
    rows = 6
    start = min(max(0, len(pick) - rows), int(p * max(1, len(pick) - rows + 1)))
    x0, y0 = AREA[0] + 24, AREA[1] + 82
    tile_w, tile_h = 400, 84
    typed = (p * len(pick)) - start
    cache = s.setdefault("tile_cache2", {})
    for j, (page, crop) in enumerate(pick[start:start + rows]):
        y = y0 + j * (tile_h + 14)
        key = id(crop)
        if key not in cache:
            cache[key] = fit(Papers.cut(page, crop), tile_w, tile_h)
        d.rectangle((x0, y, x0 + tile_w, y + tile_h), fill=(250, 250, 250))
        canvas.paste(cache[key], (x0 + (tile_w - cache[key].width) // 2, y + (tile_h - cache[key].height) // 2))
        text = crop["text"]
        if j >= typed:
            text = ""
        elif j + 1 > typed:
            text = text[: int(len(text) * (typed - j))]
        lines = wrap_px(d, text, font(24), 520)[:3]
        for k, line in enumerate(lines):
            d.text((x0 + tile_w + 22, y + 4 + k * 30), line, font=font(24), fill=TEXT)
    tx = x0 + tile_w + 22 + 540
    width = AREA[2] - tx - 16
    y = bullets(d, tx, y0, [
        "비전을 내리고 언어 블록과 출력층만 올립니다",
        "초안을 한꺼번에 내고 검증합니다",
        "표는 칸 구조를 지켜 읽습니다",
        "글자별 신뢰도를 함께 냅니다",
    ], width, size=23)
    d.text((tx, y + 10), f"실측  {secs(e['ms'])}", font=font(25, True), fill=GREEN)
    d.text((tx, y + 48), f"토큰 {r['tokens']:,}개", font=font(23), fill=MUTED)
    d.text((tx, y + 82), f"초안 {r['drafts']}번", font=font(23), fill=MUTED)
    d.text((x0, AREA[3] - 44), "모델은 글자만 읽습니다. 서류 종류와 항목은 뒤이어 규칙이 정하고 대장과 맞춥니다", font=font(23), fill=MUTED)


def bar(d, x, y, width, value, color, height=20):
    d.rounded_rectangle((x, y, x + width, y + height), 5, fill=PANEL2)
    if value > 0:
        d.rounded_rectangle((x, y, x + max(8, width * value), y + height), 5, fill=color)


def draw_decision(canvas, d, s, p):
    questions = s["questions"]
    model_head(d, "Zoro-1 판단", "글을 쓰지 않고 선택지를 가리킵니다", ORANGE)
    x0, y0 = AREA[0] + 24, AREA[1] + 82
    left = 560
    d.text((x0, y0), "모델에 주는 사실 (이름 · 번호 · 계좌 없음)", font=font(24, True), fill=TEXT)
    y = y0 + 44
    for key, value in questions[0]["detail"]["facts"]:
        d.rectangle((x0, y, x0 + left, y + 4), fill=PANEL2)
        y += 12
        d.text((x0, y), key, font=font(21), fill=MUTED)
        y = block(d, x0, y + 28, value, font(23), TEXT, left, gap=6) + 8
    y += 8
    y = bullets(d, x0, y, [
        "백본이 사실을 읽고, jev_head 가 선택지마다 점수를 매깁니다",
        "처분은 규칙이 정합니다. 모델은 같은 질문의 확률만 냅니다",
    ], left, size=23)
    tx = x0 + left + 40
    width = AREA[2] - tx - 20
    n = len(questions)
    upto = min(n, int(p * n) + 1)
    yy = y0
    for i, q in enumerate(questions[:upto]):
        detail = q["detail"]
        title = detail["task"].replace("임차 요건 심사 중 ", "").replace("보조금 집행 심사 중 ", "").replace("영업신고 수리 심사 중 ", "").replace("위임장 유효성 심사 중 ", "").replace(" 만 판단", "")
        d.text((tx, yy), title, font=font(24, True), fill=TEXT)
        d.text((tx + width - 90, yy + 2), f"{q['ms'] / 1000:.1f}초", font=font(20), fill=MUTED)
        yy += 36
        pairs = sorted(zip(detail["options"], detail["probabilities"]), key=lambda a: -a[1])[:2]
        for j, (code, prob) in enumerate(pairs):
            label = OPTION_LABEL.get(code, code)
            d.text((tx, yy - 3), label, font=font(21), fill=MUTED)
            bar(d, tx + 200, yy, width - 290, prob, ORANGE if j == 0 else (120, 130, 142))
            d.text((tx + width - 78, yy - 4), f"{prob:.3f}", font=font(21), fill=TEXT)
            yy += 32
        yy += 12
    if upto == n and s.get("verdict_note"):
        block(d, tx, max(yy + 6, AREA[3] - 92), s["verdict_note"], font(26, True), s.get("verdict_color", ORANGE), width)


WRITE_KIND = (("심사 의견", "심사 의견"), ("민원 요약", "고충 요약"), ("표준 코드", "소명 분류"))


def write_kind(system):
    for key, name in WRITE_KIND:
        if key in system:
            return name
    return "초안"


def draw_writer(canvas, d, s, p):
    e = s["event"]
    detail = e["detail"]
    kind = write_kind(detail["system"])
    model_head(d, "Zoro-1 글쓰기", f"{kind} 초안을 씁니다", PURPLE)
    x0, y0 = AREA[0] + 24, AREA[1] + 92
    width = AREA[2] - x0 - 24
    d.text((x0, y0), "지시문", font=font(22, True), fill=MUTED)
    y = block(d, x0, y0 + 34, detail["system"], font(23), MUTED, width, gap=6) + 18
    d.text((x0, y), "모델이 쓴 글", font=font(22, True), fill=MUTED)
    y += 38
    text = detail.get("text", "")
    shown = text[: int(len(text) * min(1.0, p * 1.5))]
    box_y = y
    d.rounded_rectangle((x0, box_y, x0 + width, box_y + 220), 10, fill=PANEL2)
    yy = box_y + 18
    for para in shown.split("\n"):
        yy = block(d, x0 + 22, yy, para, font(30), TEXT, width - 44, gap=10)
    y = box_y + 244
    y = bullets(d, x0, y, [
        "판단과 같은 몸통에 임베딩 표를 더 올려 토큰을 하나씩 만듭니다 (듀얼 헤드)",
        "글의 숫자와 날짜는 원문과 대조합니다. 초안은 처분을 바꾸지 않습니다",
    ], width, size=24)
    per = e["ms"] / max(1, detail["tokens"])
    d.text((x0, y + 8), f"실측  {secs(e['ms'])} · 토큰 {detail['tokens']}개 · 토큰당 {per:.0f} ms", font=font(25, True), fill=PURPLE)


def draw_gantt(canvas, d, s, p):
    tr = s["trace"]
    a, b = s["span"]
    x0, y0 = AREA[0] + 24, AREA[1] + 24
    d.text((x0, y0), "이 건에서 모델이 쓰인 순서", font=font(34, True), fill=TEXT)
    d.text((x0, y0 + 50), f"녹화 실행 {b - a:.0f}초 (설명용 대기 포함). GPU 에는 한 번에 한 모델만 있습니다", font=font(25), fill=MUTED)
    left = 230
    chart_x, chart_w = x0 + left, AREA[2] - x0 - left - 30
    y = y0 + 124
    row_h = 94
    scale = chart_w / (b - a)
    totals = {}
    for key, name, role, color in ROWS:
        d.text((x0, y + 6), name, font=font(25, True), fill=color)
        d.text((x0, y + 38), role, font=font(19), fill=MUTED)
        d.rectangle((chart_x, y + 4, chart_x + chart_w, y + row_h - 14), fill=PANEL)
        for e in tr.between(a, b, key=key):
            x1 = chart_x + (max(e["start"], a) - a) * scale
            x2 = chart_x + (e["end"] - a) * scale
            d.rectangle((x1, y + 8, max(x1 + 3, x2), y + row_h - 18), fill=color)
            totals[key] = totals.get(key, 0) + e["ms"]
        if key in totals:
            d.text((chart_x + chart_w + 6 - 90, y + 16), secs(totals[key]), font=font(21), fill=TEXT)
        y += row_h
    # 가중치 전환
    d.text((x0, y + 6), "가중치 전환", font=font(25, True), fill=TEXT)
    d.text((x0, y + 38), "올림 · 내림", font=font(19), fill=MUTED)
    d.rectangle((chart_x, y + 4, chart_x + chart_w, y + row_h - 14), fill=PANEL)
    moves = tr.between(a, b, step="올림") + tr.between(a, b, step="내림")
    for e in moves:
        x1 = chart_x + (e["end"] - a) * scale
        d.line((x1, y + 8, x1, y + row_h - 18), fill=RED if e["step"] == "내림" else TEXT, width=3)
    y += row_h + 12
    for t in range(0, int(b - a) + 1, 10):
        x1 = chart_x + t * scale
        d.text((x1 - 10, y - 6), f"{t}", font=font(19), fill=MUTED)
    y += 36
    peak = max((sum(tr.resident(e["end"]).values()) for e in tr.between(a, b)), default=0)
    d.text((x0, y), f"전환 {len(moves)}번, 합계 {sum(e['ms'] for e in moves) / 1000:.1f}초 · 가장 많이 쓸 때 GPU {peak / 1e9:.1f} GB (RX 6600 8 GB)", font=font(25), fill=TEXT)


DRAW = {"siglip2": draw_siglip2, "sign": draw_sign, "gocr_vision": draw_gocr_vision, "gocr_language": draw_gocr_language, "decision": draw_decision, "writer": draw_writer, "gantt": draw_gantt}


def side_panel(canvas, d, scene, t, screens, trace, other_name):
    sx = W - SIDE_W - 16
    other = screens[other_name].at(t)
    pip = fit(other, SIDE_W - 12, 200)
    d.rectangle((sx, TOP + 12, W - 16, TOP + 12 + 32 + pip.height + 8), fill=PANEL)
    d.text((sx + 8, TOP + 16), "공무원 화면" if other_name == "officer" else "민원인 화면", font=font(17), fill=MUTED)
    canvas.paste(pip, (sx + 6, TOP + 12 + 32))
    y = TOP + 12 + 32 + pip.height + 20
    for line in wrap_px(d, scene.get("title", ""), font(23, True), SIDE_W - 16):
        d.text((sx + 8, y), line, font=font(23, True), fill=TEXT)
        y += 33
    y += 4
    if scene.get("flow_start") is not None:
        d.text((sx + 8, y), f"실제 경과 {max(0, t - scene['flow_start']):.0f}초", font=font(21), fill=MUTED)
        if scene.get("speed", 1) > 1.2:
            d.text((sx + 190, y + 2), f"▶▶ {scene['speed']:.0f}배속", font=font(19), fill=ORANGE)
        y += 34
    # 지금 쓰는 모델
    y = max(y + 6, 500)
    d.rectangle((sx, y, W - 16, y + 402), fill=PANEL)
    d.text((sx + 12, y + 8), "지금 쓰는 모델", font=font(20, True), fill=MUTED)
    y += 42
    now = trace.active(t) if scene.get("kind") != "gantt" else None
    for key, name, role, color in ROWS:
        on = now is not None and now["key"] == key
        if on:
            d.rounded_rectangle((sx + 8, y, W - 24, y + 52), 8, fill=color)
        d.text((sx + 20, y + 1), name, font=font(21, True), fill=DARK if on else (150, 158, 170))
        d.text((sx + 20, y + 29), role, font=font(15), fill=DARK if on else (120, 128, 140))
        y += 55
    # GPU 에 올라온 가중치
    res = trace.resident(t)
    total = sum(res.values())
    models = sorted({m for m, _ in res})
    d.text((sx + 12, y + 8), "GPU 에 올라온 가중치", font=font(18, True), fill=MUTED)
    y += 34
    name = " + ".join(MODEL_NAME.get(m, m) for m in models) if models else "비어 있음"
    d.text((sx + 12, y), f"{name}  {total / 1e9:.2f} GB", font=font(19), fill=TEXT)
    bar(d, sx + 12, y + 28, SIDE_W - 28, min(1.0, total / 8.43e9), BLUE, height=12)


def frame(scene, t, p, screens, trace):
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    kind = scene.get("kind", "screen")
    if kind == "screen":
        main = screens[scene["screen"]].at(t)
        area_w, area_h = AREA[2] - AREA[0], AREA[3] - AREA[1]
        shown = fit(main, area_w, area_h)
        x, y = AREA[0] + (area_w - shown.width) // 2, AREA[1] + (area_h - shown.height) // 2
        canvas.paste(shown, (x, y))
        thumbs = scene.get("thumbs")
        if thumbs:
            tx = x + 16
            for th in thumbs:
                canvas.paste(th, (tx, y + shown.height - th.height - 16))
                d.rectangle((tx, y + shown.height - th.height - 16, tx + th.width, y + shown.height - 16), outline=(120, 130, 140))
                tx += th.width + 10
        other_name = "officer" if scene["screen"] == "kiosk" else "kiosk"
        where = "민원인 화면" if scene["screen"] == "kiosk" else "공무원 화면"
    else:
        d.rounded_rectangle(AREA, 10, fill=PANEL)
        DRAW[kind](canvas, d, scene, p)
        other_name = "kiosk" if scene.get("screen", "kiosk") == "kiosk" else "officer"
        where = "모델 동작 (실제 기록)"
    side_panel(canvas, d, scene, t, screens, trace, other_name)

    d.rectangle((0, 0, W, TOP), fill=PANEL)
    tag, color = {"auto": ("자동화 절차", GREEN), "officer": ("공무원 개입", ORANGE), "setup": ("준비", BLUE)}[scene["track"]]
    d.rounded_rectangle((24, 14, 24 + 150, 50), 8, fill=color)
    centered(d, 24 + 75, 16, tag, font(20, True), DARK)
    d.text((192, 15), where, font=font(24, True), fill=TEXT)
    label = "합성 데이터 · 실제 모델(ROCm) · 실제 앱"
    d.text((W - 24 - d.textlength(label, font=font(18)), 20), label, font=font(18), fill=MUTED)

    d.rectangle((0, H - BOTTOM, W, H), fill=PANEL)
    note = scene.get("note")
    if note:
        d.text((40, H - BOTTOM + 8), scene["label"], font=font(32, True), fill=TEXT)
        d.text((40, H - BOTTOM + 58), note, font=font(23), fill=MUTED)
    else:
        d.text((40, H - BOTTOM + 26), scene["label"], font=font(34, True), fill=TEXT)
    return canvas


def card(lines):
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    total = sum(s + 26 for _, s, _ in lines)
    y = (H - total) // 2
    for text, size, color in lines:
        centered(d, W / 2, y, text, font(size, size >= 40), color)
        y += size + 26
    return canvas


def models_card():
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    centered(d, W / 2, 110, "모델이 쓰이는 자리", font(54, True), TEXT)
    centered(d, W / 2, 186, "처분은 규칙이 정합니다. 모델은 읽고, 확인하고, 초안을 씁니다. GPU 에는 한 번에 한 모델만 올립니다", font(27), MUTED)
    rows = [
        ("SigLIP2", BLUE, "어디를 읽을지 고릅니다", "제목 · 입력란 · 표 · 서명 조각을 찾고 고정 안내문을 버립니다"),
        ("gocr 비전 → 언어", TEAL, "조각의 글자를 읽습니다", "비전으로 특징을 만든 뒤 내리고, 언어 가중치로 글자를 씁니다"),
        ("Zoro-1 판단", ORANGE, "규칙의 판정을 기준별로 다시 묻습니다", "식별 정보 없는 사실만 보고 확률을 냅니다. 처분은 바꾸지 못합니다"),
        ("Zoro-1 글쓰기", PURPLE, "요약 · 소명 분류 · 심사 의견 초안을 씁니다", "숫자와 날짜를 원문과 대조합니다. 초안은 처분을 바꾸지 못합니다"),
    ]
    y = 290
    for name, color, what, how in rows:
        d.rounded_rectangle((180, y, 1740, y + 142), 12, fill=PANEL)
        d.rounded_rectangle((204, y + 46, 204 + 330, y + 96), 10, fill=color)
        centered(d, 204 + 165, y + 53, name, font(26, True), DARK)
        d.text((570, y + 28), what, font=font(31, True), fill=TEXT)
        d.text((570, y + 80), how, font=font(25), fill=MUTED)
        y += 164
    return canvas


def sub(base, a, b):
    """장면의 일부. 서류 그림은 처음 부분에만 보입니다."""
    real = max(0.1, b - a)
    speed = 1.0 if real <= MAX_PLAIN else real / MAX_SHOWN
    out = {**base, "t0": a, "t1": b, "real": real, "speed": speed, "shown": max(real / speed, MIN_SHOWN)}
    if a > base["t0"]:
        out.pop("thumbs", None)
    return out


def verdict_note(questions, rule):
    """규칙 후보와 모델 의견을 있는 그대로 적습니다."""
    crit = [q for q in questions if "만 판단" in q["detail"]["task"]]
    if crit:
        low = min(q["detail"]["probabilities"][0] for q in crit)
        if low >= 0.95:
            return f"기준별 최솟값 {low:.3f} ≥ 기준 0.95 → 확신도 조건 충족", GREEN
        return f"기준별 최솟값 {low:.3f} < 기준 0.95 → 자동 확정하지 않습니다", RED
    last = questions[-1]["detail"]
    prob, code = max(zip(last["probabilities"], last["options"]))
    name = OPTION_LABEL.get(code, code)
    rule_name = OPTION_LABEL.get(rule, rule)
    if code == rule:
        return f"규칙 후보 {rule_name}, 모델도 같은 쪽 ({prob:.3f}). 불리한 후보는 공무원이 확정합니다", ORANGE
    return f"규칙 후보 {rule_name}, 모델은 {name} 쪽 ({prob:.3f}). 규칙 후보가 유지되고 공무원이 봅니다", ORANGE


def build_scenes(marks, end, trace, skip, cases_root, scenarios):
    titles, starts, stops = {}, {}, {}
    for i, m in enumerate(marks):
        flow = m.get("flow")
        if flow and flow not in titles:
            titles[flow] = m["label"]
            starts[flow] = m["t"]
        if flow:
            stops[flow] = marks[i + 1]["t"] if i + 1 < len(marks) else end
    scenes = []
    seen = set()
    for i, m in enumerate(marks):
        flow = m.get("flow", "")
        if flow in skip:
            continue
        t1 = marks[i + 1]["t"] if i + 1 < len(marks) else end
        # 다음 동작이 시작되기 직전에 끊습니다. 장면 끝에 다음 화면이 비치지 않게 합니다.
        real = max(0.1, t1 - m["t"] - 0.35)
        t1 = m["t"] + real
        speed = 1.0 if real <= MAX_PLAIN else real / MAX_SHOWN
        label = RELABEL.get(m["label"], m["label"])
        key = (flow, label)
        if label.startswith("차액을 물어") and key in seen:
            label = "계약서와 맞지 않는 소명은 잠그고 다시 묻습니다"
        seen.add(key)
        base = {**m, "label": label, "t0": m["t"], "t1": t1, "real": real, "speed": speed, "shown": max(real / speed, MIN_SHOWN), "title": titles.get(flow, ""), "flow_start": starts.get(flow)}
        cfg = FLOW_CUTS.get(flow, {})
        cuts = []
        is_scan = bool(m.get("papers"))
        if is_scan:
            scenario = scenarios[m["papers"]]
            names = sorted(scenario["docs"].keys())
            if m["label"].find("1장") >= 0:
                names = ["신분증"]
            else:
                names = names + ["신분증"]
            base["names"] = names
            base["thumbs"] = [Image.open(os.path.join(cases_root, m["papers"], f"{n}.jpg")).convert("RGB") for n in names[:6]]
            base["thumbs"] = [t.resize((int(t.width * 210 / t.height), 210), Image.BILINEAR) for t in base["thumbs"]]
            base["note"] = FLOW_NOTE.get(flow) or "SigLIP2 → gocr 비전 → gocr 언어 순서로 모델을 바꿔 가며 읽습니다"
            plan = (trace.between(m["t"], t1 + 0.4, step="조각 계획") or [None])[0]
            result = (trace.between(m["t"], t1 + 0.4, step="결과") or [None])[0]
            if plan:
                data = Papers(cases_root, scenario, names, plan, result)
                ink = trace.between(m["t"], t1 + 0.4, step="기울기 보정 · 잉크 분석")[0]
                sig = trace.between(m["t"], t1 + 0.4, key="siglip2")[0]
                vis = trace.between(m["t"], t1 + 0.4, key="gocr_v")[0]
                lang = trace.between(m["t"], t1 + 0.4, key="gocr_l")[0]
                common = {"papers_data": data, "track": m["track"], "title": titles.get(flow, ""), "flow_start": starts.get(flow), "screen": "kiosk", "flow": flow}
                if cfg.get("read"):
                    cuts.append({**common, "kind": "siglip2", "t0": ink["start"], "t1": plan["end"], "shown": 10.0, "event": sig, "label": "SigLIP2: 어디를 읽을지 고릅니다", "note": "문서 모델 첫 단계입니다. 읽을 조각을 줄여야 gocr 시간이 줄어듭니다"})
                    cuts.append({**common, "kind": "gocr_vision", "t0": vis["start"], "t1": vis["end"], "shown": 6.5, "event": vis, "result": result, "res_bytes": sum(trace.resident(vis["end"]).values()), "label": "gocr 비전: 조각을 글자 특징으로 바꿉니다", "note": "SigLIP2 를 내린 뒤에 올립니다. 두 모델이 GPU 에 함께 있지 않습니다"})
                    cuts.append({**common, "kind": "gocr_language", "t0": lang["start"], "t1": lang["end"], "shown": 8.5, "event": lang, "result": result, "label": "gocr 언어: 특징을 보고 글자를 씁니다", "note": "비전을 내리고 언어 가중치만 올립니다"})
                if cfg.get("sign"):
                    cuts.append({**common, "kind": "sign", "t0": ink["start"], "t1": plan["end"], "shown": 6.5, "sign_page": cfg["sign"], "label": "SigLIP2: 서명 자리를 찾습니다", "note": "서명은 규칙이 보는 사실이 됩니다"})
                if cfg.get("language"):
                    cuts.append({**common, "kind": "gocr_language", "t0": lang["start"], "t1": lang["end"], "shown": 7.0, "event": lang, "result": result, "only_page": "진정서", "label": "gocr 언어: 손글씨 진정서를 읽습니다", "note": "손글씨도 같은 경로로 읽습니다. 오독은 민원인이 화면에서 고칩니다"})
        decisions = [e for e in trace.between(m["t"], t1 + 0.4, key="zoro_j")]
        writers = trace.between(m["t"], t1 + 0.4, key="zoro_w")
        if decisions and not is_scan and "note" not in base:
            base["note"] = "규칙이 판정하고, Zoro-1 은 같은 질문의 확률을 냅니다" + (". 이어서 심사 의견 초안을 씁니다" if writers else "")
        common = {"track": m["track"], "title": titles.get(flow, ""), "flow_start": starts.get(flow), "screen": "kiosk", "flow": flow}
        if flow in FLOW_NOTE and not is_scan and "note" not in base:
            base["note"] = FLOW_NOTE[flow]
        if cfg.get("decide") and decisions and not cfg.get("_decided"):
            cfg["_decided"] = True
            rule = scenarios[FLOW_SCENARIO[flow]]["expect"]["verdict"]
            note, color = verdict_note(decisions, rule)
            shown = 3.0 + 1.5 * len(decisions)
            split = any("만 판단" in e["detail"]["task"] for e in decisions)
            what = "규칙의 판정을 기준별로 다시 묻습니다" if split else "규칙의 후보에 의견만 냅니다"
            hint = "유리한 판정만 기준별로 나눠 묻습니다. 모델이 불리한 쪽을 고르면 자동 확정이 멈춥니다" if split else "불리하거나 절차인 후보는 한 번만 묻습니다. 모델은 후보를 바꾸지 못합니다"
            cuts.append({**common, "kind": "decision", "t0": decisions[0]["start"], "t1": decisions[-1]["end"], "shown": shown, "questions": decisions, "verdict_note": note, "verdict_color": color, "label": f"Zoro-1 판단: {what}", "note": hint})
        if cfg.get("write") and writers and not cfg.get("_written"):
            cfg["_written"] = True
            cuts.append({**common, "kind": "writer", "t0": writers[0]["start"], "t1": writers[0]["end"], "shown": 8.0, "event": writers[0], "label": f"Zoro-1 글쓰기: {write_kind(writers[0]['detail']['system'])} 초안을 씁니다", "note": "판단이 끝난 뒤 임베딩 표를 더 올립니다"})
        cuts.sort(key=lambda c: c["t0"])
        cur = base["t0"]
        pieces = []
        for c in cuts:
            if c["t0"] > cur + 0.3:
                pieces.append(sub(base, cur, c["t0"]))
            pieces.append({**c, "real": c["t1"] - c["t0"], "speed": max(1.0, (c["t1"] - c["t0"]) / c["shown"])})
            cur = c["t1"]
        if not pieces:
            pieces.append(sub(base, cur, base["t1"]))
        elif is_scan and cfg.get("read"):
            pieces.append({**sub(base, cur, base["t1"]), "label": "모델은 글자만 읽습니다. 서류 종류와 항목은 규칙이 정합니다", "note": "제목과 항목 이름으로 값을 뽑고, 대장에 비슷한 값이 하나뿐일 때만 오독을 고칩니다"})
        elif base["t1"] > cur + 1.0:
            pieces.append(sub(base, cur, base["t1"]))
        scenes.extend(pieces)
        if cfg.get("gantt") and (i + 1 == len(marks) or marks[i + 1].get("flow") != flow):
            scenes.append({"kind": "gantt", "t0": t1, "t1": t1, "shown": 9.0, "trace": trace, "span": (starts[flow], stops[flow]), "track": m["track"], "title": titles[flow], "flow_start": starts[flow], "screen": "kiosk", "flow": flow, "label": "한 건의 모델 사용 순서", "note": "SigLIP2 → gocr 비전 → gocr 언어 → Zoro-1 판단 → Zoro-1 글쓰기", "real": 0.1, "speed": 1.0})
    return scenes


def model_stats(trace):
    judge = [e["ms"] for e in trace.work if e["key"] == "zoro_j"]
    write = [e["ms"] for e in trace.work if e["key"] == "zoro_w"]
    pages = sum(e["detail"]["pages"] for e in trace.work if e["key"] == "siglip2")
    sig = sum(e["ms"] for e in trace.work if e["key"] == "siglip2")
    crops = sum(e["detail"]["inputs"] for e in trace.work if e["key"] == "gocr_l")
    ocr = sum(e["ms"] for e in trace.work if e["key"] in ("gocr_v", "gocr_l"))
    return f"실측: SigLIP2 장당 {sig / max(1, pages) / 1000:.1f}초 · gocr 조각당 {ocr / max(1, crops) / 1000:.1f}초 · Zoro-1 판단 {statistics.median(judge) / 1000:.1f}초 · 글쓰기 {statistics.median(write) / 1000:.1f}초"


def main():
    folder, results_path, trace_path, out = sys.argv[1:5]
    stats = json.load(open(sys.argv[5], encoding="utf8")) if len(sys.argv) > 5 else {}
    cases_root = os.environ.get("CASES", "out/cases")
    root = os.path.dirname(os.path.dirname(cases_root.rstrip("/\\")))
    scenarios = {s["id"]: s for s in json.load(open(os.path.join(root, "core", "fixtures", "scenarios.json"), encoding="utf8"))["scenarios"]}
    frames = json.load(open(os.path.join(folder, "frames.json"), encoding="utf8"))
    results = json.load(open(results_path, encoding="utf8"))
    trace = Trace(trace_path, results["t0_epoch_ms"])
    skip = set(filter(None, os.environ.get("SKIP", "").split(",")))
    marks = results["timeline"]
    end = max(frames["kiosk"][-1]["t"], frames["officer"][-1]["t"])
    screens = {k: Screen(folder, k, v) for k, v in frames.items()}
    print("프레임", {k: len(v) for k, v in frames.items()}, "길이", round(end), "모델 기록", len(trace.events))
    scenes = build_scenes(marks, end, trace, skip, cases_root, scenarios)
    print("장면", len(scenes), "예상 길이", round(sum(s["shown"] for s in scenes)))

    stills = os.environ.get("STILLS")
    if stills:
        os.makedirs(stills, exist_ok=True)
        wanted = os.environ.get("STILL_KINDS", "")
        for i, s in enumerate(scenes):
            if s.get("kind", "screen") == "screen" and not wanted:
                continue
            for p in (0.9,):
                t = s["t0"] + p * (s["t1"] - s["t0"])
                frame(s, t, p, screens, trace).save(os.path.join(stills, f"{i:03d}_{s.get('flow', '')}_{s.get('kind', 'screen')}.png"))
        models_card().save(os.path.join(stills, "models_card.png"))
        print("장면 그림 저장", stills)
        return

    ffmpeg = os.environ["FFMPEG"]
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
        stdin=subprocess.PIPE,
    )
    written = 0

    def emit(img, count=1):
        nonlocal written
        raw = img.tobytes()
        for _ in range(count):
            proc.stdin.write(raw)
            written += 1

    emit(card([
        ("zorogravity 민원 심사 엔진", 64, TEXT),
        ("민원인 · 공무원 화면 교차 시연", 40, BLUE),
        ("정부 API 없이 합성 관계망 · 합성 서류로 동작합니다. 실제 개인정보가 아닙니다.", 28, MUTED),
        ("규칙이 처분을 정하고, 모델은 확신도만 냅니다. 불리한 처분은 자동으로 끝나지 않습니다.", 28, MUTED),
    ]), FPS * 4)
    emit(models_card(), FPS * 9)

    last_track = None
    prev_tail = None
    for s in scenes:
        if s["track"] != last_track and s["track"] in ("auto", "officer"):
            title = ("1부. 자동화 절차", "서류와 대장이 맞고 불리한 점이 없는 건만 사람 없이 끝납니다", GREEN) if s["track"] == "auto" else ("2부. 공무원 개입", "불리한 후보 · 확인이 필요한 건은 사유와 함께 공무원에게 갑니다", ORANGE)
            emit(card([(title[0], 60, title[2]), (title[1], 30, MUTED)]), int(FPS * 2.5))
            prev_tail = None
        last_track = s["track"]
        n = max(1, int(s["shown"] * FPS))
        modeled = s.get("kind", "screen") != "screen"
        for i in range(n):
            p = i / max(1, n - 1)
            t = s["t0"] + p * (s["t1"] - s["t0"]) if modeled else min(s["t0"] + (i / FPS) * s["speed"], s["t1"])
            img = frame(s, t, p, screens, trace)
            if prev_tail is not None and i < FADE * FPS:
                img = Image.blend(prev_tail, img, i / (FADE * FPS))
            emit(img)
        prev_tail = img

    lines = [("안정화 결과", 56, TEXT)]
    for text in stats.get("lines", []):
        lines.append((text, 30, MUTED))
    lines.append((model_stats(trace), 26, BLUE))
    lines.append(("모든 수치는 합성 데이터 기준입니다", 26, ORANGE))
    emit(card(lines), FPS * 7)
    proc.stdin.close()
    proc.wait()
    print("프레임", written, "길이", round(written / FPS), "초", out)


if __name__ == "__main__":
    main()
