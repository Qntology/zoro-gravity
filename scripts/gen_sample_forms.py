"""합성 민원 서류 생성기. 크롭 · OCR 시험용 이미지와 정답 상자를 만듭니다. 실제 개인정보는 들어가지 않습니다.

    python scripts/gen_sample_forms.py <출력 폴더> [문서당 장수] [시드]

출력: <종류>_<번호>.jpg, <종류>_<번호>.json
정답 상자의 kind
  title    문서 제목
  value    민원인 · 기관이 채운 값 (읽어야 함)
  sign     서명 · 날인 (읽어야 함)
  table    증빙 표 전체 (읽어야 함)
  label    항목 이름처럼 값 옆에 붙은 고정 인쇄 (값과 함께 읽음)
  guide    고정 인쇄 안내문 (읽지 않아도 됨)
손글씨 글꼴이 없어서 붓 글꼴을 글자마다 흔들어 푸른 잉크로 찍습니다.
"""

import json
import math
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = r"C:\Windows\Fonts"
GOTHIC = os.path.join(FONT_DIR, "malgun.ttf")
BOLD = os.path.join(FONT_DIR, "malgunbd.ttf")
SERIF = os.path.join(FONT_DIR, "batang.ttc")
GULIM = os.path.join(FONT_DIR, "gulim.ttc")

W, H = 1240, 1754  # A4 150 DPI
INK = (20, 20, 25)
BLUE = (25, 35, 120)
RED = (200, 30, 30)
GRAY = (232, 232, 232)

FAMILY = list("김이박최정강조윤장임한오서신권황안송류홍")
GIVEN = ["민수", "서연", "지훈", "수빈", "철수", "영희", "하늘", "도윤", "유진", "현우", "지아", "성민"]
CITY = ["서울특별시 마포구 월드컵로", "서울특별시 관악구 신림로", "부산광역시 해운대구 우동로", "대전광역시 유성구 대학로", "경기도 수원시 팔달구 효원로", "인천광역시 남동구 구월로"]
BANK = ["국민은행", "신한은행", "우리은행", "하나은행", "농협은행", "기업은행"]
ITEM = ["사무용품", "간식비", "현수막 제작", "대관료", "인쇄비", "교통비", "강사료", "다과 구입", "장비 임대"]
SHOP = ["행복상사", "대한문구", "우리인쇄", "한빛기획", "푸른식품", "미래광고"]
LAW = [
    "개인정보 보호법 제15조에 따라 수집 · 이용에 동의합니다. 동의를 거부할 수 있으나 거부 시 지원이 제한될 수 있습니다.",
    "신청 내용이 사실과 다를 경우 관계 법령에 따라 지원금이 환수될 수 있으며, 민형사상 책임을 질 수 있습니다.",
    "제출한 서류는 심사 목적 외에 사용되지 않으며, 보유 기간이 끝나면 지체 없이 파기합니다.",
    "본 신청서는 정해진 기한까지 제출하여야 하며, 기한이 지난 경우 접수되지 않을 수 있습니다.",
    "처리 기간은 접수일로부터 14일 이내이며, 서류 보완이 필요한 경우 그 기간은 산입하지 않습니다.",
    "지원 대상자는 소득 및 재산 기준을 충족하여야 하며, 중복 수혜가 확인되면 지원이 중단됩니다.",
]
CLAUSE = [
    "임차인은 임대인의 동의 없이 임차주택의 구조를 변경하거나 전대 또는 담보로 제공하지 못한다.",
    "임대차가 종료된 경우 임차인은 임차주택을 원상으로 회복하여 임대인에게 반환한다.",
    "임차인이 차임을 2기 이상 연체한 때에는 임대인은 이 계약을 해지할 수 있다.",
    "임대인은 임대 기간 동안 임차주택을 사용 · 수익에 필요한 상태로 유지하여야 한다.",
    "계약 당사자가 이 계약의 내용을 이행하지 않을 경우 상대방은 서면으로 최고하고 계약을 해제할 수 있다.",
    "이 계약에 정하지 않은 사항은 민법 및 주택임대차보호법의 규정을 따른다.",
]
COMPLAINT = [
    "저는 위 주소에 거주하는 주민으로서 인근 공사장의 소음으로 인한 피해를 진정합니다.",
    "매일 아침 일곱 시부터 저녁 여덟 시까지 공사 소음이 이어져 일상생활이 어렵습니다.",
    "노부모님이 계신데 소음 때문에 잠을 이루지 못하시고 건강이 나빠지고 계십니다.",
    "구청에 두 차례 전화를 드렸지만 아직 아무런 조치가 없어 이렇게 글을 씁니다.",
    "공사 시간을 지켜 주시고 방음벽을 설치하도록 조치해 주시기를 간곡히 부탁드립니다.",
]


def font(path, size, index=0):
    return ImageFont.truetype(path, size, index=index)


class Page:
    def __init__(self, rng, doc):
        self.rng = rng
        self.doc = doc
        self.img = Image.new("RGB", (W, H), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)
        self.gt = []
        self.margin = rng.randint(70, 110)
        # 서식마다 글꼴 · 머리 칸 색 · 필기구가 다릅니다.
        self.base = rng.choice([GOTHIC, GOTHIC, GULIM, SERIF])
        self.shade = rng.choice([GRAY, GRAY, (245, 245, 245), (225, 232, 242), (255, 255, 255)])
        self.ink = rng.choice([BLUE, BLUE, (15, 15, 20), (30, 30, 60)])
        self.blank_rate = rng.choice([0.0, 0.0, 0.1])

    def add(self, kind, box, text=""):
        self.gt.append({"kind": kind, "box": [int(v) for v in box], "text": text})

    def text(self, x, y, s, size=22, path=None, fill=INK, kind=None):
        f = font(path or self.base, size)
        self.d.text((x, y), s, font=f, fill=fill)
        l, t, r, b = self.d.textbbox((x, y), s, font=f)
        if kind:
            self.add(kind, (l, t, r, b), s)
        return r, b

    def center(self, y, s, size, path=BOLD, kind=None, x0=0, x1=W):
        path = path or self.base
        f = font(path, size)
        tw = self.d.textlength(s, font=f)
        return self.text((x0 + x1 - tw) / 2, y, s, size, path, kind=kind)

    def hand(self, x, y, s, size=30, color=None, kind="value"):
        """글자마다 크기 · 기울기 · 위치를 흔들어 찍습니다."""
        rng = self.rng
        color = color or self.ink
        cx = x
        tilt = rng.uniform(-2.5, 2.5)
        (x0, y0, x1, y1) = (x, y, x, y)
        for ch in s:
            if ch == " ":
                cx += size * 0.45
                continue
            sz = int(size * rng.uniform(0.92, 1.1))
            f = font(SERIF, sz, index=2)  # 궁서
            pad = sz // 2
            tile = Image.new("RGBA", (sz * 2, sz * 2), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((pad // 2, pad // 2), ch, font=f, fill=color + (235,))
            tile = tile.rotate(rng.uniform(-7, 7) + tilt, resample=Image.BICUBIC)
            bbox = tile.getbbox()
            if bbox is None:
                cx += sz * 0.6
                continue
            # 글자칸 안의 높이를 지켜야 쉼표가 줄 위로 올라가지 않습니다.
            drop = bbox[1] - pad // 2
            tile = tile.crop(bbox)
            py = int(y + drop + rng.uniform(-3, 3))
            self.img.paste(tile, (int(cx), py), tile)
            x1 = max(x1, cx + tile.width)
            y0, y1 = min(y0, py), max(y1, py + tile.height)
            cx += tile.width + rng.uniform(1, 6)
        if kind:
            self.add(kind, (x0, y0, x1, y1), s)
        return x1, y1

    def typed(self, x, y, s, size=24, kind="value"):
        return self.text(x, y, s, size, GULIM, kind=kind)

    def sign(self, x, y, w=150, h=56):
        """이름 조각 위에 곡선을 겹친 서명"""
        rng = self.rng
        pts = []
        for i in range(rng.randint(3, 5)):
            a = (x + rng.uniform(0, w), y + rng.uniform(0, h))
            b = (x + rng.uniform(0, w), y + rng.uniform(0, h))
            c = (x + rng.uniform(0, w), y + rng.uniform(0, h))
            for t in np.linspace(0, 1, 40):
                pts.append((((1 - t) ** 2) * a[0] + 2 * (1 - t) * t * b[0] + t * t * c[0], ((1 - t) ** 2) * a[1] + 2 * (1 - t) * t * b[1] + t * t * c[1]))
        self.d.line(pts, fill=self.ink, width=3)
        self.add("sign", (x, y, x + w, y + h))

    def stamp(self, cx, cy, name, r=46):
        self.d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=RED, width=4)
        f = font(BOLD, 26)
        tw = self.d.textlength(name[:3], font=f)
        self.d.text((cx - tw / 2, cy - 17), name[:3], font=f, fill=RED)
        self.add("sign", (cx - r, cy - r, cx + r, cy + r), name)

    def grid(self, x, y, widths, heights, width=2):
        total_w, total_h = sum(widths), sum(heights)
        self.d.rectangle((x, y, x + total_w, y + total_h), outline=INK, width=width)
        cy = y
        for h in heights[:-1]:
            cy += h
            self.d.line((x, cy, x + total_w, cy), fill=INK, width=1)
        cx = x
        for w in widths[:-1]:
            cx += w
            self.d.line((cx, y, cx, y + total_h), fill=INK, width=1)

    def fields(self, y, title, rows, hand=True, cols=1):
        """왼쪽 머리 칸(회색) + 항목 칸 + 값 칸의 입력 표. rows: (항목, 값) 목록"""
        rng = self.rng
        x = self.margin
        tw = W - 2 * x
        head = rng.randint(100, 130)
        rh = rng.randint(50, 64)
        per = cols
        n = math.ceil(len(rows) / per)
        self.d.rectangle((x, y, x + head, y + rh * n), fill=self.shade)
        self.center(y + rh * n / 2 - 16, title, 22, None, kind="label", x0=x, x1=x + head)
        body = tw - head
        unit = body / per
        label_w = rng.randint(130, 170)
        widths = [head] + sum(([label_w, unit - label_w] for _ in range(per)), [])
        self.d.rectangle((x, y, x + tw, y + rh * n), outline=INK, width=2)
        for i in range(n):
            self.d.line((x + head, y + rh * i, x + tw, y + rh * i), fill=INK, width=1)
        self.d.line((x + head, y, x + head, y + rh * n), fill=INK, width=1)
        for i, (label, value) in enumerate(rows):
            r, c = i // per, i % per
            lx = x + head + unit * c
            ly = y + rh * r
            self.d.rectangle((lx + 1, ly + 1, lx + label_w, ly + rh - 1), fill=(245, 245, 245))
            self.d.line((lx + label_w, ly, lx + label_w, ly + rh), fill=INK, width=1)
            if c:
                self.d.line((lx, ly, lx, ly + rh), fill=INK, width=1)
            self.center(ly + rh / 2 - 14, label, 20, None, kind="label", x0=lx, x1=lx + label_w)
            if value and rng.random() >= self.blank_rate:
                vx, vy = lx + label_w + rng.randint(14, 40), ly + rh / 2 - 17
                if hand:
                    self.hand(vx, vy, value, rng.randint(26, 31))
                else:
                    self.typed(vx, vy + 3, value, rng.randint(22, 25))
        return y + rh * n

    def guide(self, y, lines, size=17, box=True):
        x = self.margin
        top = y
        size += self.rng.randint(-1, 2)
        for s in lines:
            f = font(self.base, size)
            words, line = s.split(" "), ""
            for wd in words:
                if self.d.textlength(line + " " + wd, font=f) > W - 2 * x - 30:
                    self.text(x + 14, y + 8, line.strip(), size)
                    y += size + 9
                    line = wd
                else:
                    line += " " + wd
            self.text(x + 14, y + 8, line.strip(), size)
            y += size + 11
        if box:
            self.d.rectangle((x, top, W - x, y + 12), outline=(120, 120, 120), width=1)
        self.add("guide", (x, top, W - x, y + 12))
        return y + 22

    def table(self, y, header, rows, widths=None, typed=True, kind="table"):
        rng = self.rng
        x = self.margin
        tw = W - 2 * x
        n = len(header)
        ws = widths or [tw / n] * n
        ws = [w * tw / sum(ws) for w in ws]
        rh = rng.randint(38, 46)
        hh = rh + 6
        self.d.rectangle((x, y, x + tw, y + hh), fill=self.shade)
        self.grid(x, y, ws, [hh] + [rh] * len(rows))
        cx = x
        for w, s in zip(ws, header):
            self.center(y + hh / 2 - 13, s, 19, None, x0=cx, x1=cx + w)
            cx += w
        for r, row in enumerate(rows):
            cx = x
            for c, (w, s) in enumerate(zip(ws, row)):
                ty = y + hh + rh * r + rh / 2 - 12
                if typed:
                    tx = cx + w - font(GULIM, 19).getlength(s) - 10 if s[:1].isdigit() else cx + 10
                    self.text(tx, ty, s, 19, GULIM)
                elif s:
                    self.hand(cx + 10, ty - 4, s, 24, kind=None)
                cx += w
        bottom = y + hh + rh * len(rows)
        if kind:
            self.add(kind, (x, y, x + tw, bottom))
        return bottom

    def footer(self, y, who, signer):
        """날짜 · 서명 줄"""
        rng = self.rng
        self.center(y, "20 26 년    10 월    31 일", 24, GOTHIC)
        self.hand(W / 2 - 60, y - 2, "26", 26, kind=None)
        y += 60
        x = W - self.margin - 480
        ex, _ = self.text(x, y + 8, f"{who} :", 24)
        self.hand(ex + 16, y + 2, signer, 32)
        self.text(x + 330, y + 8, "(서명 또는 인)", 19, fill=(90, 90, 90))
        if rng.random() < 0.5:
            self.sign(x + 190, y - 8)
        else:
            self.stamp(x + 410, y + 18, signer, 38)
        return y + 80


def person(rng):
    return rng.choice(FAMILY) + rng.choice(GIVEN)


def address(rng):
    return f"{rng.choice(CITY)} {rng.randint(1, 240)}, {rng.randint(101, 1503)}호"


def phone(rng):
    return f"010-{rng.randint(1000, 9999)}-{rng.randint(1000, 9999)}"


def account(rng):
    return f"{rng.randint(100, 999)}-{rng.randint(10, 99)}-{rng.randint(100000, 999999)}"


def birth(rng):
    return f"{rng.randint(80, 99)}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d}"


def won(v):
    return f"{v:,}"


def header_box(p, y):
    """오른쪽 위 처리 기간 칸 (고정 인쇄)"""
    x = W - p.margin - 300
    p.grid(x, y, [150, 150], [30, 34])
    p.text(x + 26, y + 5, "처리기간", 18)
    p.text(x + 176, y + 5, "접수번호", 18)
    p.text(x + 42, y + 37, "14일", 18)
    p.add("guide", (x, y, x + 300, y + 64))


def rent_application(p):
    r = p.rng
    y = 70
    p.text(p.margin, y, "[별지 제1호 서식]", 17, fill=(80, 80, 80))
    header_box(p, y)
    y += 90
    _, y = p.center(y, "청년월세 지원 신청서", r.randint(38, 44), kind="title")
    y += 40
    name = person(r)
    y = p.fields(y, "신청인", [("성명", name), ("생년월일", birth(r)), ("주소", address(r)), ("연락처", phone(r))])
    y += 14
    rent = r.choice([300000, 350000, 400000, 450000, 500000])
    y = p.fields(y, "임대차", [("임대인", person(r)), ("소재지", address(r)), ("보증금", won(r.randint(5, 50) * 1000000) + " 원"), ("월세", won(rent) + " 원"), ("계약기간", "2026.03.01 ~ 2028.02.28")])
    y += 14
    y = p.fields(y, "지급계좌", [("은행명", r.choice(BANK)), ("계좌번호", account(r)), ("예금주", name)])
    y += 24
    y = p.guide(y, r.sample(LAW, 4))
    y += 30
    p.center(y, "위와 같이 청년월세 지원을 신청합니다.", 23, GOTHIC)
    p.footer(y + 60, "신청인", name)


def lease(p):
    r = p.rng
    y = 80
    _, y = p.center(y, "주택 임대차 계약서", r.randint(38, 44), kind="title")
    y += 34
    y = p.text(p.margin, y, "임대인과 임차인 쌍방은 아래 표시 주택에 관하여 다음과 같이 임대차 계약을 체결한다.", 19)[1] + 16
    rent = r.choice([300000, 400000, 500000, 600000])
    y = p.fields(y, "목적물", [("소재지", address(r)), ("임대부분", f"{r.randint(1, 15)}층 전부"), ("보증금", "금 " + won(r.randint(5, 50) * 1000000) + " 원정"), ("차임(월세)", "금 " + won(rent) + " 원정"), ("임대차기간", "2026.03.01 ~ 2028.02.28")], hand=False)
    y += 22
    lines = [f"제{i + 3}조 {s}" for i, s in enumerate(r.sample(CLAUSE, 5))]
    y = p.guide(y, lines, 18, box=False)
    y += 18
    p.center(y, "2026 년 02 월 15 일", 23, GOTHIC, kind="value")
    y += 50
    a, b = person(r), person(r)
    for who, name in (("임대인", a), ("임차인", b)):
        y0 = y
        y = p.fields(y, who, [("성명", name), ("주민번호", birth(r) + "-*******"), ("주소", address(r))], hand=True)
        p.stamp(W - p.margin - 70, y0 + 40, name, 38)
        y += 12


def transfer(p):
    r = p.rng
    y = 80
    p.text(p.margin, y, r.choice(BANK), 28, BOLD)
    y += 50
    _, y = p.center(y, "거래내역 확인서 (이체내역서)", r.randint(34, 40), kind="title")
    y += 34
    name = person(r)
    y = p.fields(y, "계좌정보", [("예금주", name), ("계좌번호", account(r)), ("조회기간", "2026.07.01 ~ 2026.09.30")], hand=False)
    y += 24
    rent = r.choice([300000, 400000, 500000])
    balance = r.randint(1, 9) * 1000000
    rows = []
    for month in range(7, 10):
        for k in range(r.randint(2, 4)):
            out = rent if k == 0 else r.randint(1, 40) * 10000
            balance -= out
            memo = "월세 이체" if k == 0 else r.choice(["카드대금", "통신요금", "관리비", "보험료"])
            rows.append([f"2026.{month:02d}.{r.randint(1, 28):02d}", memo, won(out), "0", won(balance)])
    y = p.table(y, ["거래일자", "적요", "출금", "입금", "잔액"], rows, [2, 2, 1.6, 1.2, 1.8])
    y += 30
    y = p.guide(y, ["본 확인서는 거래 사실을 확인하기 위해 발급되었으며 다른 용도로 사용할 수 없습니다."])
    p.stamp(W - p.margin - 80, y + 70, "은행장", 44)


def consent(p):
    r = p.rng
    y = 90
    _, y = p.center(y, "금융정보 등 제공 동의서", r.randint(36, 42), kind="title")
    y += 40
    name = person(r)
    y = p.guide(y, r.sample(LAW, 3) + [
        "금융기관 등의 장은 금융실명거래 및 비밀보장에 관한 법률 제4조에 따라 금융정보를 제공할 수 있습니다.",
        "제공 대상 정보는 예금 · 적금 · 주식 · 보험 · 대출 잔액 및 이자 소득 등입니다.",
    ], 19)
    y += 30
    y = p.fields(y, "제공자", [("성명", name), ("생년월일", birth(r)), ("주소", address(r)), ("관계", "본인")])
    y += 50
    p.center(y, "위 내용을 충분히 이해하고 금융정보 등의 제공에 동의합니다.", 23, GOTHIC)
    p.footer(y + 70, "동의자", name)


def tax_invoice(p):
    r = p.rng
    y = 90
    _, y = p.center(y, "세금계산서", r.randint(40, 46), kind="title")
    y += 30
    p.text(p.margin, y, "(공급받는자 보관용)", 18)
    y += 36
    y = p.fields(y, "공급자", [("등록번호", f"{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(10000, 99999)}"), ("상호", r.choice(SHOP)), ("성명", person(r)), ("사업장", address(r)), ("업태", "도소매"), ("종목", "문구")], hand=False, cols=2)
    y += 10
    y = p.fields(y, "공급받는자", [("등록번호", f"{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(10000, 99999)}"), ("상호", "마을공동체협의회"), ("성명", person(r)), ("사업장", address(r))], hand=False, cols=2)
    y += 24
    rows, total = [], 0
    for _ in range(r.randint(2, 5)):
        q, u = r.randint(1, 20), r.randint(1, 30) * 1000
        rows.append(["2026.10." + str(r.randint(10, 28)), r.choice(ITEM), str(q), won(u), won(q * u), won(q * u // 10)])
        total += q * u
    y = p.table(y, ["일자", "품목", "수량", "단가", "공급가액", "세액"], rows, [1.6, 2.4, 0.8, 1.3, 1.5, 1.2])
    y += 16
    y = p.fields(y, "합계", [("공급가액", won(total)), ("세액", won(total // 10)), ("합계금액", won(total + total // 10))], hand=False, cols=2)
    y += 30
    y = p.guide(y, ["이 금액을 영수함. 전자세금계산서는 국세청 홈택스에서 조회할 수 있습니다."])
    p.stamp(W - p.margin - 90, y + 70, "대표", 42)


def receipts(p):
    r = p.rng
    y = 70
    _, y = p.center(y, "영수증 첨부지", 34, kind="title")
    y += 34
    x = p.margin
    for i in range(r.randint(3, 4)):
        w, h = 470, r.randint(330, 420)
        rx = x + (i % 2) * 540 + r.randint(-8, 8)
        ry = y + (i // 2) * (h + 40) + r.randint(-6, 6)
        p.d.rectangle((rx, ry, rx + w, ry + h), fill=(252, 251, 246), outline=(170, 170, 170))
        p.text(rx + 130, ry + 14, "카드 매출전표", 24, BOLD)
        shop = r.choice(SHOP)
        amount = r.randint(3, 60) * 10000 - 2000
        lines = [("가맹점", shop), ("사업자번호", f"{r.randint(100, 999)}-{r.randint(10, 99)}-{r.randint(10000, 99999)}"), ("승인번호", str(r.randint(10000000, 99999999))), ("거래일시", f"2026-10-{r.randint(1, 28):02d} {r.randint(9, 20):02d}:{r.randint(10, 59)}"), ("품목", r.choice(ITEM)), ("합계", won(amount) + " 원")]
        ly = ry + 64
        top = ly
        for k, v in lines:
            p.text(rx + 26, ly, k, 19, GULIM)
            p.text(rx + 170, ly, v, 20, GULIM)
            ly += 40
        p.add("table", (rx + 20, top, rx + w - 20, ly))


def settlement(p):
    r = p.rng
    y = 90
    _, y = p.center(y, "보조사업 정산보고서 총괄표", r.randint(34, 40), kind="title")
    y += 36
    y = p.fields(y, "사업", [("사업명", "마을 환경 개선 사업"), ("단체명", "마을공동체협의회"), ("대표자", person(r)), ("사업기간", "2026.05.01 ~ 2026.10.31")])
    y += 24
    rows = []
    for item in r.sample(ITEM, r.randint(5, 7)):
        a = r.randint(5, 80) * 10000
        b = a - r.randint(0, 5) * 10000
        rows.append([item, won(a), won(b), won(a - b), ""])
    y = p.table(y, ["비목", "예산액", "집행액", "잔액", "비고"], rows, typed=False)
    y += 28
    y = p.guide(y, r.sample(LAW, 2))
    y += 30
    p.footer(y, "대표자", person(r))


def bankbook(p):
    r = p.rng
    x, y = p.margin + 60, r.randint(160, 360)
    w, h = 900, 520
    p.d.rectangle((x, y, x + w, y + h), outline=(120, 120, 120), width=2, fill=(250, 250, 244))
    p.text(x + 30, y + 26, r.choice(BANK), 34, BOLD)
    p.text(x + 30, y + 80, "보통예금 통장", 26)
    yy = y + 160
    name = person(r)
    for k, v in [("예금주", name), ("계좌번호", account(r)), ("개설일", f"20{r.randint(10, 24)}.{r.randint(1, 12):02d}.{r.randint(1, 28):02d}"), ("지점", r.choice(["마포", "관악", "유성", "팔달"]) + " 지점")]:
        p.text(x + 40, yy, k, 24)
        p.typed(x + 220, yy, v, 28)
        yy += 66
    p.add("title", (x + 30, y + 26, x + 400, y + 120), "통장")


def biz_report(p):
    r = p.rng
    y = 70
    p.text(p.margin, y, "[별지 제37호 서식]", 17, fill=(80, 80, 80))
    header_box(p, y)
    y += 90
    _, y = p.center(y, "영업 신고서", r.randint(38, 44), kind="title")
    y += 36
    y = p.fields(y, "신고인", [("성명", person(r)), ("생년월일", birth(r)), ("주소", address(r)), ("전화번호", phone(r))])
    y += 14
    y = p.fields(y, "영업소", [("상호", r.choice(SHOP)), ("소재지", address(r)), ("영업의 종류", "일반음식점"), ("영업장 면적", f"{r.randint(30, 180)}.{r.randint(0, 9)} ㎡")])
    y += 24
    y = p.guide(y, ["식품위생법 제37조제4항 및 같은 법 시행규칙 제42조제1항에 따라 위와 같이 신고합니다."] + r.sample(LAW, 2))
    y += 24
    p.footer(y, "신고인", person(r))
    y += 170
    y = p.guide(y, ["첨부서류: 1. 교육이수증 1부  2. 재산이나 시설 관련 서류 1부  3. 소방 · 안전시설 완비증명서 1부"], 18)


def certificate(p, title, rows, foot):
    r = p.rng
    y = r.randint(120, 200)
    p.text(p.margin, y - 50, f"제 {r.randint(2020, 2026)}-{r.randint(1000, 9999)} 호", 20, kind="value")
    _, y = p.center(y, title, r.randint(40, 46), kind="title")
    y += 80
    for k, v in rows:
        p.text(p.margin + 90, y, f"{k} :", 25)
        p.typed(p.margin + 300, y, v, 26)
        y += 66
    y += 40
    y = p.guide(y, [foot], 21, box=False)
    y += 60
    p.center(y, f"2026 년 {r.randint(1, 12):02d} 월 {r.randint(1, 28):02d} 일", 25, GOTHIC, kind="value")
    y += 80
    p.center(y, "한국식품위생협회장" if "위생" in title else "소방서장", 32)
    p.stamp(W / 2 + 200, y + 18, "직인", 44)


def hygiene_cert(p):
    r = p.rng
    certificate(p, "위생교육 이수증", [("성명", person(r)), ("생년월일", birth(r)), ("영업의 종류", "일반음식점"), ("교육 시간", "6 시간")], "위 사람은 식품위생법 제41조에 따른 위생교육을 이수하였음을 증명합니다.")


def fire_cert(p):
    r = p.rng
    certificate(p, "안전시설 등 완비증명서", [("상호", r.choice(SHOP)), ("대표자", person(r)), ("소재지", address(r)), ("업종", "일반음식점")], "위 영업장은 다중이용업소법 제9조에 따른 안전시설 등을 완비하였음을 증명합니다.")


def building_ledger(p):
    r = p.rng
    y = 80
    _, y = p.center(y, "일반건축물대장 (갑)", r.randint(34, 40), kind="title")
    y += 34
    y = p.fields(y, "건축물", [("대지위치", address(r)), ("지번", str(r.randint(100, 999))), ("연면적", f"{r.randint(100, 900)} ㎡"), ("주용도", "제2종 근린생활시설"), ("구조", "철근콘크리트"), ("높이", f"{r.randint(9, 30)} m")], hand=False, cols=2)
    y += 24
    rows = [[f"{i}층", r.choice(["철근콘크리트", "벽돌조"]), r.choice(["소매점", "일반음식점", "사무소", "주택"]), f"{r.randint(60, 240)}"] for i in range(1, r.randint(4, 7))]
    y = p.table(y, ["구분", "구조", "용도", "면적(㎡)"], rows)
    y += 24
    rows = [[f"20{r.randint(10, 25)}.{r.randint(1, 12):02d}.{r.randint(1, 28):02d}", r.choice(["사용승인", "소유권 이전", "증축", "위반건축물 표기"]), ""] for _ in range(r.randint(2, 4))]
    y = p.table(y, ["변동일", "변동내용", "비고"], rows, [1, 2, 1])
    y += 26
    p.guide(y, ["이 등본은 건축물대장의 원본 내용과 틀림없음을 증명합니다."])
    p.stamp(W - p.margin - 80, y + 110, "구청장", 44)


def poa(p):
    r = p.rng
    y = 100
    _, y = p.center(y, "인감증명서 발급 위임장", r.randint(36, 42), kind="title")
    y += 44
    a, b = person(r), person(r)
    y = p.fields(y, "위임자", [("성명", a), ("주민번호", birth(r) + "-*******"), ("주소", address(r))])
    y += 16
    y = p.fields(y, "대리인", [("성명", b), ("주민번호", birth(r) + "-*******"), ("주소", address(r)), ("관계", r.choice(["자녀", "배우자", "형제"]))])
    y += 34
    y = p.guide(y, ["위 사람에게 인감증명서 발급 신청 및 수령에 관한 권한을 위임합니다.", "위임장은 작성일로부터 6개월 이내에 사용하여야 합니다."] + r.sample(LAW, 1), 20)
    y += 40
    p.footer(y, "위임자", a)


def id_copy(p):
    r = p.rng
    x, y = r.randint(220, 340), r.randint(160, 520)
    w, h = 640, 400
    p.d.rounded_rectangle((x, y, x + w, y + h), 18, outline=(120, 120, 120), width=2, fill=(240, 245, 250))
    p.text(x + 150, y + 22, "주민등록증", 32, BOLD)
    p.add("title", (x + 150, y + 22, x + 400, y + 66), "주민등록증")
    p.d.rectangle((x + w - 190, y + 100, x + w - 40, y + 300), fill=(200, 205, 212), outline=(120, 120, 120))
    p.d.ellipse((x + w - 150, y + 130, x + w - 80, y + 205), fill=(120, 110, 105))
    p.d.pieslice((x + w - 180, y + 200, x + w - 50, y + 340), 180, 360, fill=(70, 75, 90))
    p.typed(x + 40, y + 110, person(r), 34)
    p.typed(x + 40, y + 170, birth(r) + "-*******", 26)
    p.typed(x + 40, y + 230, address(r)[:22], 21)
    p.typed(x + 40, y + 300, "2019. 5. 14.  마포구청장", 22)
    p.stamp(x + 390, y + 320, "구청", 28)


def petition(p):
    r = p.rng
    y = 120
    _, y = p.center(y, "진 정 서", r.randint(40, 46), kind="title")
    y += 60
    name = person(r)
    y = p.hand(p.margin + 20, y, f"진정인 {name}  {phone(r)}", 30)[1] + 20
    y = p.hand(p.margin + 20, y, address(r), 30)[1] + 40
    lines = r.sample(COMPLAINT, 5)
    for s in lines + r.sample(COMPLAINT, 2):
        words, line = s.split(" "), ""
        for wd in words:
            if len(line) + len(wd) > 30:
                p.d.line((p.margin, y + 46, W - p.margin, y + 46), fill=(190, 190, 215), width=1)
                p.hand(p.margin + 20, y, line, r.randint(30, 34))
                y += 62
                line = wd
            else:
                line = (line + " " + wd).strip()
        p.d.line((p.margin, y + 46, W - p.margin, y + 46), fill=(190, 190, 215), width=1)
        p.hand(p.margin + 20, y, line, r.randint(30, 34))
        y += 62
    p.footer(y + 30, "진정인", name)


def health_notice(p):
    r = p.rng
    y = 80
    _, y = p.center(y, "건강보험료 납입고지서", r.randint(36, 42), kind="title")
    y += 36
    y = p.fields(y, "납부자", [("성명", person(r)), ("고지년월", "2026년 10월"), ("납부기한", "2026.11.10"), ("고객번호", str(r.randint(1000000, 9999999)))], hand=False, cols=2)
    y += 24
    rows = [[k, won(r.randint(1, 30) * 1000)] for k in ["건강보험료", "장기요양보험료", "연체금", "경감액"]]
    y = p.table(y, ["항목", "금액"], rows)
    y += 26
    p.guide(y, r.sample(LAW, 3))


DOCS = {
    "rent_application": rent_application,
    "lease": lease,
    "transfer": transfer,
    "consent": consent,
    "tax_invoice": tax_invoice,
    "receipts": receipts,
    "settlement": settlement,
    "bankbook": bankbook,
    "biz_report": biz_report,
    "hygiene_cert": hygiene_cert,
    "fire_cert": fire_cert,
    "building_ledger": building_ledger,
    "poa": poa,
    "id_copy": id_copy,
    "petition": petition,
    "health_notice": health_notice,
}


def scan_noise(a, rng):
    """가장자리 그림자 · 접힌 자국 · 펀치 구멍"""
    h, w = a.shape[:2]
    if rng.random() < 0.4:
        edge = rng.randint(6, 26)
        fade = np.linspace(rng.uniform(0.55, 0.85), 1.0, edge, dtype=np.float32)
        side = rng.choice(["l", "r", "t"])
        if side == "l":
            a[:, :edge] *= fade[None, :, None]
        elif side == "r":
            a[:, -edge:] *= fade[::-1][None, :, None]
        else:
            a[:edge] *= fade[:, None, None]
    if rng.random() < 0.25:
        y = rng.randint(h // 3, 2 * h // 3)
        a[y : y + 2] *= 0.86
    if rng.random() < 0.2:
        for cy in (h // 3, 2 * h // 3):
            yy, xx = np.ogrid[:h, :w]
            a[(yy - cy) ** 2 + (xx - 36) ** 2 < 15**2] = 40.0
    return a


def rotate_box(box, deg):
    """정답 상자를 페이지 중심 기준으로 같이 돌립니다."""
    cx, cy = W / 2, H / 2
    a = math.radians(-deg)
    xs, ys = [], []
    for x, y in ((box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3])):
        dx, dy = x - cx, y - cy
        xs.append(cx + dx * math.cos(a) - dy * math.sin(a))
        ys.append(cy + dx * math.sin(a) + dy * math.cos(a))
    return [int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1]


def scan(img, rng):
    """스캔 느낌: 약한 기울기 · 흐림 · 종이 색 · 잡음"""
    deg = rng.uniform(-1.2, 1.2) if rng.random() < 0.3 else rng.uniform(-0.45, 0.45)
    out = img.rotate(deg, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
    out = out.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.6)))
    a = np.asarray(out).astype(np.float32)
    tint = np.array([rng.uniform(244, 255), rng.uniform(243, 254), rng.uniform(238, 252)], dtype=np.float32) / 255.0
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 2.2, a.shape)
    a = scan_noise(a * tint + noise, rng)
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if rng.random() < 0.3:
        out = out.convert("L").convert("RGB")
    return out, deg


def main():
    out = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    os.makedirs(out, exist_ok=True)
    for name, fn in DOCS.items():
        for i in range(count):
            rng = random.Random(f"{seed}/{name}/{i}")
            page = Page(rng, name)
            fn(page)
            img, deg = scan(page.img, rng)
            # 스캔 해상도도 장마다 다릅니다 (약 120 ~ 200 DPI).
            scale = rng.uniform(0.8, 1.33)
            sw, sh = int(W * scale), int(H * scale)
            img = img.resize((sw, sh), Image.LANCZOS if scale < 1 else Image.BICUBIC)
            stem = os.path.join(out, f"{name}_{i}")
            img.save(stem + ".jpg", quality=rng.randint(80, 92))
            gt = [{**g, "box": [int(v * scale) for v in rotate_box(g["box"], deg)]} for g in page.gt]
            with open(stem + ".json", "w", encoding="utf8") as f:
                json.dump({"doc": name, "width": sw, "height": sh, "gt": gt}, f, ensure_ascii=False, indent=0)
    print(f"{len(DOCS) * count} 장 생성: {out}")


if __name__ == "__main__":
    main()
