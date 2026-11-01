"""명세의 사례 표에서 jev_head 정답 사례를 뽑습니다. 명세 파일은 저장소에 없어서 결과도 out/ 에 둡니다.

    python scripts/extract_jev_cases.py <zorogravity.txt> <출력.json>

표의 한 행은 `| Case 1-B.<br> ... | 시나리오 | • `키`: 값<br> ... | **`정답 코드` (96.4%)**<br> ... |` 모양입니다.
행이 여러 줄에 걸쳐 있어 다음 `| Case` 까지 이어 붙여 읽습니다.
결과는 사례마다 id, 게이트, 사실(키 · 값), 정답 코드, 명세 확률입니다. 정답 코드가 없는 행(결정론 처리)은 뺍니다.
"""

import json
import re
import sys

# 게이트별 선택지. 명세에 나온 코드를 모두 모았습니다.
OPTIONS = {
    "G1": ["G1_DOC_MISMATCH_REJECT", "G1_SIGN_VALID", "G1_SIGN_INVALID_RETRY", "G1_SIGN_ELDERLY_STAMP", "G1_EXPIRED_MANDATORY_REJECT", "G1_INTERCEPT_FORCE_HOLD"],
    "G2": ["G2_FORCE_BANKBOOK_TRUTH", "G2_REJECT_CONTRADICTION_LOCK", "G2_REQUIRE_REPAIR_RECEIPT", "G2_BLOCK_UNPROVEN_EDIT"],
    "G3": ["G3_ELIGIBLE_NORMAL", "G3_ELIGIBLE_HOMONYM_CLEAR", "G3_VIOLATION_HIGH_RISK", "G3_DENY_UNMASK_PRIVACY", "G3_INTERCEPT_RUBBER_STAMP"],
    "G4": ["G4_FORCE_SUPPLEMENT_FORM", "G4_BLOCK_INTERNAL_LEAK", "G4_INTERCEPT_DOC_CORRUPTION"],
}

TITLES = {"G1": "형식심사", "G2": "정합성 교차대조", "G3": "실질 요건 심사", "G4": "문서 생성 통제"}


def clean(s):
    s = s.replace("`", "").replace("<br>", " ").replace("**", "")
    return re.sub(r"\s+", " ", s).strip()


def rows(text):
    """표의 사례 행들"""
    lines = text.split("\n")
    out, cur = [], None
    for line in lines:
        if line.startswith("| Case "):
            if cur is not None:
                out.append(cur)
            cur = line
        elif cur is not None:
            if line.startswith("---") or line.startswith("## ") or line.startswith("### "):
                out.append(cur)
                cur = None
            else:
                cur += "\n" + line
    if cur is not None:
        out.append(cur)
    return out


def facts_of(cell):
    """`• `키`: 값` 항목들"""
    items = re.findall(r"•\s*`([A-Za-z_0-9]+)`\s*:\s*(.*?)(?=(?:<br>\s*)*•\s*`[A-Za-z_0-9]+`\s*:|\Z)", cell, re.S)
    return [(k, clean(v)) for k, v in items]


def main():
    text = open(sys.argv[1], encoding="utf8").read()
    cases = []
    for row in rows(text):
        parts = row.split(" | ")
        if len(parts) < 4:
            continue
        cid = re.match(r"\| Case (\d-[A-Z])", parts[0]).group(1)
        answer = re.search(r"\*\*`((?:G\d|CAT)_[A-Z_0-9]+)`[^*]*?(\d+\.\d)%", row)
        if not answer:
            print("정답 코드 없음:", cid)
            continue
        code, pct = answer.group(1), float(answer.group(2))
        # 사실 칸은 정답이 처음 나오는 칸의 바로 앞 칸입니다.
        at = next(i for i, p in enumerate(parts) if f"**`{code}`" in p)
        facts = facts_of(parts[at - 1])
        gate = code[:2]
        if code not in OPTIONS.get(gate, []):
            print("선택지에 없는 코드:", cid, code)
            continue
        cases.append({"id": cid, "gate": gate, "title": TITLES[gate], "facts": facts, "answer": code, "spec_pct": pct, "options": OPTIONS[gate]})
    with open(sys.argv[2], "w", encoding="utf8") as f:
        json.dump(cases, f, ensure_ascii=False, indent=1)
    print(f"{len(cases)}건")
    for c in cases:
        print(c["id"], c["answer"], c["spec_pct"], [k for k, _ in c["facts"]])


if __name__ == "__main__":
    main()
