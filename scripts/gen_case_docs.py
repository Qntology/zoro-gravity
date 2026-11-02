"""시나리오(`core/fixtures/scenarios.json`)에 맞는 합성 서류 이미지를 만듭니다. 실제 개인정보는 들어가지 않습니다.

    python scripts/gen_case_docs.py <출력 폴더> [시드]

출력: <폴더>/<시나리오>/<서류>.jpg, <폴더>/truth.json
truth.json 은 서류별 정답 값입니다. OCR 추출 평가(`case_read`)가 씁니다.
"""

import json
import os
import random
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import gen_sample_forms as g  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")

# 서류 이름 → (서식 함수, 시나리오 값을 서식 값으로 바꾸는 함수)
def lease_values(f, s):
    return {"landlord": f["landlord_name"], "landlord_birth": f["landlord_birth"], "tenant": f["tenant_name"], "tenant_birth": s["applicant"]["birth"], "rent": f["contract_rent"], "deposit": f.get("deposit", "10000000")}


def statement_values(f, s):
    return {"name": f["holder"], "account": f["printed_account"], "rent": f["latest_transfer"]}


def application_values(f, s):
    return {"name": s["applicant"]["name"], "birth": s["applicant"]["birth"], "rent": f["applied_rent"], "account": s["form"]["handwritten_account"], "landlord": s["docs"]["임대차계약서"]["fields"]["landlord_name"]}


def consent_values(f, s):
    return {"name": s["applicant"]["name"]}


def receipt_values(f, s):
    return {"receipts": f["receipts"]}


def settlement_values(f, s):
    return {"org": f["org_name"], "program": f["program"], "claimed": f["claimed_total"], "name": s["applicant"]["name"]}


def bankbook_values(f, s):
    return {"name": f["holder"], "account": f["account"]}


def ledger_values(f, s):
    return {"address": f["address"], "use": f["use"], "violation": f.get("violation")}


def hygiene_values(f, s):
    return {"name": f["name"], "birth": f.get("birth", s["applicant"]["birth"]), "issued": f["issued"]}


def fire_values(f, s):
    return {"name": s["applicant"]["name"], "business": s["form"]["business_name"], "address": f["address"], "issued": f["issued"]}


def report_values(f, s):
    return {"name": s["applicant"]["name"], "birth": s["applicant"]["birth"], "business": s["form"]["business_name"], "address": s["form"]["address"], "kind": s["form"]["business_type"]}


def poa_values(f, s):
    return {"principal": f["principal_name"], "principal_birth": s["form"]["principal_birth"], "agent": f["agent_name"], "agent_birth": s["applicant"]["birth"], "relation": s["form"]["relation"], "date": f["poa_date"]}


def id_values(f, s):
    return {"name": f["name"], "birth": f["birth"]}


def petition_values(f, s):
    return {"name": s["applicant"]["name"], "contact": s["form"].get("contact", "010-2345-6789"), "text": f["text"]}


TEMPLATES = {
    "임대차계약서": (g.lease, lease_values),
    "이체내역서": (g.transfer, statement_values),
    "신청서": (g.rent_application, application_values),
    "금융정보동의서": (g.consent, consent_values),
    "영수증철": (g.receipts, receipt_values),
    "정산총괄표": (g.settlement, settlement_values),
    "세금계산서": (g.tax_invoice, lambda f, s: {}),
    "통장사본": (g.bankbook, bankbook_values),
    "건축물대장": (g.building_ledger, ledger_values),
    "위생교육필증": (g.hygiene_cert, hygiene_values),
    "소방안전필증": (g.fire_cert, fire_values),
    "영업신고서": (g.biz_report, report_values),
    "위임장": (g.poa, poa_values),
    "위임자신분증": (g.id_copy, id_values),
    "신분증": (g.id_copy, id_values),
    "진정서": (g.petition, petition_values),
}


def render(fn, values, rng, doc):
    page = g.Page(rng, doc, values)
    # 시나리오 서류의 값은 비우지 않습니다. 빈칸은 시나리오가 정합니다.
    page.blank_rate = 0.0
    fn(page)
    img, _ = g.scan(page.img, rng)
    scale = rng.uniform(0.9, 1.33)
    return img.resize((int(g.W * scale), int(g.H * scale)), Image.LANCZOS if scale < 1 else Image.BICUBIC)


def main():
    out = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11
    with open(os.path.join(ROOT, "core", "fixtures", "scenarios.json"), encoding="utf8") as f:
        scenarios = json.load(f)["scenarios"]
    truth = {}
    count = 0
    for s in scenarios:
        folder = os.path.join(out, s["id"])
        os.makedirs(folder, exist_ok=True)
        docs = dict(s["docs"])
        docs["신분증"] = {"fields": {"name": s["applicant"]["name"], "birth": s["applicant"]["birth"]}}
        truth[s["id"]] = {}
        for kind, doc in docs.items():
            fn, to_values = TEMPLATES[kind]
            values = to_values(doc["fields"], s)
            values["signed"] = doc.get("signed", True)
            rng = random.Random(f"{seed}/{s['id']}/{kind}")
            img = render(fn, values, rng, kind)
            img.save(os.path.join(folder, f"{kind}.jpg"), quality=rng.randint(84, 92))
            truth[s["id"]][kind] = {"kind": doc.get("as", kind), "signed": doc.get("signed", True), "fields": doc["fields"]}
            count += 1
    count += sign_probe(out, scenarios, seed, truth)
    with open(os.path.join(out, "truth.json"), "w", encoding="utf8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=1)
    print(f"{count} 장 생성: {out}")


def sign_probe(out, scenarios, seed, truth):
    """서명 판별 시험: 서명란이 있는 서식을 서명한 쪽과 안 한 쪽으로 같은 수만큼 만듭니다."""
    count = 0
    picks = [s for s in scenarios if s["id"] in ("W1-clean", "S1-clean", "P1-clean", "C1-clean", "G1-noise")]
    for s in picks:
        for kind, doc in s["docs"].items():
            if kind not in ("신청서", "금융정보동의서", "정산총괄표", "영업신고서", "위임장", "진정서"):
                continue
            case = f"sign-{kind}"
            folder = os.path.join(out, case)
            os.makedirs(folder, exist_ok=True)
            truth.setdefault(case, {})
            fn, to_values = TEMPLATES[kind]
            for i in range(4):
                signed = i % 2 == 0
                values = to_values(doc["fields"], s)
                values["signed"] = signed
                name = f"{'signed' if signed else 'blank'}{i}"
                rng = random.Random(f"{seed}/sign/{kind}/{i}")
                render(fn, values, rng, kind).save(os.path.join(folder, f"{name}.jpg"), quality=rng.randint(84, 92))
                truth[case][name] = {"kind": kind, "signed": signed, "fields": {}}
                count += 1
    return count


if __name__ == "__main__":
    main()
