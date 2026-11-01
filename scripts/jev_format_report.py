"""jev_format_search 가 남긴 점수로 형식 조합을 교차 검증합니다.

    python scripts/jev_format_report.py out/jev_scores.txt

조합을 많이 재면 가장 잘 맞는 조합이 우연히 뽑힙니다. 그래서 사례를 절반씩 나눠 한쪽으로 조합을 고르고
다른 쪽으로 재는 일을 되풀이합니다 (사례 단위라 정순 · 역순이 함께 움직입니다).
스케일은 순위를 바꾸지 않으므로 교차 검증에는 영향이 없고, 확률의 크기(기준 0.95 를 넘는지)만 바꿉니다.
"""
import math
import random
import sys
from collections import defaultdict

WRAPPERS = ["시스템+턴", "턴", "시스템+턴+think", "시스템+턴+빈think", "턴+think", "턴+빈think", "틀없음"]
BODIES = ["task/facts", "최상위키", "context/options", "글", "한줄JSON", "줄목록", "제목+줄목록"]
QUERIES = ["끝", "턴닫음", "assistant", "선택지닫음"]
OPTION_POS = ["끝글자", "첫글자", "평균", "여는따옴표", "닫는따옴표"]
NORMS = ["정규화후", "정규화전"]
BEFORE = (0, 4, 0, 0, 0)
SCALE = 1 / 16


def load(path):
    cases, rows = {}, defaultdict(dict)
    with open(path, encoding="utf8") as f:
        for line in f:
            if line.startswith("#"):
                _, i, name, gate, pct = line.split()
                cases[int(i)] = (name, gate, float(pct))
                continue
            p = line.split()
            key = tuple(int(v) for v in p[:5])
            rows[key][(int(p[5]), int(p[6]))] = (int(p[7]), [float(v) for v in p[8:]])
    return cases, rows


def softmax(s, scale):
    top = max(s)
    e = [math.exp((v - top) * scale) for v in s]
    t = sum(e)
    return [v / t for v in e]


def hit(entry):
    want, s = entry
    return max(range(len(s)), key=s.__getitem__) == want


def label(k):
    return f"{WRAPPERS[k[0]]} · {BODIES[k[1]]} · 질의 {QUERIES[k[2]]} · 선택지 {OPTION_POS[k[3]]} · {NORMS[k[4]]}"


def main():
    cases, rows = load(sys.argv[1])
    case_ids = sorted(cases)
    hits = {k: {c: sum(hit(v[(c, o)]) for o in (0, 1)) for c in case_ids} for k, v in rows.items()}
    total = {k: sum(v.values()) for k, v in hits.items()}
    n = 2 * len(case_ids)

    print(f"조합 {len(rows)}개 · 질문 {n}개 · 탐색 전 조합 {total[BEFORE]}/{n}")

    print("\n축별 평균 정확도")
    for name, labels, axis in [("틀", WRAPPERS, 0), ("본문", BODIES, 1), ("질의", QUERIES, 2), ("선택지", OPTION_POS, 3), ("정규화", NORMS, 4)]:
        parts = []
        for i, l in enumerate(labels):
            vals = [t for k, t in total.items() if k[axis] == i]
            parts.append(f"{l} {100 * sum(vals) / (len(vals) * n):.1f}%")
        print(f"  {name}: " + " · ".join(parts))

    print("\n같은 조건에서 선택지 위치만 바꾼 짝 비교 (닫는따옴표 - 끝글자)")
    pairs = [(total[k], total[(k[0], k[1], k[2], 0, k[4])]) for k in rows if k[3] == 4]
    better = sum(a > b for a, b in pairs)
    worse = sum(a < b for a, b in pairs)
    print(f"  조건 {len(pairs)}개: 닫는따옴표가 나음 {better} · 같음 {len(pairs) - better - worse} · 못함 {worse} · 평균 차이 {sum(a - b for a, b in pairs) / len(pairs) / n * 100:+.1f}%p")
    gain = {c: sum(hits[k][c] - hits[(k[0], k[1], k[2], 0, k[4])][c] for k in rows if k[3] == 4) / len(pairs) / 2 for c in case_ids}
    line = sorted(gain.items(), key=lambda v: v[1])
    print("  가장 나빠진 사례: " + ", ".join(f"{cases[c][0]} {v:+.2f}" for c, v in line[:3]))
    print("  가장 좋아진 사례: " + ", ".join(f"{cases[c][0]} {v:+.2f}" for c, v in line[-3:]))

    print("\n절반으로 고르고 나머지 절반으로 재기 (무작위 나눔 2000번)")
    print("  사례 하나 빼기는 동점 조합이 많으면 뺀 사례를 틀린 조합을 고르게 되어 너무 비관적이라 쓰지 않습니다.")
    rng = random.Random(7)
    for name, keep in [("모든 조합", lambda k: True), ("닫는따옴표만", lambda k: k[3] == 4), ("끝글자만", lambda k: k[3] == 0)]:
        pool = [k for k in rows if keep(k)]
        scores, chosen = [], defaultdict(int)
        for _ in range(2000):
            ids = case_ids[:]
            rng.shuffle(ids)
            half = len(ids) // 2
            pick, test = ids[:half], ids[half:]
            top = max(sum(hits[k][c] for c in pick) for k in pool)
            ties = [k for k in pool if sum(hits[k][c] for c in pick) == top]
            # 동점이면 어느 것을 골라도 되므로 평균을 냅니다.
            scores.append(sum(sum(hits[k][c] for c in test) for k in ties) / len(ties) / (2 * len(test)))
            for k in ties:
                chosen[k] += 1
        common = max(chosen, key=chosen.get)
        print(f"  {name}: 평균 {100 * sum(scores) / len(scores):.1f}% · 가장 자주 뽑힘: {label(common)}")

    print("\n축마다 평균이 가장 높은 값을 합친 조합")
    best = []
    for axis, size in enumerate([len(WRAPPERS), len(BODIES), len(QUERIES), len(OPTION_POS), len(NORMS)]):
        means = [sum(t for k, t in total.items() if k[axis] == i) / max(1, sum(1 for k in total if k[axis] == i)) for i in range(size)]
        best.append(max(range(size), key=means.__getitem__))
    best = tuple(best)
    print(f"  {label(best)}: {total[best]}/{n}")

    print("\n닫는따옴표 상위 조합과 확률 (스케일 1/16 · 정답 1순위 건의 평균 · 0.95 이상 건의 정밀도)")
    top = sorted((k for k in rows if k[3] == 4), key=lambda k: -total[k])[:12]
    seen = set()
    for k in top:
        sig = (k[1], k[3], k[4], k[2], k[0] if k[2] == 0 else (k[0] in (0, 2, 3)))
        if sig in seen:
            continue
        seen.add(sig)
        confident, right, conf = 0, 0, []
        for v in rows[k].values():
            want, s = v
            p = softmax(s, SCALE)
            pick = max(range(len(p)), key=p.__getitem__)
            if pick == want:
                conf.append(p[want])
            if p[pick] >= 0.95:
                confident += 1
                right += pick == want
        last_pick = sum(max(range(len(s)), key=s.__getitem__) == len(s) - 1 for _, s in rows[k].values())
        last_want = sum(want == len(s) - 1 for want, s in rows[k].values())
        print(f"  {total[k]:>2}/{n} 평균 {sum(conf) / max(1, len(conf)):.3f} · ≥0.95 {confident}건 중 {right}건 맞음 · 마지막 선택지 고름 {last_pick} (정답 {last_want}) · {','.join(map(str, k))} · {label(k)}")

    print("\n명세 사례의 기대 확률과 비교 (스케일별 정답 확률 평균, 명세 평균과 비교)")
    spec = [c for c in case_ids if cases[c][2] > 0]
    want_mean = sum(cases[c][2] for c in spec) / len(spec) / 100
    print(f"  명세 평균 {want_mean:.3f}")
    for k in [BEFORE] + sorted((k for k in rows if k[3] == 4), key=lambda k: -total[k])[:1]:
        line = []
        for scale in (1 / 32, 1 / 16, 1 / 8, 1 / 4):
            ps = []
            for c in spec:
                for o in (0, 1):
                    want, s = rows[k][(c, o)]
                    ps.append(softmax(s, scale)[want])
            line.append(f"×{scale * 16:g} {sum(ps) / len(ps):.3f}")
        print(f"  {label(k)}: " + " · ".join(line))


    print("\n스케일 보정: 질문 전체의 평균 음의 로그우도 (낮을수록 확률이 실제 정답률과 맞습니다)")
    candidates = [BEFORE, (0, 4, 0, 4, 0), (0, 5, 1, 4, 0), (1, 0, 1, 4, 1), (0, 3, 2, 4, 0), (0, 4, 1, 4, 0)]
    grid = (1 / 64, 1 / 32, 1 / 24, 1 / 16, 1 / 12, 1 / 8, 1 / 4)
    for k in candidates:
        cells = []
        for scale in grid:
            nll = 0.0
            for want, s in rows[k].values():
                nll -= math.log(max(softmax(s, scale)[want], 1e-12))
            cells.append(f"1/{1 / scale:g} {nll / len(rows[k]):.2f}")
        print(f"  {','.join(map(str, k))} {total[k]}/{n}: " + " · ".join(cells))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf8")
    main()
