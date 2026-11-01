"""크롭용 시각 앵커 학습: siglip2 패치 특징 → 영역 종류 (form · sign · table · title · guide).

    python scripts/train_crop_probe.py <특징 폴더> <서식 폴더> <출력.bin> [시험 특징 폴더] [시험 서식 폴더] [교차 검증 앵커 폴더]

교차 검증 앵커 폴더를 주면 서식 종류를 하나씩 뺀 앵커 crop_probe.<종류>.bin 도 저장합니다 (`crop_eval` 의 네 번째 인자).

한국어 문구 앵커는 패치 해상도에서 값과 제목을 구분하지 못합니다 (AUC 0.5~0.6). 같은 특징에 정답 서식으로 학습한
선형 가중치를 쓰면 처음 보는 서식 종류에서도 구분합니다. 앵커 문구의 벡터 자리에 들어가는 것이 전부라서 실행할 때 비용이 거의 없습니다.

서식 종류를 통째로 빼고 학습하는 교차 검증으로 λ 를 고르고, 처음 보는 서식 종류에서의 AUC 를 보고합니다.
출력 형식 (little endian): b"ZZP1", u32 k, u32 d, k 개의 (u32 길이, 이름), f32 mean[d], f32 weights[k*d], f32 intercept[k]
"""

import glob
import json
import os
import struct
import sys

import numpy as np

CLASSES = ["form", "sign", "table", "title", "guide"]
SOURCE = {"value": "form", "label": "form", "sign": "sign", "table": "table", "title": "title", "guide": "guide"}
# 칸의 이 비율 이상을 덮으면 그 종류로 칩니다.
COVER = 0.10


def load_grid(path):
    raw = open(path, "rb").read()
    r, c, d = struct.unpack("<III", raw[:12])
    a = np.frombuffer(raw[12:], dtype="<f4")
    return r, c, a[: r * c * d].reshape(r * c, d)


def labels_of(gt, r, c):
    """칸마다 종류별로 덮인 비율"""
    cw, ch = gt["width"] / c, gt["height"] / r
    cover = np.zeros((r * c, len(CLASSES)), dtype=np.float32)
    for g in gt["gt"]:
        name = SOURCE.get(g["kind"])
        if name is None:
            continue
        k = CLASSES.index(name)
        x0, y0, x1, y1 = g["box"]
        for i in range(max(0, int(y0 // ch)), min(r, int((y1 - 1) // ch) + 1)):
            for j in range(max(0, int(x0 // cw)), min(c, int((x1 - 1) // cw) + 1)):
                ox = min(x1, (j + 1) * cw) - max(x0, j * cw)
                oy = min(y1, (i + 1) * ch) - max(y0, i * ch)
                cover[i * c + j, k] += max(ox, 0) * max(oy, 0) / (cw * ch)
    return cover >= COVER


def load_set(feat, forms, maxp=256):
    X, Y, D, P = [], [], [], []
    for path in sorted(glob.glob(os.path.join(feat, f"*.p{maxp}.bin"))):
        stem = os.path.basename(path).split(".")[0]
        gt = json.load(open(os.path.join(forms, stem + ".json"), encoding="utf8"))
        r, c, x = load_grid(path)
        X.append(x)
        Y.append(labels_of(gt, r, c))
        D += [gt["doc"]] * len(x)
        P += [stem] * len(x)
    return np.concatenate(X), np.concatenate(Y), np.array(D), np.array(P)


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    x = np.concatenate([pos, neg])
    rank = np.empty(len(x))
    rank[x.argsort()] = np.arange(1, len(x) + 1)
    return (rank[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class Ridge:
    def __init__(self, X, Y, lam):
        self.mean = X.mean(0)
        A = (X - self.mean).astype(np.float64)
        G = A.T @ A + lam * np.eye(A.shape[1])
        y = Y.astype(np.float64)
        self.intercept = y.mean(0)
        self.w = np.linalg.solve(G, A.T @ (y - self.intercept))

    def score(self, X):
        return (X - self.mean) @ self.w + self.intercept


def cv(X, Y, D, lam):
    S = np.zeros(Y.shape)
    for doc in sorted(set(D)):
        te = D == doc
        S[te] = Ridge(X[~te], Y[~te], lam).score(X[te])
    return [auc(S[Y[:, k], k], S[~Y[:, k], k]) for k in range(len(CLASSES))], S


def save(model, path):
    with open(path, "wb") as f:
        f.write(b"ZZP1")
        f.write(struct.pack("<II", len(CLASSES), model.w.shape[0]))
        for name in CLASSES:
            b = name.encode()
            f.write(struct.pack("<I", len(b)) + b)
        f.write(model.mean.astype("<f4").tobytes())
        f.write(model.w.T.astype("<f4").tobytes())
        f.write(model.intercept.astype("<f4").tobytes())


def main():
    feat, forms, out = sys.argv[1:4]
    X, Y, D, _ = load_set(feat, forms)
    print(f"패치 {X.shape[0]} · 양성 비율 " + " ".join(f"{n} {Y[:, k].mean():.2f}" for k, n in enumerate(CLASSES)))
    best, best_lam = -1.0, None
    for lam in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0):
        a, _ = cv(X, Y, D, lam)
        print(f"λ={lam:<6} 처음 보는 서식 AUC " + " ".join(f"{n} {v:.3f}" for n, v in zip(CLASSES, a)))
        if np.mean(a) > best:
            best, best_lam = float(np.mean(a)), lam
    model = Ridge(X, Y, best_lam)
    save(model, out)
    print(f"선택 λ={best_lam} → {out}")
    if len(sys.argv) > 5:
        Xt, Yt, _, _ = load_set(sys.argv[4], sys.argv[5])
        s = model.score(Xt)
        print("시험 세트 AUC " + " ".join(f"{n} {auc(s[Yt[:, k], k], s[~Yt[:, k], k]):.3f}" for k, n in enumerate(CLASSES)))
    if len(sys.argv) > 6:
        os.makedirs(sys.argv[6], exist_ok=True)
        for doc in sorted(set(D)):
            keep = D != doc
            save(Ridge(X[keep], Y[keep], best_lam), os.path.join(sys.argv[6], f"crop_probe.{doc}.bin"))


if __name__ == "__main__":
    main()
