"""Offline-проверка retrieval-аима: вместо MLP ищем ближайшую траекторию в
датасете (окно WINDOW последних тиков относительно себя) и берём дельты прицела
учителя (target[0]=dyaw, target[1]=dpitch) от ближайшего соседа.

Метрика качества — совпадение предсказанного вектора наводки с учительским
(cosine + MAE + корреляция). Чем ближе к 1 cosine и меньше MAE, тем точнее
retrieval повторяет «как наводился друг».

Использование:
  python aim_knn.py                         # оценка на dataset_rec.npz
  python aim_knn.py --k 5 --index 40000 --query 3000 --window 16
"""
import argparse
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pvp_bot import dataset, config


def build_windows(states, targets, window):
    """Все эпизоды -> (N, window*FEATURE_DIM) ключей и (N, 2) целей (dyaw,dpitch)."""
    Xs, Ys = [], []
    for s, t in zip(states, targets):
        x, y = dataset.make_windows(s, t, window)
        if x.shape[0]:
            Xs.append(x)
            Ys.append(y[:, :2])  # только aim-компоненты
    if not Xs:
        return np.zeros((0, window * config.FEATURE_DIM)), np.zeros((0, 2))
    return np.vstack(Xs), np.vstack(Ys)


def knn_predict(idx_X, idx_Y, q_X, k):
    """Поиск k ближайших соседей чанками (без sklearn). Возвращает среднее по k
    соседям дельт (dyaw,dpitch) для каждого запроса."""
    idx_X = idx_X.astype(np.float32)
    idx_sq = (idx_X * idx_X).sum(axis=1)  # (M,)
    nq = q_X.shape[0]
    out = np.zeros((nq, 2), dtype=np.float64)
    q = q_X.astype(np.float32)
    CH = 256  # запросов в чанке, чтобы не раздуть матрицу расстояний
    for start in range(0, nq, CH):
        chunk = q[start:start + CH]                       # (b, D)
        csq = (chunk * chunk).sum(axis=1)[:, None]        # (b, 1)
        d2 = csq + idx_sq[None, :] - 2.0 * (chunk @ idx_X.T)  # (b, M)
        np.maximum(d2, 0.0, out=d2)
        kk = min(k, d2.shape[1])
        part = np.argpartition(d2, kk - 1, axis=1)[:, :kk]     # (b, kk)
        rows = np.arange(chunk.shape[0])[:, None]
        best = d2[rows, part]                                  # (b, kk)
        w = 1.0 / (np.sqrt(best) + 1e-6)                       # веса по расстоянию
        w = w / w.sum(axis=1, keepdims=True)
        neigh = idx_Y[part]                                    # (b, kk, 2)
        out[start:start + chunk.shape[0]] = (neigh * w[:, :, None]).sum(axis=1)
    return out


def evaluate(X, Y, window, k, index_n, query_n, seed):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    index_n = min(index_n, max(1, N - 1))
    query_n = min(query_n, N - index_n) if N - index_n > 0 else index_n
    idx_sel = perm[:index_n]
    q_sel = perm[index_n:index_n + query_n]

    idx_X, idx_Y = X[idx_sel], Y[idx_sel]
    q_X, q_Y = X[q_sel], Y[q_sel]

    pred = knn_predict(idx_X, idx_Y, q_X, k)

    def report(name, p, t):
        mae = np.abs(p - t).mean()
        std = t.std() + 1e-9
        corr = np.corrcoef(p, t)[0, 1] if t.std() > 1e-6 else float("nan")
        print(f"  {name:6s}: mae={mae:7.3f}  (std_учит.={std:6.3f})  corr={corr:6.3f}")

    print(f"[aim_knn] window={window} k={k} index={idx_X.shape[0]} query={q_X.shape[0]}")
    report("dyaw", pred[:, 0], q_Y[:, 0])
    report("dpitch", pred[:, 1], q_Y[:, 1])

    # cosine между предсказанным и учительским вектором наводки (на тик)
    a = q_Y
    b = pred
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-9)
    # учесть только тики, где учитель реально крутил прицел (нормы не нулевые)
    mask = np.linalg.norm(a, axis=1) > 0.5
    print(f"  aim cosine: mean={cos.mean():.3f}  mean(|учит.дельта|>0.5)={cos[mask].mean():.3f}")

    # baseline «ничего не крутим»: предсказываем средний вектор датасета
    base = idx_Y.mean(axis=0)
    cosb = (a * base).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(base) + 1e-9)
    print(f"  baseline(const) cosine: mean={cosb.mean():.3f}")


def export_index(states, targets, window, out_path, n, seed):
    """Строит индекс для мода: случайная подвыборка окон (M,D) + aim-дельты (M,2),
    пишет кастомный бинарь (little-endian), читаемый RecAim.java."""
    rng = np.random.default_rng(seed)
    X, Y = build_windows(states, targets, window)
    n = min(n, X.shape[0])
    sel = rng.permutation(X.shape[0])[:n]
    W = X[sel].astype(np.float32)
    Yaim = Y[sel].astype(np.float32)
    with open(out_path, "wb") as f:
        f.write(b"PVRI")
        f.write(struct.pack("<ii", W.shape[0], W.shape[1]))
        f.write(W.tobytes())
        f.write(Yaim.tobytes())
    print(f"[export] {out_path}: M={W.shape[0]} D={W.shape[1]}  (okna={X.shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "dataset_rec.npz"))
    ap.add_argument("--window", type=int, default=config.WINDOW)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--index", type=int, default=40000, help="размер базы соседей")
    ap.add_argument("--query", type=int, default=3000, help="сколько тиков тестируем")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", default=None,
                    help="путь для записи rec_index.bin (индекс для мода)")
    ap.add_argument("--export-n", type=int, default=15000,
                    help="число окон в индексе")
    args = ap.parse_args()

    states, targets = dataset.load_npz(args.data)
    X, Y = build_windows(states, targets, args.window)
    print(f"[aim_knn] окон всего: {X.shape}, целей(aim): {Y.shape}")
    if X.shape[0] == 0:
        print("пусто — проверь путь к датасету"); return
    if args.export:
        export_index(states, targets, args.window, args.export, args.export_n, args.seed)
        return
    evaluate(X, Y, args.window, args.k, args.index, args.query, args.seed)


if __name__ == "__main__":
    main()
