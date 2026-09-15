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


def knn_predict(idx_X, idx_Y, q_X, k, circ_cols=None, ww_arr=None):
    """Поиск k ближайших соседей чанками (без sklearn). Возвращает среднее по k
    соседям дельт (dyaw,dpitch) для каждого запроса. circ_cols — индексы
    столбцов циклического yaw_diff в проекции; ww_arr — квадраты весов
    расстояния на каждый столбец проекции (FEAT_W[f]*TEMP_W[w]), зеркалят
    метрику RecAim.dist2: d2 = Σ_w Σ_f FEAT_W[f]*TEMP_W[w]*diff², плюс
    циклическая поправка для yaw_diff (свёртка разности по окружности к
    (-180,180]°), как в игре."""
    raw_idx = idx_X.astype(np.float32)
    raw_q = q_X.astype(np.float32)
    if ww_arr is not None:
        sw = np.sqrt(ww_arr.astype(np.float32))
        idx_X = raw_idx * sw
        q_X = raw_q * sw
    idx_sq = (idx_X * idx_X).sum(axis=1)  # (M,)
    nq = q_X.shape[0]
    out = np.zeros((nq, 2), dtype=np.float64)
    CH = 256  # запросов в чанке, чтобы не раздуть матрицу расстояний
    for start in range(0, nq, CH):
        chunk = q_X[start:start + CH]                     # (b, D) взвешенные
        raw_chunk = raw_q[start:start + CH]               # (b, D) сырые (для циклики)
        csq = (chunk * chunk).sum(axis=1)[:, None]        # (b, 1)
        d2 = csq + idx_sq[None, :] - 2.0 * (chunk @ idx_X.T)  # (b, M)
        # циклическая поправка для yaw_diff: свёртку по окружности надо делать на
        # СЫРОМ нормализованном yaw_diff (до весов) — иначе не совпадает с
        # RecAim.dist2. В Java: diff = wrapDegrees((a-b)*180)/180, затем
        # AIM_W*TEMP_W*diff². Евклидов вклад в d2 уже равен ww*eucl_raw² (взвешенный),
        # поэтому добавляем ww*(circ_raw² - eucl_raw²), где eucl_raw/circ_raw — СЫРЫЕ
        # разности, а circ_raw получен свёрткой сырой разности (wrap — нелинеен, веса
        # применяем только к итоговому квадрату, как в Java).
        if circ_cols is not None and ww_arr is not None:
            corr = np.zeros((chunk.shape[0], idx_X.shape[0]), dtype=np.float64)
            for c in circ_cols:
                eucl = raw_chunk[:, c, None] - raw_idx[None, :, c]               # (b, M) сырая
                circ = ((eucl * 180.0 + 180.0) % 360.0 - 180.0) / 180.0         # к (-1,1]
                ww = ww_arr[c].astype(np.float64)
                corr += ww * (circ * circ - eucl * eucl)
            d2 = d2 + corr
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

    # проекция на aim-признаки + взвешенное расстояние (как в игре RecAim):
    # aim-ошибка (yaw_diff/pitch_diff) и aim_center доминируют, сосед выбирается
    # по похожей ошибке прицела, а не по случайным признакам. Веса расстояния на
    # каждый столбец проекции = FEAT_W[f]*TEMP_W[w], повторяют RecAim.dist2
    # (недавние кадры окна весят больше — быстрая реакция на прыжок/flick).
    cols = _aim_cols(window, FEAT_SEL)
    yaw_pos = FEAT_SEL.index(7)
    circ_cols = [w * len(FEAT_SEL) + yaw_pos for w in range(window)]
    temp_w = np.array([0.15 + 0.85 * (w / (window - 1)) for w in range(window)],
                     dtype=np.float32)
    fw = np.tile(np.array(FEAT_W, dtype=np.float32), window)        # FEAT_W[f]
    tw = np.repeat(temp_w, len(FEAT_SEL))                          # TEMP_W[w]
    # доп. буст yaw_diff(поз.1)/pitch_diff(поз.2) на последних RECENT_N кадрах —
    # свежая угловая ошибка должна доминировать над историей (анализ #9). Зеркало
    # RecAim.dist2: ww = FEAT_W[f]*TEMP_W[w]*boost, boost=RECENT_BOOST на свежих
    # yaw/pitch, иначе 1.0.
    recent_mul = np.ones(window * len(FEAT_SEL), dtype=np.float32)
    for w in range(window - RECENT_N, window):
        recent_mul[w * len(FEAT_SEL) + 1] = RECENT_BOOST   # yaw_diff
        recent_mul[w * len(FEAT_SEL) + 2] = RECENT_BOOST   # pitch_diff
    ww = fw * tw * recent_mul
    pred = knn_predict(idx_X[:, cols], idx_Y, q_X[:, cols], k,
                       circ_cols=circ_cols, ww_arr=ww)

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


# Только признаки, реально связанные с наводкой (индексы в FEATURE_DIM=30):
# dist=6, yaw_diff=7, pitch_diff=8, tar_fwd=27, tar_side=28, aim_center=29.
# То же множество использует RecAim.java — иначе ошибка прицела «тонет» среди
# 480 признаков и KNN залипает. См. анализ Srafd.
FEAT_SEL = [6, 7, 8, 27, 28, 29]
# Веса расстояния по aim-признакам: доминируют ошибка прицеливания (yaw_diff,
# pitch_diff) и aim_center — сосед выбирается по ПОХОЖЕЙ ОШИБКЕ ПРИЦЕЛА, а не по
# случайным признакам состояния. Должны совпадать с AIM_W в RecAim.java.
FEAT_W = [0.5, 4.0, 4.0, 1.0, 1.0, 2.0]  # dist, yaw_diff, pitch_diff, tar_fwd, tar_side, aim_center

# Зеркало RecAim.java: на последних RECENT_N кадрах окна yaw_diff (поз.1) и
# pitch_diff (поз.2) получают доп. буст, чтобы свежая угловая ошибка доминировала
# над историей (анализ #9). Умеренный, без фанатизма.
RECENT_N = 4
RECENT_BOOST = 2.5


def _aim_cols(window, feat_sel):
    cols = []
    for w in range(window):
        base = w * config.FEATURE_DIM
        for i in feat_sel:
            cols.append(base + i)
    return cols


def export_index(states, targets, window, out_path, n, seed):
    """Строит индекс для мода: случайная подвыборка окон (M,D) + aim-дельты (M,2),
    пишет кастомный бинарь (little-endian), читаемый RecAim.java."""
    rng = np.random.default_rng(seed)
    X, Y = build_windows(states, targets, window)
    n = min(n, X.shape[0])
    sel = rng.permutation(X.shape[0])[:n]
    cols = _aim_cols(window, FEAT_SEL)
    W = X[sel][:, cols].astype(np.float32)   # (n, 96)
    Yaim = Y[sel].astype(np.float32)
    with open(out_path, "wb") as f:
        f.write(b"PVRI")
        f.write(struct.pack("<ii", W.shape[0], W.shape[1]))
        f.write(W.tobytes())
        f.write(Yaim.tobytes())
    print(f"[export] {out_path}: M={W.shape[0]} D={W.shape[1]}  (okna={X.shape[0]})")


def dump_target_stats(Y):
    """Распределение aim-целей (dyaw, dpitch) по ВСЕМУ датасету. Диагностика
    вырожденности targets (Srafd): std≈0 / unique≈1 означали бы, что KNN не с
    чего учить. Внимание: старый evaluate() при query=1 выдавал std_учит.=0.000
    лишь потому, что оценивал ОДИН сэмпл — это артефакт, не признак константных
    данных."""
    for name, col in [("dyaw", Y[:, 0]), ("dpitch", Y[:, 1])]:
        print(f"[stats:{name}] min={col.min():.4f} max={col.max():.4f} "
              f"mean={col.mean():.4f} std={col.std():.4f} unique={len(np.unique(col))}")
    qs = (0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    print("[stats] quantiles dyaw  :", [round(float(np.quantile(Y[:, 0], q)), 4) for q in qs])
    print("[stats] quantiles dpitch:", [round(float(np.quantile(Y[:, 1], q)), 4) for q in qs])
    print("[stats] first 30 targets (dyaw, dpitch):")
    for row in Y[:30]:
        print(f"    {row[0]:+.4f} {row[1]:+.4f}")


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
    dump_target_stats(Y)
    if X.shape[0] == 0:
        print("пусто — проверь путь к датасету"); return
    if args.export:
        export_index(states, targets, args.window, args.export, args.export_n, args.seed)
        return
    evaluate(X, Y, args.window, args.k, args.index, args.query, args.seed)


if __name__ == "__main__":
    main()
