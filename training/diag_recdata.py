"""Диагностика качества датасета записей (record_*.jsonl -> dataset_rec.npz).

Проверяет, есть ли в данных правильная причинно-следственная связь
«ошибка прицела -> действие человека»:
  - корреляция corr(yawDiff, dyaw) и corr(pitchDiff, dpitch) (~0.9+ = хорошо,
    ~0 = recorder/action подозрительно рассинхронизированы);
  - покрытие больших ошибок (нужны recovery-траектории на все углы);
  - таблица «диапазон yawDiff -> средний dyaw» (знак и рост величины при
    большой ошибке — признак осмысленных данных, а не хаоса).

Использование:
  python diag_recdata.py --data data/dataset_rec.npz
"""
import argparse
import os

import numpy as np

sys_path = os.path.dirname(__file__)
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)
from pvp_bot import dataset, config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "dataset_rec.npz"))
    args = ap.parse_args()

    states, targets = dataset.load_npz(args.data)
    S = np.vstack(states) if len(states) > 1 else states[0]
    T = np.vstack(targets) if len(targets) > 1 else targets[0]
    n = S.shape[0]
    print(f"[diag] тиков={n}  state_dim={S.shape[1]} target_dim={T.shape[1]}")

    yawDiff = S[:, 7] * 180.0
    pitchDiff = S[:, 8] * 90.0
    dyaw = T[:, 0]
    dpitch = T[:, 1]

    def desc(name, x):
        print(f"  {name:10s}: min={x.min():8.2f} max={x.max():8.2f} "
              f"mean={x.mean():7.2f} std={x.std():7.2f}")

    print("[diag] распределения (град):")
    desc("yawDiff", yawDiff)
    desc("pitchDiff", pitchDiff)
    desc("dyaw", dyaw)
    desc("dpitch", dpitch)

    cy = np.corrcoef(yawDiff, dyaw)[0, 1]
    cp = np.corrcoef(pitchDiff, dpitch)[0, 1]
    print(f"[diag] corr(yawDiff, dyaw)   = {cy:.3f}   (>=0.9 — данные осмысленны)")
    print(f"[diag] corr(pitchDiff, dpitch)= {cp:.3f}   (>=0.9 — данные осмысленны)")

    # покрытие больших ошибок (recovery-сценарии)
    print("[diag] покрытие ошибок (доля тиков):")
    for thr in (30, 60, 90, 120):
        print(f"  |yawDiff|>{thr:3d}°: {np.mean(np.abs(yawDiff) > thr):6.2%}   "
              f"|pitchDiff|>{thr:3d}°: {np.mean(np.abs(pitchDiff) > thr):6.2%}")

    # aim_center (f[29]): угловая близость к центру цели (cosine), НЕ попадание в
    # AABB. Настоящий ray-vs-AABB считается только в BotController во время игры.
    centered = S[:, 29] > 0
    print(f"[diag] доля тиков с aim_center>0 (угловая близость к центру): {centered.mean():6.2%}")

    # таблица yawDiff -> средний dyaw
    print("[diag] yawDiff диапазон -> mean dyaw (count):")
    edges = [-180, -120, -60, -30, -10, 0, 10, 30, 60, 120, 180]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (yawDiff >= lo) & (yawDiff < hi)
        if m.sum() == 0:
            print(f"  [{lo:>4},{hi:>4}) :  (нет данных)")
        else:
            print(f"  [{lo:>4},{hi:>4}) : mean_dyaw={dyaw[m].mean():+7.3f}  (n={m.sum()})")

    # таблица pitchDiff -> средний dpitch
    print("[diag] pitchDiff диапазон -> mean dpitch (count):")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pitchDiff >= lo) & (pitchDiff < hi)
        if m.sum() == 0:
            print(f"  [{lo:>4},{hi:>4}) :  (нет данных)")
        else:
            print(f"  [{lo:>4},{hi:>4}) : mean_dpitch={dpitch[m].mean():+7.3f}  (n={m.sum()})")

    # analysis #9: где teacher-действие вырождается в dyaw≈0 при большой ошибке.
    # Абсолютные корзины ошибки; для каждой — число, средний ЗНАК (направление
    # совпадает с ошибкой?), mean|dyaw| (величина поворота) и доля нулевых
    # действий (|dyaw|<0.5°, как в aim_knn eval). Если на больших ошибках
    # zero% велика и mean|dyaw| мала — данные содержат «пустые» recovery-примеры.
    ZERO_EPS = 0.5
    print(f"[diag] |yawDiff| корзина -> n, mean dyaw(знак), mean|dyaw|, zero%(|dyaw|<{ZERO_EPS}):")
    ay = np.abs(yawDiff)
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 60), (60, 90), (90, 120), (120, 181)]:
        m = (ay >= lo) & (ay < hi)
        if m.sum() == 0:
            print(f"  [{lo:>3},{hi:>3}) : (нет данных)")
        else:
            zero = np.mean(np.abs(dyaw[m]) < ZERO_EPS)
            print(f"  [{lo:>3},{hi:>3}) : n={m.sum():5d}  mean_dyaw={dyaw[m].mean():+7.3f}  "
                  f"mean|dyaw|={np.abs(dyaw[m]).mean():6.3f}  zero%={zero:6.2%}")
    print(f"[diag] |pitchDiff| корзина -> n, mean dpitch(знак), mean|dpitch|, zero%(|dpitch|<{ZERO_EPS}):")
    ap = np.abs(pitchDiff)
    for lo, hi in [(0, 10), (10, 20), (20, 40), (40, 60), (60, 90), (90, 120), (120, 181)]:
        m = (ap >= lo) & (ap < hi)
        if m.sum() == 0:
            print(f"  [{lo:>3},{hi:>3}) : (нет данных)")
        else:
            zero = np.mean(np.abs(dpitch[m]) < ZERO_EPS)
            print(f"  [{lo:>3},{hi:>3}) : n={m.sum():5d}  mean_dpitch={dpitch[m].mean():+7.3f}  "
                  f"mean|dpitch|={np.abs(dpitch[m]).mean():6.3f}  zero%={zero:6.2%}")


if __name__ == "__main__":
    main()
