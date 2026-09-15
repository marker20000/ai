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

    # внутри хитбокса (aim_center f[29] > 0)
    inside = S[:, 29] > 0
    print(f"[diag] доля тиков «в хитбоксе» (f29>0): {inside.mean():6.2%}")

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


if __name__ == "__main__":
    main()
