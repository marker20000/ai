"""Конвертер СТАРЫХ записей data/dat/rec_*.json в формат (27,6) для обучения.

Эти файлы — иной схемы, чем record_*.jsonl модa: {"groups":[[пер-тик словари
diffYaw/deltaYaw/ownForward/canAttack/...]]}. Они БОГАТЫ аим-данными
(deltaYaw/deltaPitch — применённые дельты прицеливания, diffYaw/diffPitch —
ошибка к цели), но не содержат полного 27-мерного StateVector (нет hp/брони,
флагов оружия, абсолютных yaw/pitch и их скоростей). Здесь мы отображаем их
поля на (27,6), подставляя недостающие признаки разумными значениями, и
объединяем с восстановленным dataset.npz (record_*.jsonl).

Обучается в первую очередь ПРИЦЕЛИВАНИЕ (target[0]/[1] = deltaYaw/deltaPitch),
поэтому потеря нескольких признаков состояния некритична: сеть опирается на
реальные diffYaw/diffPitch/dist/tarVel/cooldown/isOnGround.

Использование:
  python convert_rec.py --out data/dataset_all.npz
"""
import argparse
import glob
import json
import math
import os

import numpy as np

# нормализация — как в mod/config и pvp_bot/config
MAX_DIST = 8.0
MAX_SPEED = 0.35
MAX_YAW_RT = 22.5
MAX_HP = 20.0
MAX_ARMOR = 20.0

REC_DIRS = [
    os.path.join("data", "dat", "datasets"),
    os.path.join("data", "dat", "test_datasets"),
]


def _rec_tick_to_state(d):
    """Одна запись rec_*.json -> 27-мерный признак (по docs/feature_spec.md)."""
    dist = float(d.get("dist", 0.0))
    dist2d = float(d.get("dist2d", dist))
    diff_yaw = float(d.get("diffYaw", 0.0))
    diff_pitch = float(d.get("diffPitch", 0.0))
    delta_y = float(d.get("deltaY", 0.0))
    is_ground = float(d.get("isOnGround", 1.0))
    cd = float(d.get("cooldown", 1.0))
    # dx/dz восстанавливаем из dist2d и угла к цели (diffYaw ~ yaw_to_target):
    # StateVector: desiredYaw = atan2(-dx, dz) => dx = -hdist*sin, dz = hdist*cos
    yr = np.radians(diff_yaw)
    dx = -dist2d * np.sin(yr)
    dz = dist2d * np.cos(yr)
    tvx = float(d.get("tarVelX", 0.0))
    tvy = float(d.get("tarVelY", 0.0))
    tvz = float(d.get("tarVelZ", 0.0))

    f = np.zeros(30, dtype=np.float64)
    f[0] = dx / MAX_DIST
    f[1] = delta_y / MAX_DIST
    f[2] = dz / MAX_DIST
    f[3] = tvx / MAX_SPEED
    f[4] = tvy / MAX_SPEED
    f[5] = tvz / MAX_SPEED
    f[6] = min(dist, MAX_DIST) / MAX_DIST
    f[7] = diff_yaw / 180.0
    f[8] = diff_pitch / 90.0
    # f[9], f[10] fwd/strafe speed — без yaw-базиса не восстановить -> 0
    # f[11], f[12] yaw/pitch — отсутствуют -> 0
    # f[13], f[14] yaw/pitch rate — отсутствуют -> 0
    f[15] = cd  # attackStrengthScale ~ cooldown (1 = готов)
    f[16] = 1.0  # sword (PvP по умолчанию; флаг не критичен для аима)
    # f[17] axe=0, f[18] none=0
    # f[19] crit=0
    f[20] = is_ground
    f[21] = 1.0 - is_ground
    # f[22] water=0
    f[23] = 1.0  # self hp placeholder (полное)
    # f[24] self armor=0
    f[25] = 1.0  # opp hp placeholder
    # f[26] opp armor=0
    # f[27], f[28] — ориентация цели (tarForward/tarSide) из rec_*.json
    f[27] = float(d.get("tarForward", 0.0))
    f[28] = float(d.get("tarSide", 0.0))
    # f[29] — aim_center: 1=в центре хитбокса, 0=край (cos≈0.85), -1≈180° от цели.
    # угол между взглядом и целью восстанавливаем из ошибки прицела (diffYaw/diffPitch).
    _dy = float(d.get("diffYaw", 0.0))
    _dp = float(d.get("diffPitch", 0.0))
    _cos = math.cos(math.radians(_dy)) * math.cos(math.radians(_dp))
    f[29] = float(max(-1.0, min(1.0, (_cos - 0.85) / (1.0 - 0.85))))
    return f


def _rec_tick_to_target(d):
    dy = float(d.get("deltaYaw", 0.0))
    dp = float(d.get("deltaPitch", 0.0))
    t = np.zeros(6, dtype=np.float64)
    t[0] = np.clip(dy, -MAX_YAW_RT, MAX_YAW_RT)
    t[1] = np.clip(dp, -MAX_YAW_RT, MAX_YAW_RT)
    t[2] = float(d.get("inputForward", 0.0))
    t[3] = float(d.get("inputSide", 0.0))
    # атака: прокси — canAttack (в радиусе). Бьём в рамках кулдауна сам мод.
    t[4] = float(d.get("canAttack", 0.0))
    # t[5] block=0 (в этих записях нет блока)
    return t


def rec_episodes(src_dirs):
    states, targets = [], []
    for d in src_dirs:
        if not os.path.isdir(d):
            continue
        files = sorted(glob.glob(os.path.join(d, "rec_*.json")))
        for fp in files:
            with open(fp, encoding="utf-8") as fh:
                data = json.load(fh)
            for grp in data.get("groups", []):
                if not grp:
                    continue
                S = np.array([_rec_tick_to_state(x) for x in grp], dtype=np.float64)
                T = np.array([_rec_tick_to_target(x) for x in grp], dtype=np.float64)
                states.append(S)
                targets.append(T)
    return states, targets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("data", "dataset_rec.npz"))
    args = ap.parse_args()

    # Обучаем ТОЛЬКО на rec_*.json (без слияния с record_*.jsonl), чтобы не
    # смешивать источники и не засорять чистые данные приблизительными признаками.
    rec_s, rec_t = rec_episodes(REC_DIRS)
    rec_ticks = sum(s.shape[0] for s in rec_s)
    print(f"[convert_rec] rec_*.json: эпизодов={len(rec_s)} тиков={rec_ticks}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out,
             states=np.array(rec_s, dtype=object),
             targets=np.array(rec_t, dtype=object))
    print(f"[convert_rec] итого эпизодов={len(rec_s)} тиков={rec_ticks} -> {args.out}")


if __name__ == "__main__":
    main()
