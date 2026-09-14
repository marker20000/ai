"""Работа с датасетом: загрузка, скользящее окно, синтетический генератор.

Формат датасета на диске (npz):
  states  : (E, T, FEATURE_DIM) — на каждый тик нормализованный вектор
  targets : (E, T, TARGET_DIM)  — целевые дельты/флаги

Эпизод = от удара до удара (см. ТЗ). Для обучения окно WINDOW последних
тиков склеивается в один входной вектор.
"""
import math

import numpy as np

from . import config

REACH = 3.0          # дистанция удара (блоков) — для синтетики
IDEAL_DIST = 2.0     # дистанция, которую держит "эксперт" (внутри REACH, чтобы бил)
# Усиление наводки при обучении аима. K=0.5 давал «рыхлого» стрелка: сеть держит
# цель в широком конусе и сходится медленно, отсюда в игре «не хватает скорости
# поворота» и «бьёт как попало». K=0.8 учит ПЛОТНУЮ и БЫСТРУЮ наводку (для больших
# ошибок упирается в MAX_YAW_RT, для малых — почти полная поправка за тик), но
# остаётся <1, поэтому цель никогда не «перескакивает» (нет раскачки). Конвенция
# pitch (положительный = ВНИЗ) уже исправлена, так что дрейфа вверх/вниз нет.
AIM_GAIN = 0.8
# Геометрия, как в моде: глаз игрока выше ног, центр хитбокса цели выше её позиции.
EYE_H = 1.62
OPP_CENTER_H = 0.9


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["states"], data["targets"]


def make_windows(states, targets, window=None):
    """Из одного эпизода (T, F) -> (N, window*F) и (N, D)."""
    window = window or config.WINDOW
    T = states.shape[0]
    X, Y = [], []
    for t in range(window - 1, T):
        x = states[t - window + 1:t + 1].reshape(-1)
        X.append(x)
        Y.append(targets[t])
    if not X:
        return np.zeros((0, window * config.FEATURE_DIM)), np.zeros((0, config.TARGET_DIM))
    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)


def dataset_windows(states_all, targets_all, window=None):
    """Из всех эпизодов -> единый обучающий массив."""
    Xs, Ys = [], []
    for s, t in zip(states_all, targets_all):
        x, y = make_windows(s, t, window)
        if x.shape[0]:
            Xs.append(x)
            Ys.append(y)
    if not Xs:
        return np.zeros((0, (window or config.WINDOW) * config.FEATURE_DIM)), np.zeros((0, config.TARGET_DIM))
    return np.vstack(Xs), np.vstack(Ys)


def make_toy_dataset(n_episodes=60, T=80, seed=0):
    """Синтетические эпизоды: эксперт-П-регулятор на СЛУЧАЙНЫХ состояниях.

    ВАЖНО: каждый тик — независимое случайное состояние (позиция врага,
    yaw/pitch бота, дистанция, скорости). Это гарантирует ПОЛНЫЙ охват
    yaw_diff/pitch_diff/dist во всём диапазоне, т.е. сеть учит сам закон
    управления «наведение = функция ошибки прицела».

    Предыдущая версия была closed-loop с экспертом, который мгновенно
    выравнивал бота (turn = yaw_diff за тик): 99% тиков имели yaw_diff≈0,
    и сеть вырождалась в константу out≈0 — бот в игре не целился вообще.
    """
    rng = np.random.default_rng(seed)
    states_list, targets_list = [], []
    MOVE_SPEED = 0.28  # блоков/тик (совпадает с physics.MOVE_SPEED)
    for _ in range(n_episodes):
        S = np.zeros((T, config.FEATURE_DIM))
        Y = np.zeros((T, config.TARGET_DIM))
        for t in range(T):
            # случайное состояние: враг относительно бота
            ang = rng.uniform(-np.pi, np.pi)
            d0 = rng.uniform(1.0, 7.0)
            dy = rng.uniform(-1.0, 2.5)
            dx = np.cos(ang) * d0
            dz = np.sin(ang) * d0
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz)) + 1e-6
            hdist = float(np.sqrt(dx * dx + dz * dz)) + 1e-6
            bot_yaw = rng.uniform(-180, 180)
            bot_pitch = rng.uniform(-60, 60)
            vx = rng.normal(0, 0.12); vy = rng.normal(0, 0.05); vz = rng.normal(0, 0.12)
            # желаемые углы (конвенция MC: положительный pitch = ВНИЗ,
            # поэтому цель выше (dy>0) => desired_pitch<0 — смотрим вверх).
            # Точно совпадает с physics.angle_to_target / StateVector.collect.
            des_yaw = np.degrees(np.arctan2(-dx, dz))
            # вертикаль: от глаз бота (bot.y+EYE_H) к центру хитбокса цели
            # (dy+OPP_CENTER_H) — совпадает с StateVector.collect (глаз vs центр),
            # иначе в игре постоянное смещение прицела. dy здесь = opp.y - bot.y
            # (бот в начале координат), поэтому глаз бота = EYE_H, центр цели = dy+OPP_CENTER_H.
            dy_pitch = (dy + OPP_CENTER_H) - EYE_H
            des_pitch = np.degrees(np.arctan2(-dy_pitch, hdist))
            yaw_diff = (des_yaw - bot_yaw + 180) % 360 - 180
            pitch_diff = des_pitch - bot_pitch
            # эксперт — П-регулятор с усилением AIM_GAIN: учим ПЛОТНУЮ и БЫСТРУЮ
            # наводку на цель. Шум маленький (0.15), чтобы цель держалась вплотную,
            # а не «приблизительно». gain<1 → без перескока/раскачки.
            dyaw = float(np.clip(yaw_diff * AIM_GAIN + rng.normal(0, 0.15), -config.MAX_YAW_RT, config.MAX_YAW_RT))
            dpitch = float(np.clip(pitch_diff * AIM_GAIN + rng.normal(0, 0.15), -config.MAX_YAW_RT, config.MAX_YAW_RT))
            # дистанция: держать IDEAL_DIST (внутри REACH) — подходить к врагу.
            # fwd>0 = вперёд (к врагу после наводки) => при dist>IDEAL_DIST (далеко)
            # fwd>0 => err = dist - IDEAL_DIST (иначе бот убегает).
            err = dist - IDEAL_DIST
            fwd = float(np.clip(err * 0.6 + rng.normal(0, 0.05), -1, 1))
            strafe = float(np.clip(rng.normal(0, 0.2), -1, 1))
            # атака = НАМЕРЕНИЕ бить, когда в радиусе (кулдаун гейтит сам удар).
            attack = 1.0 if (dist < REACH) else 0.0
            block = 1.0 if rng.random() < 0.03 else 0.0

            # базис MC (совпадает с arena/physics): вперёд=(-sin yaw, cos yaw)
            ry = np.radians(bot_yaw)
            fdx, fdz = -np.sin(ry), np.cos(ry)
            rdx, rdz = -np.cos(ry), -np.sin(ry)
            mvlen = float(np.hypot(fwd, strafe))
            if mvlen > 1e-9:
                mvx = (fdx * fwd + rdx * strafe) / mvlen * MOVE_SPEED
                mvz = (fdz * fwd + rdz * strafe) / mvlen * MOVE_SPEED
            else:
                mvx = mvz = 0.0
            fwd_spd = (mvx * fdx + mvz * fdz) / config.MAX_SPEED
            str_spd = (mvx * rdx + mvz * rdz) / config.MAX_SPEED

            wp = rng.integers(0, 3)
            crit = 1.0 if rng.random() < 0.1 else 0.0
            g_ground = 1.0 if crit == 0.0 else 0.0
            g_air = 1.0 - g_ground
            cd = rng.uniform(0, 1.0)

            S[t] = np.array([
                dx / config.MAX_DIST, dy / config.MAX_DIST, dz / config.MAX_DIST,
                vx / config.MAX_SPEED, vy / config.MAX_SPEED, vz / config.MAX_SPEED,
                min(dist / config.MAX_DIST, 1.0),
                yaw_diff / 180.0, pitch_diff / 90.0,
                fwd_spd, str_spd,
                bot_yaw / 180.0, bot_pitch / 90.0,
                float(rng.normal(0, 0.1)), float(rng.normal(0, 0.1)),  # yaw_rate, pitch_rate
                cd,
                1.0 if wp == 0 else 0.0, 1.0 if wp == 1 else 0.0, 1.0 if wp == 2 else 0.0,
                crit, g_ground, g_air, 0.0,
                rng.uniform(10, 20) / config.MAX_HP, rng.uniform(0, 20) / config.MAX_ARMOR,
                rng.uniform(10, 20) / config.MAX_HP, rng.uniform(0, 20) / config.MAX_ARMOR,
                0.0,  # tar_fwd (synthetic: нет данных о движении цели)
                0.0,  # tar_side
                float(np.clip((math.cos(math.radians(yaw_diff)) * math.cos(math.radians(pitch_diff)) - config.AIM_COS_HIT) / (1.0 - config.AIM_COS_HIT), -1.0, 1.0)),  # aim_center
            ])
            Y[t] = [dyaw, dpitch, fwd, strafe, attack, block]
        states_list.append(S)
        targets_list.append(Y)
    return np.array(states_list), np.array(targets_list)
