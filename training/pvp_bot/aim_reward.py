"""Награда за прицеливание для обучения на датасетах.

Идея пользователя: чем ДОЛЬШЕ ИИ держит прицел на движущемся противнике и чем
БЛИЖЕ к центру хитбокса — тем больше награда. Это model-driven (MLP учится по
награде, не скрипт): награда лишь взвешивает imitation-loss, подчёркивая тики,
где учитель целился плотно и держал цель.

Награда на тик t эпизода:
  center[t] = max(0, 1 - (|diffYaw| + |diffPitch|) / (2*MAX_YAW_RT))
              — близость взгляда к центру (diffYaw/diffPitch — ошибка к цели).
  track[t]  = min(1, streak[t]/10), streak растёт на подряд идущих тиках с center>0.5
              — «долго держит прицел».
  reward[t] = 0.5*center[t] + 0.5*track[t]
"""
import numpy as np

from . import config, dataset


def per_tick_reward_episode(s, t):
    """s: (T,27) норм-состояние, t: (T,6) цель. Возвращает (T,) награду."""
    s = np.asarray(s, dtype=np.float64)
    T = s.shape[0]
    # diffYaw/diffPitch хранятся нормированными: f[7]=diffYaw/180, f[8]=diffPitch/90
    diff_yaw = s[:, 7] * 180.0
    diff_pitch = s[:, 8] * 90.0
    center = np.maximum(0.0, 1.0 - (np.abs(diff_yaw) + np.abs(diff_pitch)) / (2.0 * config.MAX_YAW_RT))
    on = center > 0.5
    streak = np.zeros(T, dtype=np.float64)
    run = 0.0
    for i in range(T):
        run = run + 1.0 if on[i] else 0.0
        streak[i] = run
    track = np.minimum(1.0, streak / 10.0)
    return 0.5 * center + 0.5 * track


def build_windows_weighted(states, targets, window=None):
    """Как dataset.dataset_windows, но ещё возвращает веса выборки = награда
    последнего тика каждого окна (выровнены с X)."""
    window = window or config.WINDOW
    Xs, Ys, Ws = [], [], []
    for s, t in zip(states, targets):
        x, y = dataset.make_windows(s, t, window)
        if x.shape[0] == 0:
            continue
        r = per_tick_reward_episode(s, t)  # (T,)
        # окно k соответствует тику (window-1)+k эпизода
        w = r[window - 1: window - 1 + x.shape[0]]
        Xs.append(x)
        Ys.append(y)
        Ws.append(w)
    if not Xs:
        return (np.zeros((0, window * config.FEATURE_DIM)),
                np.zeros((0, config.TARGET_DIM)),
                np.zeros(0))
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(Ws)
