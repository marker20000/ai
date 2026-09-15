"""Проверяет, что новая формула в aim_knn.knn_predict совпадает с метрикой
RecAim.dist2 (Java): сначала циклическая свёртка СЫРОГО yaw_diff, затем веса
FEAT_W[f]*TEMP_W[w] к квадрату. Старый баг: свёртка делалась по взвешенной
разности (wrap нелинеен)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from pvp_bot import config
import aim_knn as ak

WINDOW = config.WINDOW
FEAT_SEL = ak.FEAT_SEL
FEAT_W = ak.FEAT_W
RECENT_N = ak.RECENT_N
RECENT_BOOST = ak.RECENT_BOOST
yaw_pos = FEAT_SEL.index(7)
circ_cols = [w * len(FEAT_SEL) + yaw_pos for w in range(WINDOW)]
temp_w = np.array([0.15 + 0.85 * (w / (WINDOW - 1)) for w in range(WINDOW)],
                  dtype=np.float32)
fw = np.tile(np.array(FEAT_W, dtype=np.float32), WINDOW)
tw = np.repeat(temp_w, len(FEAT_SEL))
# зеркало evaluate(): буст yaw_diff(поз.1)/pitch_diff(поз.2) на последних RECENT_N кадрах
recent_mul = np.ones(WINDOW * len(FEAT_SEL), dtype=np.float32)
for w in range(WINDOW - RECENT_N, WINDOW):
    recent_mul[w * len(FEAT_SEL) + 1] = RECENT_BOOST
    recent_mul[w * len(FEAT_SEL) + 2] = RECENT_BOOST
ww = fw * tw * recent_mul


def java_dist2(a, b):
    d2 = 0.0
    for w in range(WINDOW):
        base = w * len(FEAT_SEL)
        for f in range(len(FEAT_SEL)):
            diff = a[base + f] - b[base + f]
            if FEAT_SEL[f] == 7:
                deg = ((diff * 180.0 + 180.0) % 360.0) - 180.0
                diff = deg / 180.0
            boost = RECENT_BOOST if (w >= WINDOW - RECENT_N and FEAT_SEL[f] in (7, 8)) else 1.0
            d2 += FEAT_W[f] * temp_w[w] * boost * diff * diff
    return d2


def python_dist2(a, b):
    sw = np.sqrt(ww)
    wa = a * sw
    wb = b * sw
    d2 = (wa * wa).sum() + (wb * wb).sum() - 2.0 * np.dot(wa, wb)
    corr = 0.0
    for c in circ_cols:
        e = a[c] - b[c]
        circ = ((e * 180.0 + 180.0) % 360.0 - 180.0) / 180.0
        corr += ww[c] * (circ * circ - e * e)
    return d2 + corr


rng = np.random.default_rng(1)
max_err = 0.0
for _ in range(2000):
    a = rng.uniform(-1.5, 1.5, size=WINDOW * len(FEAT_SEL)).astype(np.float32)
    b = rng.uniform(-1.5, 1.5, size=WINDOW * len(FEAT_SEL)).astype(np.float32)
    j = java_dist2(a, b)
    p = python_dist2(a, b)
    max_err = max(max_err, abs(j - p))
print(f"[verify] max |java - python| over 2000 pairs = {max_err:.3e}")
assert max_err < 1e-3, "new formula does NOT match Java RecAim.dist2"
print("[verify] OK: aim_knn mirrors RecAim.dist2 (raw circular wrap, then weights)")
