import numpy as np
from pvp_bot import dataset

S, T = dataset.load_npz("data/dataset_rec.npz")
s = np.vstack([x for x in S if len(x)])
t = np.vstack([x for x in T if len(x)])
print("shape s =", s.shape, " t =", t.shape)

def corr(a, b):
    a = np.asarray(a, dtype=np.float64) - np.asarray(a, dtype=np.float64).mean()
    b = np.asarray(b, dtype=np.float64) - np.asarray(b, dtype=np.float64).mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else float("nan")

print("corr(f8=diffPitch, dpitch) =", round(corr(s[:, 8], t[:, 1]), 4))   # ожидается > 0
print("corr(f1=deltaY,     dpitch) =", round(corr(s[:, 1], t[:, 1]), 4))  # ожидается < 0
print("corr(f7=yaw_diff,   dyaw)   =", round(corr(s[:, 7], t[:, 0]), 4))  # ожидается > 0 (yaw ок)
print("mean dpitch =", round(float(t[:, 1].mean()), 4),
      " mean f8 =", round(float(s[:, 8].mean()), 4),
      " mean f1 =", round(float(s[:, 1].mean()), 4))
print("corr(|f8|, f29=aim_center) =", round(corr(np.abs(s[:, 8]), s[:, 29]), 4))  # ожидается < 0
