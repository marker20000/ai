import numpy as np
from pvp_bot import dataset

S, T = dataset.load_npz("data/dataset_rec.npz")
s = np.vstack([x for x in S if len(x)])
t = np.vstack([x for x in T if len(x)])
print("shape s =", s.shape, " t =", t.shape)

print("corr(f8=diffPitch, dpitch) =", round(float(np.corrcoef(s[:, 8], t[:, 1])[0, 1]), 4))   # ожидается > 0
print("corr(f1=deltaY,     dpitch) =", round(float(np.corrcoef(s[:, 1], t[:, 1])[0, 1]), 4))  # ожидается < 0
print("corr(f7=yaw_diff,   dyaw)   =", round(float(np.corrcoef(s[:, 7], t[:, 0])[0, 1]), 4))  # ожидается > 0 (yaw ок)
print("mean dpitch =", round(float(t[:, 1].mean()), 4),
      " mean f8 =", round(float(s[:, 8].mean()), 4),
      " mean f1 =", round(float(s[:, 1].mean()), 4))
print("corr(|f8|, f29=aim_center) =", round(float(np.corrcoef(np.abs(s[:, 8]), s[:, 29])[0, 1]), 4))  # ожидается < 0
