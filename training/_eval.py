import numpy as np
from pvp_bot import dataset as D, serialize

S, T = D.load_npz("data/dataset.npz")
X, Y = D.dataset_windows(S, T)
print("windows:", X.shape, "targets:", Y.shape)

m = serialize.from_json("data/model.json")
P = m.predict(X.T).T  # (N, 6)

names = ["dyaw", "dpitch", "fwd", "strafe", "attack", "block"]
for i, n in enumerate(names):
    if n in ("attack", "block"):
        pred = (P[:, i] > 0.5).astype(int)
        acc = (pred == Y[:, i].round()).mean()
        print(f"  {n}: acc={acc:.3f}  (pred_pos={int(pred.sum())} true_pos={int(Y[:,i].sum())})")
    else:
        err = np.abs(P[:, i] - Y[:, i])
        print(f"  {n}: mae={err.mean():.3f}  std={err.std():.3f}  corr={np.corrcoef(P[:,i], Y[:,i])[0,1]:.3f}")

# Aim quality: does predicted (dyaw,dpitch) reduce aim error? Compare to a "do nothing" baseline.
# Reconstruct per-tick aim: predicted delta vs actual delta. Higher cosine sim = better.
cos = np.zeros(len(Y))
for k in range(len(Y)):
    a = np.array([Y[k,0], Y[k,1]]); p = np.array([P[k,0], P[k,1]])
    na, np_ = np.linalg.norm(a), np.linalg.norm(p)
    cos[k] = (a @ p) / (na*np_ + 1e-9)
print(f"\naim direction cosine (pred vs human delta): mean={cos.mean():.3f} std={cos.std():.3f}")
print("OK")
