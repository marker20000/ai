import numpy as np
d = np.load("data/dataset.npz", allow_pickle=True)
states, targets = d["states"], d["targets"]
print("episodes:", len(states))
for i in range(len(states)):
    S, T = states[i], targets[i]
    assert S.shape == (S.shape[0], 27), S.shape
    assert T.shape == (T.shape[0], 6), T.shape
    print(f"ep {i}: S={S.shape} T={T.shape}")
    print("  state min/max:", round(float(np.nanmin(S)),3), round(float(np.nanmax(S)),3))
    print("  target min/max per dim:", np.round(np.nanmin(T, axis=0),2), np.round(np.nanmax(T, axis=0),2))
    print("  state |x|>2 frac:", round(float(np.mean(np.abs(S) > 2.0)),4))
    print("  yaw out:", int(((T[:,0] < -22.5) | (T[:,0] > 22.5)).sum()),
          " pitch out:", int(((T[:,1] < -22.5) | (T[:,1] > 22.5)).sum()))
    print("  attack:", np.unique(T[:,4]), " block:", np.unique(T[:,5]))
    print("  fwd:", np.unique(T[:,2])[:8], " strafe:", np.unique(T[:,3])[:8])
print("\nfirst state ep0:", np.round(states[0][0], 4))
print("first target ep0:", np.round(targets[0][0], 4))
print("OK")
