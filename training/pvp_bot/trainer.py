"""Цикл обучения imitation learning: градиентный спуск (Adam) вручную.

train/val сплит, отслеживание переобучения, сохранение лучших весов по val.
"""
import numpy as np

from .mlp import MLP
from .optim import Adam
from .loss import loss_and_grad, evaluate
from . import config


def train(model, X, Y, epochs=40, batch_size=64, lr=0.001,
          val_frac=0.15, seed=0, cont_weight=1.0, bce_weight=1.0, verbose=True,
          sample_weight=None):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n == 0:
        raise ValueError("пустой датасет")
    perm_all = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    val_idx, tr_idx = perm_all[:n_val], perm_all[n_val:]
    Xtr, Ytr = X[tr_idx], Y[tr_idx]
    Xval, Yval = X[val_idx], Y[val_idx]
    if sample_weight is not None:
        sw_all = np.asarray(sample_weight, dtype=np.float64)
        sw_tr = sw_all[tr_idx]
        sw_val = sw_all[val_idx]
    else:
        sw_tr = sw_val = None

    opt = Adam(lr=lr)
    best_val = float("inf")
    best_flat = model.get_flat()
    history = []

    # авто-взвешивание редких классов (атака/блок): pos_weight = neg/pos, без потолка
    # сеть игнорирует редкие положительные примеры и вырождается в «не атакуй».
    pos = Ytr[:, config.LOGIT].sum(axis=0)
    neg = Xtr.shape[0] - pos
    pw2 = np.where(pos > 0, np.clip(neg / np.maximum(pos, 1.0), 1.0, 30.0), 1.0)
    pos_weight = np.ones(config.TARGET_DIM)
    pos_weight[config.LOGIT] = pw2

    for ep in range(epochs):
        order = rng.permutation(Xtr.shape[0])
        ep_loss, count = 0.0, 0
        for s in range(0, Xtr.shape[0], batch_size):
            bi = order[s:s + batch_size]
            xb = Xtr[bi].T                 # (D_in, B)
            yb = Ytr[bi]                   # (B, D_out)
            wb = sw_tr[bi] if sw_tr is not None else None
            out, cache = model.forward(xb) # out: (D_out, B)
            loss, dout = loss_and_grad(out.T, yb, cont_weight, bce_weight, pos_weight, wb)
            gW, gb = model.backward(dout.T, cache)
            opt.step(model, gW, gb)
            ep_loss += loss * len(bi)
            count += len(bi)
        ep_loss /= max(1, count)
        val = evaluate(model.forward(Xval.T)[0].T, Yval, cont_weight, bce_weight, pos_weight, sw_val)
        history.append((ep_loss, val))
        if val < best_val:
            best_val = val
            best_flat = model.get_flat()
        if verbose:
            print(f"epoch {ep + 1:3d}  train={ep_loss:.4f}  val={val:.4f}")

    model.set_flat(best_flat)
    return history
