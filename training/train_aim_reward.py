"""Обучение MLP с наградой за прицеливание (на датасетах).

Награда (pvp_bot/aim_reward): близость к центру хитбокса + длительность
удержания прицела. Она взвешивает MSE на дельтах yaw/pitch — сеть сильнее
учит те тики, где учитель целился плотно и держал цель.

Использование:
  python train_aim_reward.py --data data/dataset_rec.npz --out data/model_aim_reward.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pvp_bot import dataset, trainer, serialize, config
from pvp_bot.aim_reward import build_windows_weighted


def quick_eval(model, X, Y):
    P = model.predict(X.T).T
    names = ["dyaw", "dpitch", "fwd", "strafe", "attack", "block"]
    for i, n in enumerate(names):
        if n in ("attack", "block"):
            pred = (P[:, i] > 0.5).astype(int)
            acc = (pred == Y[:, i].round()).mean()
            print(f"  {n}: acc={acc:.3f}")
        else:
            err = np.abs(P[:, i] - Y[:, i])
            print(f"  {n}: mae={err.mean():.3f} corr={np.corrcoef(P[:, i], Y[:, i])[0, 1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "dataset_rec.npz"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", default=os.path.join("data", "model_aim_reward.json"))
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--window", type=int, default=config.WINDOW)
    args = ap.parse_args()

    S, T = dataset.load_npz(args.data)
    X, Y, W = build_windows_weighted(S, T, args.window)
    print(f"[train_aim_reward] X={X.shape} Y={Y.shape} "
          f"W(mean)={W.mean():.3f} W(min..max)={W.min():.3f}..{W.max():.3f}")

    model = trainer.MLP(seed=7)
    trainer.train(model, X, Y, epochs=args.epochs, lr=args.lr, verbose=True, sample_weight=W)
    serialize.to_json(model, args.out)
    serialize.to_binary(model, args.out.replace(".json", ".bin"))
    print(f"[train_aim_reward] сохранено: {args.out}")
    print("[train_aim_reward] офлайн-метрики (на всём датасете):")
    quick_eval(model, X, Y)
    print("OK")


if __name__ == "__main__":
    main()
