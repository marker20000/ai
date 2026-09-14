"""Self-play: 6 ботов по парам учатся друг на друге.

Каждое поколение — round-robin из 6 ботов (15 боёв). Счёт = сумма
(урон нанесённый − полученный) по всем матчам. Топ-3 выживают, остальные
3 порождаются скрещиванием/усреднением весов топ-3 + мутация. Лучший бот
сохраняется как чемпион.

Это эволюция поверх imitation-модели (без RL-бэкпропа). Можно стартовать
с обученной модели (--base) или с синтетической затравки.
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from pvp_bot.mlp import MLP
from pvp_bot import dataset, trainer, serialize
from pvp_bot.evolution import mutate
from sim.arena import run, make_policy


def build_base(base_path):
    if base_path and os.path.exists(base_path):
        return serialize.from_json(base_path) if base_path.endswith(".json") else serialize.from_binary(base_path)
    S, T = dataset.make_toy_dataset(n_episodes=200, T=100, seed=1)
    X, Y = dataset.dataset_windows(S, T)
    m = MLP(seed=3)
    # больше эпох/больше lr: малая сеть иначе «вырождается» в константу
    # (не учит тождество out[1]≈pitch_diff), и бот в игре не целится.
    trainer.train(m, X, Y, epochs=300, lr=0.006, verbose=False)
    return m


def selfplay(base_path, generations, matches, T, sigma, out, seed):
    rng = np.random.default_rng(seed)
    base = build_base(base_path)
    sizes, acts = base.sizes, base.acts
    flat0 = base.get_flat()

    # 6 стартовых ботов: база + вариации
    bots = [base.copy()]
    for _ in range(5):
        m = MLP(sizes, acts)
        m.set_flat(mutate(flat0, 0.03, rng))
        bots.append(m)

    best_flat = flat0.copy()
    best_score = float("-inf")

    for gen in range(generations):
        scores = [0.0] * 6
        for i in range(6):
            for j in range(i + 1, 6):
                res = run(make_policy(bots[i]), make_policy(bots[j]), T=T, seed=seed + gen * 31 + i * 7 + j)
                scores[i] += res["score_a"]
                scores[j] += res["score_b"]
        order = sorted(range(6), key=lambda k: -scores[k])
        if scores[order[0]] > best_score:
            best_score = scores[order[0]]
            best_flat = bots[order[0]].get_flat().copy()
        print(f"gen {gen+1:3d}  scores={[round(s,1) for s in scores]}  leader={order[0]}")

        survivors = [bots[order[k]].copy() for k in range(3)]
        new_bots = survivors[:]
        while len(new_bots) < 6:
            a, b = survivors[rng.integers(0, 3)], survivors[rng.integers(0, 3)]
            child = MLP(sizes, acts)
            child.set_flat((a.get_flat() + b.get_flat()) / 2.0 + mutate(np.zeros_like(flat0), sigma, rng))
            new_bots.append(child)
        bots = new_bots

    champ = MLP(sizes, acts)
    champ.set_flat(best_flat)
    serialize.to_json(champ, out)
    bin_out = out
    if bin_out.endswith(".json"):
        bin_out = bin_out[:-5] + ".bin"
    serialize.to_binary(champ, bin_out)
    print(f"[selfplay] чемпион сохранён: {out}")
    print(f"[selfplay] бинарь для мода:  {bin_out}  (лучший счёт {best_score:.1f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=None)
    ap.add_argument("--generations", type=int, default=15)
    ap.add_argument("--matches", type=int, default=1)
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--out", default="data/selfplay_champion.json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    selfplay(args.base, args.generations, args.matches, args.T, args.sigma, args.out, args.seed)


if __name__ == "__main__":
    main()
