"""Эволюционная дошлифовка: мутации весов + отбор по арене.

Без RL-бэкпропа через reward. Несколько вариаций весов сражаются в
headless-симуляции, выживают лучшие по метрике «урон нанесённый − полученный».
"""
import numpy as np

from .mlp import MLP
from sim.arena import run, make_policy

# Простой эвристический противник («гринда»): держит дистанцию и бьёт вблизи.
# Читает последний тик из окна (локальные индексы признаков совпадают с feature_spec).
IDEAL_DIST_NORM = 3.5 / 8.0
REACH_NORM = 3.0 / 8.0


def scrub_policy(window):
    f = window[-27:]  # последний тик
    yaw_diff = f[7] * 180.0
    pitch_diff = f[8] * 90.0
    dist_norm = f[6]
    cd = f[15]
    dyaw = float(np.clip(yaw_diff * 0.2, -22.5, 22.5))
    dpitch = float(np.clip(pitch_diff * 0.2, -22.5, 22.5))
    fwd = 0.6 if dist_norm > IDEAL_DIST_NORM else -0.3
    strafe = 0.2 if (int(dist_norm * 1000) % 2) else -0.2
    attack = 1.0 if (dist_norm < REACH_NORM and cd >= 1.0) else 0.0
    return np.array([dyaw, dpitch, fwd, strafe, attack, 0.0])


def evaluate_weights(flat, sizes, acts, matches=3, T=200, seed=0):
    m = MLP(sizes, acts)
    m.set_flat(flat)
    policy = make_policy(m)
    total = 0.0
    for i in range(matches):
        res = run(policy, scrub_policy, T=T, seed=seed + i * 101)
        total += res["score_a"]
    return total / matches


def mutate(flat, sigma, rng):
    return flat + rng.normal(0.0, sigma, size=flat.shape)


def evolve(base_model, generations=20, pop=12, sigma=0.02, matches=3,
           T=200, seed=0, elite=2, verbose=True):
    rng = np.random.default_rng(seed)
    sizes, acts = base_model.sizes, base_model.acts
    base = base_model.get_flat()

    pop_flats = [base.copy()]
    for _ in range(pop - 1):
        pop_flats.append(mutate(base, sigma, rng))

    best_flat = base.copy()
    best_score = float("-inf")

    for gen in range(generations):
        scored = []
        for flat in pop_flats:
            h = int(abs(np.sum(flat[:8])) * 1e6) % 100000
            s = evaluate_weights(flat, sizes, acts, matches, T, seed=seed + gen * 7 + h)
            scored.append((s, flat))
        scored.sort(key=lambda x: -x[0])
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_flat = scored[0][1].copy()
        if verbose:
            print(f"gen {gen+1:3d}  best={scored[0][0]:+.2f}  mean={np.mean([x[0] for x in scored]):+.2f}")

        new_pop = [scored[i][1].copy() for i in range(min(elite, len(scored)))]
        half = max(1, len(scored) // 2)
        while len(new_pop) < pop:
            p1 = scored[rng.integers(0, half)][1]
            p2 = scored[rng.integers(0, half)][1]
            child = (p1 + p2) / 2.0 + mutate(np.zeros_like(base), sigma, rng)
            new_pop.append(child)
        pop_flats = new_pop

    champ = MLP(sizes, acts)
    champ.set_flat(best_flat)
    return champ
