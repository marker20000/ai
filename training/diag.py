"""Диагностика: умеет ли база/чемпион целиться и бить эталон."""
import numpy as np
from pvp_bot.mlp import MLP
from pvp_bot import serialize
from pvp_bot.evolution import scrub_policy
from sim.arena import run, make_policy, Agent
from selfplay import build_base

# средний прицел (dot(look,to_opp)) и дистанция за бой
def aim_stats(policy, seed=7):
    a = Agent([0,0,0], yaw=0, weapon=0, hp=20, armor=10)
    b = Agent([4,0,0], yaw=180, weapon=0, hp=20, armor=10)
    from sim import physics as phys
    aims, dists, hits = [], [], 0
    rng = np.random.default_rng(seed)
    for t in range(150):
        if not a.alive or not b.alive: break
        fa,_ = a.build_feature(b, b.pos-b.prev_pos)
        fb,_ = b.build_feature(a, a.pos-a.prev_pos)
        a.window.append(fa); b.window.append(fb)
        act_a = policy(a.input_vector())
        act_b = scrub_policy(b.input_vector())
        a.apply_action(act_a); b.apply_action(act_b)
        a.pos += a.last_move; b.pos += b.last_move
        a.prev_pos=a.pos.copy(); b.prev_pos=b.pos.copy()
        rel = b.pos - a.pos; d = float(np.linalg.norm(rel))+1e-9
        look = phys.look_vector(a.yaw,a.pitch)
        aims.append(float(np.dot(look, rel/d)))
        dists.append(d)
        before = b.hp
        a.deal_damage(b)
        if b.hp < before: hits += 1
    return np.mean(aims), np.mean(dists), hits

print("=== БАЗА (имитация) ===")
base = build_base(None)
pa = make_policy(base)
# прямая проверка: учит ли МЛП отображать yaw_diff -> dyaw и pitch_diff -> dpitch?
for yv, pv in ((0.5, 0.0), (-0.5, 0.0), (0.3, 0.5), (0.3, -0.5)):
    x = np.zeros(432); x[7] = yv; x[8] = pv
    o = pa(x)
    print(f"  feat[7]={yv:+.2f} feat[8]={pv:+.2f} -> dyaw={o[0]:+.2f} dpitch={o[1]:+.2f}")
m = aim_stats(pa)
print(f"  mean_aim(dot)={m[0]:.3f}  mean_dist={m[1]:.2f}  hits={m[2]}")

print("=== ЧЕМПИОН (итог эволюции) ===")
ch = serialize.from_binary("data/selfplay_champion.bin")
pc = make_policy(ch)
m = aim_stats(pc)
print(f"  mean_aim(dot)={m[0]:.3f}  mean_dist={m[1]:.2f}  hits={m[2]}")
