import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from pvp_bot import serialize
from pvp_bot.evolution import scrub_policy
from sim.arena import make_policy, Agent
from sim import physics as phys

MOD = r"C:\Users\user2\Desktop\ai bot\fabric-example-mod-26.2\run\config\pvpbot\model.bin"
m = serialize.from_binary(MOD)
print("sizes", m.sizes, "acts", m.acts, "params", m.num_params())


def aim_stats(policy, seed=7):
    a = Agent([0, 0, 0], yaw=0, weapon=0, hp=20, armor=10)
    b = Agent([4, 0, 0], yaw=180, weapon=0, hp=20, armor=10)
    aims, dists, hits = [], [], 0
    for t in range(150):
        if not a.alive or not b.alive:
            break
        fa, _ = a.build_feature(b, b.pos - b.prev_pos)
        fb, _ = b.build_feature(a, a.pos - a.prev_pos)
        a.window.append(fa)
        b.window.append(fb)
        aa = policy(a.input_vector())
        ab = scrub_policy(b.input_vector())
        a.apply_action(aa)
        b.apply_action(ab)
        a.pos += a.last_move
        b.pos += b.last_move
        a.prev_pos = a.pos.copy()
        b.prev_pos = b.pos.copy()
        rel = b.pos - a.pos
        d = float(np.linalg.norm(rel)) + 1e-9
        look = phys.look_vector(a.yaw, a.pitch)
        aims.append(float(np.dot(look, rel / d)))
        dists.append(d)
        before = b.hp
        a.deal_damage(b)
        if b.hp < before:
            hits += 1
    aims = np.array(aims)
    dists = np.array(dists)
    # стабильность: дисперсия прицела и средний прицел на «установившемся» участке
    return dict(mean=aims.mean(), std=aims.std(), mn=aims.min(),
                settled=aims[40:].mean(), settled_std=aims[40:].std(),
                dist=dists.mean(), hits=hits)


pa = make_policy(m)
r = aim_stats(pa)
print("MOD model.bin vs scrub:")
print("  mean_aim=%.3f std=%.3f min=%.3f" % (r["mean"], r["std"], r["mn"]))
print("  settled(с 40-го тика) mean=%.3f std=%.3f  -> std мал = нет раскачки" % (r["settled"], r["settled_std"]))
print("  mean_dist=%.2f hits=%d" % (r["dist"], r["hits"]))
