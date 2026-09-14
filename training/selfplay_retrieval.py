"""Self-play двух агентов на ДАТАСЕТНОМ retrieval-аиме (KNN по rec_*.json).

Пользователь: «на данных дата сетах [противник] просто ротацию, которая ищет
ближайшие значения, и тем самым наводится на игрока» — это retrieval-аим
(ближайшая траектория из датасета), уже применяемый в моде как @rec. Здесь
два таких «игрока» спарятся headless и логируют в консоль.

Что модельно-управляемо (model-driven): НАВОДКА (dyaw/dpitch) берётся
ближайшими соседями по окну состояния в датасете (k=5, взвешенно по
расстоянию). Движение/атака — лёгкая доменная логика (rec-данные содержат
только ротацию: в них нет полного состояния для обучения ходьбы/удара),
поэтому держать дистанцию и бить решает простой подход, а КУДА смотреть —
всегда retrieval.

Запуск:
  python selfplay_retrieval.py --matches 5 --verbose
  python selfplay_retrieval.py --kA 3 --kB 7 --model data/model_aim_reward.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pvp_bot import config
from pvp_bot import serialize
from sim import physics as phys
from sim.arena import Agent, run as arena_run, HIT_BONUS
import convert_rec
import aim_knn

DATASET_NPZ = os.path.join("data", "dataset_rec.npz")
REACH = phys.REACH
IDEAL_DIST = 2.0


def load_or_build_dataset():
    """Берём dataset_rec.npz, при отсутствии пересобираем из rec_*.json."""
    if not os.path.exists(DATASET_NPZ):
        print(f"[selfplay_retrieval] {DATASET_NPZ} нет — конвертирую rec_*.json...")
        s, t = convert_rec.rec_episodes(convert_rec.REC_DIRS)
        os.makedirs(os.path.dirname(DATASET_NPZ) or ".", exist_ok=True)
        np.savez(DATASET_NPZ, states=np.array(s, dtype=object), targets=np.array(t, dtype=object))
        print(f"[selfplay_retrieval] эпизодов={len(s)} -> {DATASET_NPZ}")
    data = np.load(DATASET_NPZ, allow_pickle=True)
    return data["states"], data["targets"]


# Признаки, по которым ищем ближайшего соседа. В rec_*.json позиция
# (dx/dz) ВОССТАНОВЛЕНА приближённо, поэтому полное 432-мерное окно даёт
# плохих соседей. dist/yaw_diff/pitch_diff (индексы 6/7/8) присутствуют и
# согласованы между симуляцией и датасетом — по ним retrieval повторяет
# поворот учителя (deltaYaw/deltaPitch) корректно.
FEAT_SEL = [6, 7, 8]


def _sel_cols():
    cols = []
    for w in range(config.WINDOW):
        base = w * config.FEATURE_DIM
        for i in FEAT_SEL:
            cols.append(base + i)
    return cols


COLS = _sel_cols()


def build_retrieval_aim(k):
    """KNN-аим по датасету: окно (432) -> (dyaw, dpitch)."""
    states, targets = load_or_build_dataset()
    X, Y = aim_knn.build_windows(states, targets, config.WINDOW)
    Xs = X[:, COLS].astype(np.float32)
    Xsq = (Xs * Xs).sum(axis=1)

    def aim(window):
        q = np.asarray(window, dtype=np.float32).reshape(1, -1)[:, COLS]
        csq = (q * q).sum(axis=1)[:, None]
        d2 = csq + Xsq[None, :] - 2.0 * (q @ Xs.T)
        np.maximum(d2, 0.0, out=d2)
        kk = min(k, d2.shape[1])
        part = np.argpartition(d2, kk - 1, axis=1)[:, :kk]
        rows = np.arange(1)[:, None]
        best = d2[rows, part]
        w = 1.0 / (np.sqrt(best) + 1e-6)
        w = w / w.sum(axis=1, keepdims=True)
        neigh = Y[part]
        return (neigh * w[:, :, None]).sum(axis=1)[0]
    return aim, X.shape[0]


class RetrievalController:
    """Наводка — retrieval; движение/удар — подход чтобы бой был осмысленным.

    gain — усиление поворота (как AIM_GAIN в обучении): записанные поправки
    учителя маленькие (несколько градусов/тик), поэтому чистый retrieval
    отстаёт от движущейся цели; gain>1 делает наводку отзывчивой, НАПРАВЛЕНИЕ
    всё равно берётся из ближайших соседей датасета (model-driven).
    """

    def __init__(self, aim_fn, name="A", gain=2.5):
        self.aim = aim_fn
        self.name = name
        self.gain = gain

    def act(self, self_a, opp, window):
        dyaw, dpitch = self.aim(window)
        dyaw = float(np.clip(dyaw * self.gain, -config.MAX_YAW_RT, config.MAX_YAW_RT))
        dpitch = float(np.clip(dpitch * self.gain, -config.MAX_YAW_RT, config.MAX_YAW_RT))
        # --- движение: чисто сим-скаффолд, чтобы бой был осмысленным. ---
        # rec-данные содержат ТОЛЬКО ротацию (ходьба/удар не записаны), поэтому
        # держать дистанцию решает простая геометрия, а КУДА смотреть — retrieval.
        # Проекция вектора «к противнику» на собственный базис агента (forward/
        # right) даёт fwd/strafe, которые ведут к цели независимо от текущего yaw.
        rel = opp.pos - self_a.pos
        dist = float(np.linalg.norm(rel)) + 1e-6
        to_xz = np.array([rel[0], rel[2]])
        fy = phys.forward_dir(self_a.yaw)
        ry = phys.right_dir(self_a.yaw)
        fwd = float(np.clip(np.dot(to_xz, fy) / dist * np.clip((dist - IDEAL_DIST) / 1.0, -1.0, 1.0), -1.0, 1.0))
        strafe = float(np.clip(np.dot(to_xz, ry) / dist, -1.0, 1.0)) * 0.3 + 0.3 * np.sin(getattr(self_a, "_t", 0) * 0.2)
        look = phys.look_vector(self_a.yaw, self_a.pitch)
        to_opp = rel / dist
        aim_cos = float(np.dot(look, to_opp))
        attack = 1.0 if (dist < REACH and aim_cos > phys.AIM_COS_LO) else 0.0
        return [dyaw, dpitch, fwd, strafe, attack, 0.0]


def match(ctrl_a, ctrl_b, T=240, seed=0, verbose=False):
    """Один бой двух контроллеров. Возвращает сводку + (опц.) лог тиков."""
    rng = np.random.default_rng(seed)
    hp = 20.0
    armor = 10.0
    weapon = 0
    ang = rng.uniform(-np.pi, np.pi)
    d0 = rng.uniform(3.0, 5.0)
    a = Agent([0.0, 0.0, 0.0], yaw=rng.uniform(-180, 180), weapon=weapon, hp=hp, armor=armor, rng=rng)
    b = Agent([np.cos(ang) * d0, rng.uniform(0, 1.5), np.sin(ang) * d0], yaw=rng.uniform(-180, 180),
              weapon=weapon, hp=hp, armor=armor, rng=rng)

    # Старт спарринга «в дистрибуции» retrieval: оба смотрят друг на друга
    # (реальный бой — противник перед тобой, а не на 150° сзади). rec-данные
    # почти не содержат тиков с большой ошибкой прицела, поэтому retrieval не
    # умеет «выправляться» с нуля — наводка держится, пока ты УЖЕ целишься.
    ay, ap = phys.angle_to_target(a.pos, b.pos)
    by, bp = phys.angle_to_target(b.pos, a.pos)
    a.yaw, a.pitch, a.prev_yaw, a.prev_pitch = ay, ap, ay, ap
    b.yaw, b.pitch, b.prev_yaw, b.prev_pitch = by, bp, by, bp

    log = []
    sum_err_a = sum_err_b = 0.0
    n_err = 0
    for t in range(T):
        a._t = b._t = t
        if not a.alive or not b.alive:
            break
        opp_vel_a = b.pos - b.prev_pos
        opp_vel_b = a.pos - a.prev_pos
        fa, _ = a.build_feature(b, opp_vel_a)
        fb, _ = b.build_feature(a, opp_vel_b)
        a.window.append(fa)
        b.window.append(fb)
        act_a = ctrl_a.act(a, b, a.input_vector())
        act_b = ctrl_b.act(b, a, b.input_vector())
        a.apply_action(act_a)
        b.apply_action(act_b)
        a.pos += a.last_move
        b.pos += b.last_move
        a.prev_pos = a.pos.copy()
        b.prev_pos = b.pos.copy()
        a.deal_damage(b)
        b.deal_damage(a)

        rel_a = b.pos - a.pos
        rel_b = a.pos - b.pos
        dist_a = float(np.linalg.norm(rel_a)) + 1e-6
        dist_b = float(np.linalg.norm(rel_b)) + 1e-6
        look_a = phys.look_vector(a.yaw, a.pitch)
        look_b = phys.look_vector(b.yaw, b.pitch)
        err_a = float(np.degrees(np.arccos(np.clip(np.dot(look_a, rel_a / dist_a), -1, 1))))
        err_b = float(np.degrees(np.arccos(np.clip(np.dot(look_b, rel_b / dist_b), -1, 1))))
        sum_err_a += err_a
        sum_err_b += err_b
        n_err += 1
        if verbose:
            log.append(f"[t={t:04d}] A d={dist_a:4.2f} aimErr={err_a:5.1f}° hit={'1' if a._attack_flag else '0'} "
                       f"| B d={dist_b:4.2f} aimErr={err_b:5.1f}° hit={'1' if b._attack_flag else '0'} "
                       f"| dmgA={a.dmg_dealt:5.1f} dmgB={b.dmg_dealt:5.1f}")

    kill_a = 1 if (not b.alive and a.alive) else 0
    kill_b = 1 if (not a.alive and b.alive) else 0
    score_a = a.dmg_dealt - a.dmg_taken + kill_a * 50.0 + a.hits * HIT_BONUS
    score_b = b.dmg_dealt - b.dmg_taken + kill_b * 50.0 + b.hits * HIT_BONUS
    return {
        "dmg_a": a.dmg_dealt, "dmg_b": b.dmg_dealt,
        "taken_a": a.dmg_taken, "taken_b": b.dmg_taken,
        "score_a": score_a, "score_b": score_b,
        "kill_a": kill_a, "kill_b": kill_b,
        "winner": "a" if score_a > score_b else ("b" if score_b > score_a else "draw"),
        "ticks": t + 1,
        "aim_a": sum_err_a / max(1, n_err),
        "aim_b": sum_err_b / max(1, n_err),
    }, log


def mlp_controller(path, name="MLP"):
    """Контроллер из обученной MLP-модели (для сравнения с retrieval)."""
    model = serialize.from_json(path) if path.endswith(".json") else serialize.from_binary(path)
    from sim.arena import make_policy
    pol = make_policy(model)

    class C:
        def __init__(self, n): self.name = n
        def act(self, self_a, opp, window):
            self_a._t = getattr(self_a, "_t", 0)
            return list(pol(window))
    return C(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kA", type=int, default=5, help="k для retrieval-агента A")
    ap.add_argument("--kB", type=int, default=5, help="k для retrieval-агента B")
    ap.add_argument("--matches", type=int, default=5)
    ap.add_argument("--T", type=int, default=240)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--verbose", action="store_true", help="логировать каждый тик матча 1")
    ap.add_argument("--model", default=None, help="путь к MLP-модели: B будет MLP (сравнение)")
    args = ap.parse_args()

    aim_a, n = build_retrieval_aim(args.kA)
    print(f"[selfplay_retrieval] индекс датасета: окон={n} (rec_*.json), kA={args.kA}")
    ca = RetrievalController(aim_a, name="A")

    if args.model:
        cb = mlp_controller(args.model, name="MLP")
        print(f"[selfplay_retrieval] агент B = MLP ({args.model})")
    else:
        aim_b, _ = build_retrieval_aim(args.kB)
        cb = RetrievalController(aim_b, name="B")
        print(f"[selfplay_retrieval] агент B = retrieval k={args.kB}")

    wins_a = wins_b = draws = 0
    sum_dmg_a = sum_dmg_b = 0.0
    for m in range(args.matches):
        res, log = match(ca, cb, T=args.T, seed=args.seed + m, verbose=(args.verbose and m == 0))
        if args.verbose and m == 0:
            print("——— пер-тик лог матча 1 ———")
            for line in log:
                print(line)
            print("———————————————")
        if res["winner"] == "a":
            wins_a += 1
        elif res["winner"] == "b":
            wins_b += 1
        else:
            draws += 1
        sum_dmg_a += res["dmg_a"]
        sum_dmg_b += res["dmg_b"]
        print(f"MATCH {m + 1:2d}  winner={res['winner']:>4s}  "
              f"dmgA={res['dmg_a']:6.1f} dmgB={res['dmg_b']:6.1f}  "
              f"takenA={res['taken_a']:6.1f} takenB={res['taken_b']:6.1f}  "
              f"aimErr A={res['aim_a']:5.1f}° B={res['aim_b']:5.1f}°  ticks={res['ticks']}")

    print("——— итог self-play ———")
    print(f"победы: A={wins_a}  B={wins_b}  ничьи={draws}  (всего {args.matches})")
    print(f"средний урон: A={sum_dmg_a / args.matches:6.1f}  B={sum_dmg_b / args.matches:6.1f}")


if __name__ == "__main__":
    main()
