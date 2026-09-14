"""Headless-арена: бой бот-против-бота без рендера, для оценки весов.

Используется эволюцией и self-play, чтобы отбирать лучшие вариации весов
по метрике «нанесённый урон − полученный урон» + бонус за убийство.

Для уменьшения разрыва «симуляция → реальная игра»:
  - конвенция движения/поворота совпадает с StateVector.java (forward/right dir);
  - признаки 9/10 — знаковые fwd/strafe-скорости (как в клиенте), а не модуль;
  - флаги поверхности (земля/воздух) берутся из реального состояния агента;
  - стартовые условия рандомизированы (оружие, HP, броня, дистанция, высота) —
    бот видит «нестандартные ситуации», а не только один сценарий;
  - при попадании жертва получает knockback (динамика дистанции).
"""
import numpy as np
from collections import deque

from pvp_bot import config
from . import physics as phys

# Награда за КАЖДЫЙ нанесённый удар: поощряет серию попаданий, а не один
# случайный удар (решает жалобу «делает только первый удар»).
HIT_BONUS = 2.0


class Agent:
    def __init__(self, pos, yaw=0.0, weapon=0, hp=20.0, armor=10.0, rng=None):
        self.pos = np.array(pos, dtype=np.float64)
        self.prev_pos = self.pos.copy()
        self.yaw = yaw
        self.pitch = 0.0
        self.prev_yaw = yaw
        self.prev_pitch = 0.0
        self.hp = hp
        self.armor = armor
        self.cd = 1.0
        self.weapon = weapon          # 0 sword, 1 axe, 2 hand
        self.wdmg = phys.WEAPON_DMG[weapon]
        self.airborne = 0
        self.last_move = np.zeros(3)
        self.window = deque(maxlen=config.WINDOW)
        self.dmg_dealt = 0.0
        self.dmg_taken = 0.0
        self.hits = 0.0
        self.alive = True
        self.rng = rng or np.random.default_rng()

    def build_feature(self, opp, opp_vel):
        rel = opp.pos - self.pos
        dist = float(np.linalg.norm(rel)) + 1e-6
        des_yaw, des_pitch = phys.angle_to_target(self.pos, opp.pos)
        yaw_diff = phys.angle_wrap(des_yaw - self.yaw)
        pitch_diff = des_pitch - self.pitch
        yaw_rate = phys.angle_wrap(self.yaw - self.prev_yaw)
        pitch_rate = self.pitch - self.prev_pitch

        airborne = self.airborne > 0
        crit = 1.0 if airborne else 0.0

        # знаковые fwd/strafe-скорости (совпадают с StateVector.java)
        vx, vz = self.last_move[0], self.last_move[2]
        fwd_dir = phys.forward_dir(self.yaw)
        rgt_dir = phys.right_dir(self.yaw)
        fwd_spd = float(np.clip((vx * fwd_dir[0] + vz * fwd_dir[1]) / config.MAX_SPEED, -1.0, 1.0))
        str_spd = float(np.clip((vx * rgt_dir[0] + vz * rgt_dir[1]) / config.MAX_SPEED, -1.0, 1.0))

        g_ground = 0.0 if airborne else 1.0
        g_air = 1.0 if airborne else 0.0

        feat = np.array([
            rel[0] / config.MAX_DIST, rel[1] / config.MAX_DIST, rel[2] / config.MAX_DIST,
            opp_vel[0] / config.MAX_SPEED, opp_vel[1] / config.MAX_SPEED, opp_vel[2] / config.MAX_SPEED,
            min(dist / config.MAX_DIST, 1.0),
            yaw_diff / 180.0, pitch_diff / 90.0,
            fwd_spd, str_spd,
            self.yaw / 180.0, self.pitch / 90.0,
            yaw_rate / config.MAX_YAW_RT, pitch_rate / config.MAX_YAW_RT,
            self.cd,
            1.0 if self.weapon == 0 else 0.0, 1.0 if self.weapon == 1 else 0.0, 1.0 if self.weapon == 2 else 0.0,
            crit, g_ground, g_air, 0.0,
            self.hp / config.MAX_HP, self.armor / config.MAX_ARMOR,
            opp.hp / config.MAX_HP, opp.armor / config.MAX_ARMOR,
            # f[27], f[28]: ориентация цели — скорость цели в базисе "вперёд/вправо" агента
            float(np.clip((opp_vel[0] * fwd_dir[0] + opp_vel[2] * fwd_dir[1]) / config.MAX_SPEED, -1.0, 1.0)),
            float(np.clip((opp_vel[0] * rgt_dir[0] + opp_vel[2] * rgt_dir[1]) / config.MAX_SPEED, -1.0, 1.0)),
            # f[29]: aim_center = 1 в центре хитбокса, 0 на краю (cos=AIM_COS_HIT), -1 ~180° от цели.
            float(np.clip((np.dot(phys.look_vector(self.yaw, self.pitch), rel / dist) - config.AIM_COS_HIT) / (1.0 - config.AIM_COS_HIT), -1.0, 1.0)),
        ], dtype=np.float64)
        return feat, dist

    def input_vector(self):
        pad = config.WINDOW - len(self.window)
        parts = [np.zeros(config.FEATURE_DIM) for _ in range(pad)]
        parts += list(self.window)
        return np.concatenate(parts)

    def apply_action(self, action):
        dyaw, dpitch, fwd, strafe, attack, block = action
        self.prev_yaw, self.prev_pitch = self.yaw, self.pitch
        self.yaw = phys.angle_wrap(self.yaw + phys.clamp_turn(dyaw))
        self.pitch = float(np.clip(self.pitch + phys.clamp_turn(dpitch), -90, 90))
        self.last_move = phys.move_vector(self.yaw, self.pitch, float(np.clip(fwd, -1, 1)), float(np.clip(strafe, -1, 1)))
        self._attack_flag = attack > config.ATTACK_THRESHOLD
        self._block_flag = block > config.BLOCK_THRESHOLD
        # эпизодический прыжок для крита (создаёт воздушные ситуации)
        if self.airborne == 0 and self.rng.random() < 0.02:
            self.airborne = 6
        if self.airborne > 0:
            self.airborne -= 1
        self.cd = min(1.0, self.cd + phys.ATTACK_CD_RECOVER)

    def deal_damage(self, opp):
        if not self._attack_flag or self.cd < 1.0:
            return
        rel = opp.pos - self.pos
        dist = float(np.linalg.norm(rel))
        if dist > phys.REACH:
            return
        # прицеливание: урон только если взгляд направлен на цель.
        # без этого модель никогда не учит наводку (била бы «просто рядом»).
        # Мягкий множитель: полный урон при AIM_COS, спадает до 0 к AIM_COS_LO —
        # даёт нейроэволюции плавный сигнал учить наводку.
        look = phys.look_vector(self.yaw, self.pitch)
        to_opp = rel / (dist + 1e-9)
        aim = float(np.dot(look, to_opp))
        if aim < phys.AIM_COS_LO:
            return
        aim_mult = min(1.0, (aim - phys.AIM_COS_LO) / (phys.AIM_COS - phys.AIM_COS_LO))
        dmg = self.wdmg * aim_mult
        if self.airborne > 0:
            dmg *= phys.CRIT_MULT
        if opp._block_flag:
            dmg *= 0.4
        dmg = max(0.5, dmg - opp.armor * 0.2)
        opp.hp -= dmg
        opp.dmg_taken += dmg
        self.dmg_dealt += dmg
        # бонус за удар масштабируется ТОЧНОСТЬЮ прицела: неточный удар почти не
        # поощряется, поэтому бот обязан держать прицел на цели, а не просто «бить вблизи».
        self.hits += max(0.0, aim_mult)
        self.cd = 0.0
        # knockback — жертва отбрасывается от атакующего (динамика дистанции)
        if dist > 1e-6:
            opp.pos = opp.pos + rel / dist * phys.KNOCKBACK
        if opp.hp <= 0:
            opp.hp = 0.0
            opp.alive = False


def run(policy_a, policy_b, T=240, seed=0, rng=None, fair=False):
    """Один бой. Возвращает словарь с уроном, убийствами и победителем.

    fair=True — у обоих ботов равные HP/броня/оружие и старт в радиусе,
    чтобы отчёт мерил чистое мастерство, а не удачу стартовых условий.
    """
    rng = rng or np.random.default_rng(seed)
    ang = rng.uniform(-np.pi, np.pi)
    if fair:
        hp = 20.0
        armor = 10.0
        weapon = 0
        d0 = rng.uniform(3.0, 5.0)
    else:
        hp = float(rng.uniform(10, 20))
        armor = float(rng.uniform(0, 20))
        weapon = int(rng.integers(0, 3))
        d0 = rng.uniform(2.0, 7.0)
    a = Agent([0.0, 0.0, 0.0],
              yaw=rng.uniform(-180, 180), weapon=weapon,
              hp=hp, armor=armor, rng=rng)
    b = Agent([np.cos(ang) * d0, rng.uniform(0, 1.5), np.sin(ang) * d0],
              yaw=rng.uniform(-180, 180), weapon=weapon,
              hp=hp, armor=armor, rng=rng)

    for t in range(T):
        if not a.alive or not b.alive:
            break
        opp_vel_a = b.pos - b.prev_pos
        opp_vel_b = a.pos - a.prev_pos
        fa, _ = a.build_feature(b, opp_vel_a)
        fb, _ = b.build_feature(a, opp_vel_b)
        a.window.append(fa)
        b.window.append(fb)
        act_a = policy_a(a.input_vector())
        act_b = policy_b(b.input_vector())
        a.apply_action(act_a)
        b.apply_action(act_b)
        a.pos += a.last_move
        b.pos += b.last_move
        a.prev_pos = a.pos.copy()
        b.prev_pos = b.pos.copy()
        a.deal_damage(b)
        b.deal_damage(a)

    kill_a = 1 if (not b.alive and a.alive) else 0
    kill_b = 1 if (not a.alive and b.alive) else 0
    # кто убивает — тому награда (+50). Каждый удар даёт HIT_BONUS — это заставляет
    # эволюцию учить СУСТЕЙН-прицеливание (бить несколько раз подряд), а не один раз.
    score_a = a.dmg_dealt - a.dmg_taken + kill_a * 50.0 + a.hits * HIT_BONUS
    score_b = b.dmg_dealt - b.dmg_taken + kill_b * 50.0 + b.hits * HIT_BONUS
    return {
        "dmg_a": a.dmg_dealt, "dmg_b": b.dmg_dealt,
        "taken_a": a.dmg_taken, "taken_b": b.dmg_taken,
        "score_a": score_a, "score_b": score_b,
        "kill_a": kill_a, "kill_b": kill_b,
        "winner": "a" if score_a > score_b else ("b" if score_b > score_a else "draw"),
    }


def make_policy(model):
    """Обёртка MLP в callable(window) -> действие."""
    def policy(window):
        x = np.asarray(window, dtype=np.float64).reshape(-1, 1)  # (INPUT_DIM, 1)
        out = model.predict(x)  # (D_out, 1)
        return out[:, 0]
    return policy
