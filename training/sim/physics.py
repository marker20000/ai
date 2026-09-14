"""Лёгкая кинематика для headless-симуляции боя (без рендера Minecraft).

Это НЕ полная физика игры — упрощённая модель для оценки весов бота в
эволюции/self-play. Координаты — блоки, время — тики.
"""
import numpy as np

from pvp_bot import config

REACH = 3.0          # дистанция удара
MOVE_SPEED = 0.28    # блоков/тик при fwd=1 (бег)
SWORD_DMG = 6.0
AXE_DMG = 9.0
HAND_DMG = 1.0
WEAPON_DMG = [SWORD_DMG, AXE_DMG, HAND_DMG]
CRIT_MULT = 1.5
ATTACK_CD_RECOVER = 1 / 16.0   # кулдаун удара ~16 тиков
KNOCKBACK = 0.6      # сила отбрасывания при попадании (блоков)


def angle_wrap(a):
    return (a + 180.0) % 360.0 - 180.0


def angle_to_target(src, dst):
    dx, dy, dz = dst[0] - src[0], dst[1] - src[1], dst[2] - src[2]
    yaw = np.degrees(np.arctan2(-dx, dz))
    # ВАЖНО: в реальном MC положительный pitch = ВНИЗ (getLookAngle: y = -sin(pitch)),
    # поэтому тут -atan2(dy,..): цель выше (dy>0) => желаемый pitch < 0 (смотрим вверх).
    # Симуляция обязана совпадать с MC, иначе модель учит инвертированный наклон и
    # в игре «плывёт» вверх/вниз.
    pitch = np.degrees(np.arctan2(-dy, np.hypot(dx, dz)))
    return yaw, pitch


def forward_dir(yaw):
    """Направление «вперёд» в (x,z) — совпадает с MC: (-sin yaw, cos yaw).

    Важно: это должно совпадать с тем, куда реально двигает клавиша «вперёд» в
    Minecraft, иначе обученный «идти к врагу» (fwd>0) в игре даст движение ОТ врага.
    """
    ry = np.radians(yaw)
    return np.array([-np.sin(ry), np.cos(ry)])


def right_dir(yaw):
    """Направление «вправо» в (x,z) — MC-базис, перпендикулярен forward_dir."""
    ry = np.radians(yaw)
    return np.array([-np.cos(ry), -np.sin(ry)])


# порог прицеливания: косинус угла отклонения взгляда от цели.
# AIM_COS — полный урон (≈20°, совпадает с жёстким гейтом в моде),
# AIM_COS_LO — ниже этого урона нет вообще (≈66°): довольно свободно, чтобы
# нейроэволюция сразу получала сигнал «лицом к врагу → урон» и учила наводку
# постепенно, а не упиралась в ноль.
AIM_COS = 0.94
AIM_COS_LO = 0.4


def look_vector(yaw, pitch):
    """Единичный вектор «взгляда» агента (инверсия angle_to_target).

    При (yaw,pitch) == angle_to_target(src,dst) вектор указывает точно в dst.
    Используется в арене, чтобы требовать от бота РЕАЛЬНОГО прицеливания —
    иначе модель не получает сигнал учить наводку.
    """
    ry = np.radians(yaw)
    rp = np.radians(pitch)
    cp = np.cos(rp)
    return np.array([
        -np.sin(ry) * cp,
        -np.sin(rp),   # MC: look.y = -sin(pitch); положительный pitch = ВНИЗ
         np.cos(ry) * cp,
    ])


def move_vector(yaw, pitch, fwd, strafe):
    """Горизонтальное перемещение по yaw (pitch не влияет на высоту в упрощении).

    Конвенция совпадает с клиентским модом: fwd вдоль forward_dir, strafe вдоль
    right_dir, нормировка по L2 (как в MC, без ускорения по диагонали).
    """
    fwd = float(np.clip(fwd, -1.0, 1.0))
    strafe = float(np.clip(strafe, -1.0, 1.0))
    mv = forward_dir(yaw) * fwd + right_dir(yaw) * strafe
    n = float(np.linalg.norm(mv))
    if n > 1e-9:
        mv = mv / n * MOVE_SPEED
    return np.array([mv[0], 0.0, mv[1]])


def clamp_turn(delta):
    return float(np.clip(delta, -config.MAX_YAW_RT, config.MAX_YAW_RT))
