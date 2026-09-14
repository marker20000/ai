"""Единый источник правды о размерностях и нормализации.

Должен совпадать с docs/feature_spec.md и mod/bot/StateVector.java.
"""

# --- окно и размерности ---
WINDOW = 16                 # последних тиков на входе
FEATURE_DIM = 30            # признаков на тик (см. feature_spec.md); +2: tarForward/tarSide, +1: aim_center
INPUT_DIM = FEATURE_DIM * WINDOW   # 480
TARGET_DIM = 6             # dyaw, dpitch, fwd, strafe, attack, block

# --- архитектура сети (умолчание) ---
# Сеть намеренно НЕБОЛЬШАЯ: крупная (432->256->128->64->6, ~130k весов) не
# училась прицеливанию на синтетике (вырождалась в слабое/инвертированное
# отображение yaw_diff->dyaw), поэтому бот в игре «не целился». Малая сеть
# быстро и устойчиво учит П-регулятор прицеливания.
LAYER_SIZES = [INPUT_DIM, 96, 48, TARGET_DIM]
ACTIVATIONS = ["relu", "relu", "linear"]

# --- нормализация (совпадает с feature_spec.md) ---
MAX_DIST = 8.0
MAX_SPEED = 0.35
MAX_YAW_RT = 22.5
MAX_HP = 20.0
MAX_ARMOR = 20.0
AIM_COS_HIT = 0.85          # край хитбокса (порог удара в моде): aim_center=0 на этом cos

# --- пороги инференса ---
ATTACK_THRESHOLD = 0.5
BLOCK_THRESHOLD = 0.5

# --- индексы выходных каналов ---
IDX_DYAW, IDX_DPITCH, IDX_FWD, IDX_STRAFE, IDX_ATTACK, IDX_BLOCK = range(6)
CONTINUOUS = [IDX_DYAW, IDX_DPITCH, IDX_FWD, IDX_STRAFE]   # MSE
LOGIT = [IDX_ATTACK, IDX_BLOCK]                            # sigmoid + BCE

# --- индексы входных признаков (для отладки / логов) ---
FEATURE_NAMES = [
    "dx", "dy", "dz", "vx", "vy", "vz", "dist", "yaw_diff", "pitch_diff",
    "fwd_speed", "strafe_speed", "yaw", "pitch", "yaw_rate", "pitch_rate",
    "weapon_cd", "w_sword", "w_axe", "w_hand", "crit",
    "g_ground", "g_air", "g_water", "hp_self", "arm_self", "hp_opp", "arm_opp", "tar_fwd", "tar_side", "aim_center",
]
assert len(FEATURE_NAMES) == FEATURE_DIM
