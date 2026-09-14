================================================================================
 PvP Bot — как сейчас работает ИИ (актуальное состояние)
================================================================================

Клиентский мод на Fabric (MC 1.21.8 / mapping 26.2). Наводка и атака
управляются нейросетью (ручной numpy-MLP) и/или retrieval-аимом (KNN по
записям). Бот ТОЛЬКО целится и бьёт — ходить игроку вручную (коррекция
движения намеренно убрана). Это model-driven: куда смотреть, решает сеть/
датасет, а не скриптовая геометрия.

--------------------------------------------------------------------------------
1. ДВА РЕЖИМА НАВОДКИ
--------------------------------------------------------------------------------
  @bot  — MLP: model.forward(окно) -> [dyaw, dpitch, fwd, strafe, atk, blk]
  @rec  — retrieval: KNN по rec_index.bin, берёт dyaw/dpitch ближайших соседей
          (имитация through retrieval; движение/удар всё равно от сети/стаба)
  Атака/блок в обоих режимах идут от MLP (чтобы не терять логику удара).

--------------------------------------------------------------------------------
2. ПРИЗНАКИ: 30 на тик, окно 16 тиков => ВХОД MLP = 480 чисел
--------------------------------------------------------------------------------
  Индекс  Имя          Смысл
  0..2    dx,dy,dz     относительная позиция цели (нормировано на MAX_DIST=8)
  3..5    vx,vy,vz     скорость цели (нормировано на MAX_SPEED=0.35)
  6       dist         расстояние (clip/MAX_DIST)
  7       yaw_diff     ошибка прицела по yaw /180
  8       pitch_diff   ошибка прицела по pitch /90
  9..10   fwd,str_spd  скорость бота в базисе вперёд/вправо (нормировано)
  11..12  yaw,pitch    абс. yaw/180, pitch/90
  13..14  yaw,ptch_rt  скорость поворота /MAX_YAW_RT
  15      weapon_cd    готовность атаки (0..1)
  16..18  w_sword/w_axe/w_hand  one-hot оружие
  19      crit         в воздухе (крит)
  20..22  g_ground/g_air/g_water  флаги поверхности
  23..24  hp_self/arm_self  HP/броня бота
  25..26  hp_opp/arm_opp    HP/броня цели
  27..28  tar_fwd/tar_side   скорость цели, спроецированная на базис
                              "вперёд/вправо" бота (как rec tarForward/tarSide)
  29      aim_center  1=в центре хитбокса, 0=на краю (cos≈AIM_COS=0.85),
                     -1≈180° от цели. Гладкий сигнал «остаток ошибки»,
                     считается как dot(look, toTarget) -> remap.

  Окно склеивается в один вектор длиной FEATURE_DIM*WINDOW = 30*16 = 480.

--------------------------------------------------------------------------------
3. СЕТЬ
--------------------------------------------------------------------------------
  Слои:  [480, 96, 48, 6],  активации: [relu, relu, linear]
  Выход: dyaw, dpitch, fwd, strafe — MSE;  attack, block — sigmoid + BCE.
  Параметров ~49 600. Веса в бинаре "PVPW" (little-endian float32), архитектура
  в model.json. Специально МАЛЕНЬКАЯ: крупная сеть не учила наводку (вырождалась
  в инвертированный yaw_diff->dyaw). Нормализации слоёв НЕТ — сеть крошечная и
  и так учится; BatchNorm/LayerNorm здесь не нужен и только ломает инференс.

--------------------------------------------------------------------------------
4. ОБУЧЕНИЕ (reward-weighted BC на rec_*.json)
--------------------------------------------------------------------------------
  Учимся ТОЛЬКО на rec_*.json (данные вращения/прицеливания), без слияния с
  record_*.jsonl. Награда (pvp_bot/aim_reward) = 0.5*center + 0.5*track и
  взвешивает MSE на дельтах прицеливания: сеть сильнее учит тики, где учитель
  целился плотно и держал цель. Это imitation learning, а не скрипт.

  Целевые дельты clipped в ±MAX_YAW_RT (22.5°/тик) — поэтому «голый» MLP ведёт
  себя как П-регулятор с остаточной ошибкой ~25–35° (асимптотика, не баг).

--------------------------------------------------------------------------------
5. ИНФЕРЕНС (как бот целится в игре)
--------------------------------------------------------------------------------
  BotController.onTick собирает 30-мерный признак (StateVector), кладёт в скользящее
  окно (480), прогоняет через MLP или retrieval -> dyaw/dpitch. Сеть/retrieval
  лишь СТАВИТ ЦЕЛЬ (tgtYaw/tgtPitch); плавный доворот между кадрами делает
  smoothAimFrame() через Mth.rotLerp (без дёрганья).

  GAIN: AIM_GAIN=2.5 применяется ТОЛЬКО к yaw (out[0]). Это устраняет жалобу
  «не может довернуть» по горизонтали. По тангажу gain НЕ применяется — в
  in-game retrieval (полноразмерное KNN-окно) gain на pitch вызывал уход тангажа
  в насыщение («смотрит вверх»). Тангаж и так отрабатывался без gain.

--------------------------------------------------------------------------------
6. ВОСПРОИЗВЕДЕНИЕ (пошагово)
--------------------------------------------------------------------------------
  cd training
  python convert_rec.py                       # rec_*.json -> data/dataset_rec.npz (30-dim)
  python train_aim_reward.py                  # обучение -> data/model_aim_reward.json/.bin
  python aim_knn.py --export data/rec_index.bin --export-n 15000   # KNN-индекс (480-wide)
  # деплой в мод:
  copy data\model_aim_reward.json ..\fabric-example-mod-26.2\run\config\pvpbot\model.json
  copy data\model_aim_reward.bin   ..\fabric-example-mod-26.2\run\config\pvpbot\model.bin
  copy data\rec_index.bin          ..\fabric-example-mod-26.2\run\config\pvpbot\rec_index.bin
  python selfplay_retrieval.py --matches 100   # headless: два retrieval-бота дерутся, лог в консоль
  # в игре (runClient):  @dummy  -> ставит зомби-манекен
  #                       @bot on / @rec on -> включает наводку
  #                       @start / @stop -> запуск/стоп

================================================================================
7. ОСНОВНОЙ КОД (обучение и воспроизведение) — всё в одном файле
================================================================================

--- 7.1  Признак 30-мер (convert_rec.py): окно строится из rec_*.json ----------
def _rec_tick_to_state(d):
    """Одна запись rec_*.json -> 30-мерный признак."""
    dist = float(d.get("dist", 0.0)); dist2d = float(d.get("dist2d", dist))
    diff_yaw = float(d.get("diffYaw", 0.0)); diff_pitch = float(d.get("diffPitch", 0.0))
    delta_y = float(d.get("deltaY", 0.0)); is_ground = float(d.get("isOnGround", 1.0))
    cd = float(d.get("cooldown", 1.0))
    yr = np.radians(diff_yaw)
    dx = -dist2d * np.sin(yr); dz = dist2d * np.cos(yr)   # восстановление dx/dz
    tvx = float(d.get("tarVelX", 0.0)); tvy = float(d.get("tarVelY", 0.0)); tvz = float(d.get("tarVelZ", 0.0))

    f = np.zeros(30, dtype=np.float64)
    f[0] = dx / MAX_DIST;  f[1] = delta_y / MAX_DIST;  f[2] = dz / MAX_DIST
    f[3] = tvx / MAX_SPEED; f[4] = tvy / MAX_SPEED;     f[5] = tvz / MAX_SPEED
    f[6] = min(dist, MAX_DIST) / MAX_DIST
    f[7] = diff_yaw / 180.0;  f[8] = diff_pitch / 90.0
    # f[9..14] (fwd/str speed, yaw/pitch, их скорости) в rec нет -> 0
    f[15] = cd                                          # готовность атаки
    f[16] = 1.0                                         # sword (по умолчанию)
    f[20] = is_ground;  f[21] = 1.0 - is_ground
    f[23] = 1.0;  f[25] = 1.0                           # HP-плейсхолдеры
    f[27] = float(d.get("tarForward", 0.0))              # ориентация цели
    f[28] = float(d.get("tarSide", 0.0))
    # f[29] aim_center: 1=центр, 0=край хитбокса (cos≈0.85), -1≈180°
    _cos = math.cos(math.radians(diff_yaw)) * math.cos(math.radians(diff_pitch))
    f[29] = max(-1.0, min(1.0, (_cos - 0.85) / (1.0 - 0.85)))
    return f

--- 7.2  Награда за прицеливание (pvp_bot/aim_reward.py) ----------------------
def per_tick_reward_episode(s, t):
    s = np.asarray(s, dtype=np.float64); T = s.shape[0]
    diff_yaw = s[:, 7] * 180.0;  diff_pitch = s[:, 8] * 90.0
    center = np.maximum(0.0, 1.0 - (np.abs(diff_yaw) + np.abs(diff_pitch)) / (2.0 * config.MAX_YAW_RT))
    on = center > 0.5
    streak = np.zeros(T, dtype=np.float64); run = 0.0
    for i in range(T):
        run = run + 1.0 if on[i] else 0.0; streak[i] = run
    track = np.minimum(1.0, streak / 10.0)
    return 0.5 * center + 0.5 * track

def build_windows_weighted(states, targets, window=None):
    window = window or config.WINDOW
    Xs, Ys, Ws = [], [], []
    for s, t in zip(states, targets):
        x, y = dataset.make_windows(s, t, window)
        if x.shape[0] == 0: continue
        r = per_tick_reward_episode(s, t)
        w = r[window - 1: window - 1 + x.shape[0]]
        Xs.append(x); Ys.append(y); Ws.append(w)
    if not Xs:
        return (np.zeros((0, window*config.FEATURE_DIM)), np.zeros((0, config.TARGET_DIM)), np.zeros(0))
    return np.vstack(Xs), np.vstack(Ys), np.concatenate(Ws)

--- 7.3  Обучение (train_aim_reward.py) ---------------------------------------
import numpy as np
from pvp_bot import dataset, trainer, serialize, config
from pvp_bot.aim_reward import build_windows_weighted

S, T = dataset.load_npz("data/dataset_rec.npz")
X, Y, W = build_windows_weighted(S, T, config.WINDOW)
model = trainer.MLP(seed=7)                              # [480,96,48,6] relu/relu/linear
trainer.train(model, X, Y, epochs=60, lr=0.001, verbose=True, sample_weight=W)
serialize.to_json(model, "data/model_aim_reward.json")
serialize.to_binary(model, "data/model_aim_reward.bin")

--- 7.4  Экспорт KNN-индекса (aim_knn.py) ------------------------------------
def export_index(states, targets, window, out_path, n, seed):
    rng = np.random.default_rng(seed)
    X, Y = build_windows(states, targets, window)        # X: (N, 480), Y: (N,2) dyaw/dpitch
    sel = rng.permutation(X.shape[0])[:n]
    W = X[sel].astype(np.float32);  Yaim = Y[sel].astype(np.float32)
    with open(out_path, "wb") as f:
        f.write(b"PVRI")
        f.write(struct.pack("<ii", W.shape[0], W.shape[1]))
        f.write(W.tobytes());  f.write(Yaim.tobytes())
# вызов:  python aim_knn.py --export data/rec_index.bin --export-n 15000

--- 7.5  Retrieval-аим, KNN (RecAim.java, читается модом) --------------------
// окно (float[480]) -> [dyaw, dpitch]: k=5 ближайших, взвешено по расстоянию
public float[] aim(float[] window) {
    float[] bestD = new float[K]; int[] bestI = new int[K];
    for (int k = 0; k < K; k++) { bestD[k] = Float.MAX_VALUE; bestI[k] = -1; }
    for (int i = 0; i < M; i++) {
        float d2 = 0f;
        for (int j = 0; j < D; j++) { float diff = window[j] - W[i][j]; d2 += diff * diff; }
        if (d2 < bestD[K - 1]) {
            bestD[K - 1] = d2; bestI[K - 1] = i;
            for (int t = K - 2; t >= 0; t--) {            // всплываем (сортировка)
                if (bestD[t] > bestD[t + 1]) {
                    float td = bestD[t]; bestD[t] = bestD[t + 1]; bestD[t + 1] = td;
                    int ti = bestI[t]; bestI[t] = bestI[t + 1]; bestI[t + 1] = ti;
                } else break;
            }
        }
    }
    float dyaw = 0f, dpitch = 0f, wsum = 0f;
    for (int k = 0; k < K; k++) {
        if (bestI[k] < 0) continue;
        float w = (float) (1.0 / (Math.sqrt(bestD[k]) + 1e-3));
        dyaw += w * Y[bestI[k]][0]; dpitch += w * Y[bestI[k]][1]; wsum += w;
    }
    if (wsum > 0f) { dyaw /= wsum; dpitch /= wsum; }
    return new float[]{dyaw, dpitch};
}
// M и D читаются из заголовка бинаря -> при 30-dim окне D=480 подхватится автоматически.

--- 7.6  Headless self-play (selfplay_retrieval.py): два retrieval-бота --------
FEAT_SEL = [6, 7, 8]                  # dist, yaw_diff, pitch_diff (согласованы с датасетом)
COLS = [w * config.FEATURE_DIM + i for w in range(config.WINDOW) for i in FEAT_SEL]  # срез окна

class RetrievalController:
    def __init__(self, aim_fn, name="A", gain=2.5): self.aim = aim_fn; self.name = name; self.gain = gain
    def act(self, self_a, opp, window):
        dyaw, dpitch = self.aim(window)
        dyaw   = float(np.clip(dyaw   * self.gain, -config.MAX_YAW_RT, config.MAX_YAW_RT))
        dpitch = float(np.clip(dpitch * self.gain, -config.MAX_YAW_RT, config.MAX_YAW_RT))
        rel = opp.pos - self_a.pos; dist = float(np.linalg.norm(rel)) + 1e-6
        to_xz = np.array([rel[0], rel[2]])
        fy = phys.forward_dir(self_a.yaw); ry = phys.right_dir(self_a.yaw)
        fwd = float(np.clip(np.dot(to_xz, fy) / dist * np.clip((dist - IDEAL_DIST)/1.0, -1, 1), -1, 1))
        strafe = float(np.clip(np.dot(to_xz, ry) / dist, -1, 1)) * 0.3 + 0.3*np.sin(getattr(self_a,"_t",0)*0.2)
        look = phys.look_vector(self_a.yaw, self_a.pitch); to_opp = rel / dist
        aim_cos = float(np.dot(look, to_opp))
        attack = 1.0 if (dist < REACH and aim_cos > phys.AIM_COS_LO) else 0.0
        return [dyaw, dpitch, fwd, strafe, attack, 0.0]
# оба стартуют лицом друг к другу (in-distribution) -> retrieval держит ~25–35°.

--- 7.7  In-game применение + GAIN (BotController.java) ----------------------
// out[0]=dyaw, out[1]=dpitch из MLP или recAim; tgtYaw/tgtPitch — цель.
if (useRecAim && recAim != null) {
    float[] ra = recAim.aim(window);
    out[0] = Mth.clamp(ra[0], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
    out[1] = Mth.clamp(ra[1], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
}
// GAIN только по yaw (устраняет недокрут); по pitch gain НЕТ (иначе "смотрит вверх")
out[0] = out[0] * AIM_GAIN;
tgtYaw   = (float) Mth.wrapDegrees(self.getYRot() + Mth.clamp(out[0], -Config.MAX_YAW_RT, Config.MAX_YAW_RT));
tgtPitch = (float) Mth.clamp(self.getXRot() + Mth.clamp(out[1], -Config.MAX_YAW_RT, Config.MAX_YAW_RT), -90.0F, 90.0F);

--- 7.8  Признак aim_center в живом моде (StateVector.java) -------------------
f[p++] = (float) ((oppVel.x * fX + oppVel.z * fZ) / Config.MAX_SPEED);  // tar_fwd
f[p++] = (float) ((oppVel.x * rX + oppVel.z * rZ) / Config.MAX_SPEED);  // tar_side
Vec3 svLook  = self.getLookAngle();
Vec3 svToOpp = new Vec3(oppX - selfEye.x, oppCenterY - selfEye.y, oppZ - selfEye.z).normalize();
double svAimCos = svLook.dot(svToOpp);
f[p++] = (float) Math.max(-1.0, Math.min(1.0, (svAimCos - Config.AIM_COS) / (1.0 - Config.AIM_COS)));  // aim_center

================================================================================
8. ИЗВЕСТНЫЕ ОГРАНИЧЕНИЯ / ЧТО МОЖНО ДОКРУТИТЬ
================================================================================
  * «Голый» MLP — П-регулятор; остаточная ошибка ~25–35° (порог удара AIM_COS≈31°).
    GAIN по yaw снижает её по горизонтали. Дальше давить gain опасно (раскачка).
  * rec-данные не содержат полной геометрии (нет fwd/str speed, abs yaw/pitch) —
    они восстановлены приближённо; обучается в основном ПРИЦЕЛИВАНИЕ.
  * In-game retrieval использует полноразмерное 480-мерное окно для KNN; self-play —
    только [dist, yaw_diff, pitch_diff]. При желании унифицировать.
  * Нормализация слоёв (BatchNorm/LayerNorm) НЕ добавлялась: не лечит асимптотику
    и ломает инференс (нужно менять и Java Model + формат весов).
================================================================================
