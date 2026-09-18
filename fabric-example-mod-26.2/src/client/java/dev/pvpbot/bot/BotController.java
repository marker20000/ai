package dev.pvpbot.bot;

import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import java.util.Random;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.InteractionHand;
import net.minecraft.world.phys.Vec3;
import net.minecraft.world.phys.AABB;
import net.minecraft.util.Mth;

import dev.pvpbot.config.Config;
import dev.pvpbot.net.Model;
import dev.pvpbot.net.RecAim;

/**
 * Инференс сети на каждый client-tick и применение действий к локальному игроку.
 * Скорость поворота/движения клэмпится, чтобы бот не был «снайпером».
 */
public final class BotController {
    private boolean enabled = false;
    private boolean silentMode = false; // @silent: камера не движется локально, только на сервере
    private int leadTicks = 2; // количество тиков для lead compensation (по умолчанию 2)
    private Model model = null;
    private RecAim recAim = null;     // retrieval-индекс (rec_index.bin), опц.
    private boolean useRecAim = false;
    private final float[] window = new float[Config.INPUT_DIM];
    private int count = 0;
    private int dbg = 0;
    // предыдущее состояние для признаков скорости (скорость цели / yaw-rate)
    private Vec3 prevOpp = null;
    private float prevYaw = 0f;
    private float prevPitch = 0f;
    private float lastFwd = 0f;     // применённое (оклэмпенное) движение — для отладки
    private float lastStrafe = 0f;
    private final Random jitter = new Random(); // разброс прицела, чтобы не бить в одну точку
    
    // Серверный yaw/pitch для @silent режима
    private float serverYaw = 0f;
    private float serverPitch = 0f;

    // Плавная наводка: MLP каждый тик ставит ЦЕЛЬ (tgtYaw/tgtPitch), а доворот
    // между тиками делает smoothAimFrame() через Mth.rotLerp — чтобы взгляд не
    // «дёргался» дискретно раз в тик. РЕШЕНИЕ всё равно за сетью, это лишь
    // механическая интерполяция (как сглаживание мыши).
    private float tgtYaw = 0f;
    private float tgtPitch = 0f;
    private boolean aimTarget = false;
    private boolean lastAimInside = false; // флаг «луч в хитбоксе» для дебага
    
    // Плавная межкадровая интерполяция: увеличено с 0.8 до 0.95 для плавности по FPS
    // При 60 FPS (tickDelta ≈ 0.05) это даёт ~0.05 движения за кадр = 20 кадров до цели
    // Чем ближе к 1.0, тем плавнее (но медленнее), чем ближе к 0.5, тем резче (но быстрее)
    private static final float AIM_PER_TICK = 0.95f;
    
    // Уменьшено с 2.5 до 1.8: меньше резких рывков при больших ошибках
    private static final float AIM_GAIN = 1.8f;
    
    // EMA сглаживание для плавной наводки
    private float smoothedDyaw = 0f;
    private float smoothedDpitch = 0f;
    private static final float SMOOTH_ALPHA = 0.4f; // увеличено с 0.35 до 0.4 для более быстрой реакции

    public boolean isEnabled() { return enabled; }
    public boolean isModelLoaded() { return model != null; }
    public boolean isRecAim() { return useRecAim; }
    public boolean isRecAimLoaded() { return recAim != null; }
    public boolean isSilentMode() { return silentMode; }
    public int getLeadTicks() { return leadTicks; }
    
    public float getServerYaw() { return serverYaw; }
    public float getServerPitch() { return serverPitch; }
    
    public void setSilentMode(boolean silent) {
        silentMode = silent;
        System.out.println("[pvpbot] silent mode=" + silentMode);
    }
    
    public void setLeadTicks(int ticks) {
        leadTicks = Math.max(0, Math.min(10, ticks)); // 0-10 тиков
        System.out.println("[pvpbot] lead compensation=" + leadTicks + " ticks");
    }

    public void setRecAim(boolean on, Minecraft client) {
        if (on && recAim == null) {
            try { recAim = RecAim.load(Config.DIR); }
            catch (RuntimeException ignored) {}
        }
        useRecAim = on && recAim != null;
        System.out.println("[pvpbot] recAim=" + useRecAim + " index=" + (recAim != null ? "loaded" : "NULL"));
    }

    public void setEnabled(boolean on, Minecraft client) {
        enabled = on;
        if (on) {
            if (model == null) {
                try { model = Model.load(Config.DIR); }
                catch (RuntimeException e) { model = null; System.out.println("[pvpbot] ошибка загрузки модели: " + e); }
            }
            if (recAim == null) {
                try { recAim = RecAim.load(Config.DIR); }
                catch (RuntimeException ignored) {}
            }
            count = 0; dbg = 0;
            lastFwd = 0f; lastStrafe = 0f;
            smoothedDyaw = 0f; smoothedDpitch = 0f; // сброс сглаживания
            LocalPlayer p = client.player;
            if (p != null) { 
                prevYaw = p.getYRot(); 
                prevPitch = p.getXRot();
                serverYaw = p.getYRot();
                serverPitch = p.getXRot();
            }
            prevOpp = null;
        }
        if (!on && client != null) {
            // отпускаем клавиши движения, чтобы не «залипли» после выключения бота
            client.options.keyUp.setDown(false);
            client.options.keyDown.setDown(false);
            client.options.keyLeft.setDown(false);
            client.options.keyRight.setDown(false);
        }
        System.out.println("[pvpbot] @bot enabled=" + enabled + " model=" + (model != null ? "loaded" : "NULL"));
    }

    public void onTick(Minecraft client) {
        if (!enabled) return;
        LocalPlayer self = client.player;
        if (self == null || client.level == null) return;
        LivingEntity opp = nearestOpponent(client, self);
        if (opp == null) {
            aimTarget = false; // не к чему прицеливаться
            if (dbg++ % 40 == 0) System.out.println("[pvpbot] нет цели (игрок/моб) в радиусе 16 блоков");
            return;
        }

        float yawAtF = self.getYRot();
        float pitchAtF = self.getXRot();
        Vec3 oppAtF = new Vec3(opp.getX(), opp.getY(), opp.getZ());
        
        // Lead compensation: предсказываем позицию цели на N тиков вперёд для компенсации lag
        Vec3 oppVel = opp.getDeltaMovement();
        Vec3 predictedPos = oppAtF.add(oppVel.scale(leadTicks)); // настраиваемое количество тиков
        
        // Используем предсказанную позицию для расчёта state vector
        float[] s = StateVector.collect(self, opp, prevOpp, prevYaw, prevPitch, false);
        
        // Пересчитываем yawDiff и pitchDiff к предсказанной позиции для признаков 7 и 8
        Vec3 selfEye = self.getEyePosition();
        double dx = predictedPos.x - self.getX();
        double dy = (predictedPos.y + opp.getBbHeight() * 0.5) - selfEye.y;
        double dz = predictedPos.z - self.getZ();
        double hdist = Math.sqrt(dx * dx + dz * dz);
        double desiredYaw = Math.toDegrees(Math.atan2(-dx, dz));
        double desiredPitch = Math.toDegrees(Math.atan2(-dy, hdist));
        double yawDiffPredicted = Mth.wrapDegrees(desiredYaw - self.getYRot());
        double pitchDiffPredicted = desiredPitch - self.getXRot();
        
        // Заменяем yawDiff и pitchDiff на предсказанные значения
        s[7] = (float) (yawDiffPredicted / 180.0);
        s[8] = (float) (pitchDiffPredicted / 90.0);
        
        float pitchDiff = StateVector.pitchDiffTo(self, opp);

        // запоминаем состояние ДО применения действия — для признаков скорости на
        // след. тике. Обновляем сразу после collect (до раннего return при
        // заполнении окна): иначе во время warm-up prev остаётся заморожен на
        // значениях момента включения, и tar_fwd/tar_side/yawRate/pitchRate
        // считаются неверно первые 15 тиков.
        prevYaw = yawAtF;
        prevPitch = pitchAtF;
        prevOpp = oppAtF;

        // сдвиговое окно: сдвигаем влево на FEATURE_DIM, кладём новый тик в конец
        if (count < Config.INPUT_DIM) {
            System.arraycopy(s, 0, window, count, Config.FEATURE_DIM);
            count += Config.FEATURE_DIM;
            if (count < Config.INPUT_DIM) return;
        } else {
            System.arraycopy(window, Config.FEATURE_DIM, window, 0, Config.INPUT_DIM - Config.FEATURE_DIM);
            System.arraycopy(s, 0, window, Config.INPUT_DIM - Config.FEATURE_DIM, Config.FEATURE_DIM);
        }

        // Инференс: MLP, либо чистый retrieval-режим (@rec) без загруженного MLP.
        float[] out;
        if (model != null) {
            out = model.forward(window);
        } else if (useRecAim && recAim != null) {
            // без MLP: aim берёт retrieval, атака/блок — простой порог по дистанции
            // (удар всё равно сработает только при совпадении взгляда с хитбоксом, см. apply).
            out = new float[Config.TARGET_DIM];
            out[Config.ATTACK_CH] = (self.distanceToSqr(opp) <= Config.REACH * Config.REACH) ? 1f : 0f;
        } else {
            if (dbg++ % 40 == 0) System.out.println("[pvpbot] @bot ВКЛ, но нет model.bin и recAim — нечего применять");
            return;
        }
        apply(self, client, opp, out, window, pitchDiff);
        if (dbg++ % 20 == 0) {
            String oodInfo = (useRecAim && recAim != null)
                ? String.format("ood=%d knnD=%.3f", recAim.isOod() ? 1 : 0, recAim.getLastDist())
                : "ood=n/a";
            System.out.println(String.format("[pvpbot] цель=%s d=%.1f dyaw=%.3f dpitch=%.3f pitchDiff=%.2f inHit=%d %s atk=%.2f blk=%.2f",
                opp.getName().getString(), Math.sqrt(self.distanceToSqr(opp)), out[0], out[1], pitchDiff,
                lastAimInside ? 1 : 0, oodInfo, out[4], out[5]));
            if (useRecAim && recAim != null) {
                System.out.println("[pvpbot] KNN top-5: " + recAim.getNeighborDebug());
            }
        }
    }

    private void apply(LocalPlayer self, Minecraft client, LivingEntity opp, float[] out, float[] window, float pitchDiff) {
        // --- Наводка ---
        // Обычно дельты (out[0]/out[1]) даёт MLP. В режиме retrieval-аима их берём
        // из ближайшего окна в датасете друга (RecAim.aim) — это imitation через
        // retrieval, а не скриптовая геометрия. Атака/блок (out[4]/out[5]) остаются
        // от MLP, чтобы не терять логику удара. ---
        if (useRecAim && recAim != null) {
            float[] ra = recAim.aim(window);
            float rawDyaw = Mth.clamp(ra[0], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
            float rawDpitch = Mth.clamp(ra[1], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
            
            // Adaptive gain для tracking: при быстром движении цели усиливаем действия
            int lastFrame = (Config.WINDOW - 1) * Config.FEATURE_DIM;
            float tarVelX = window[lastFrame + 3] * Config.MAX_SPEED;
            float tarVelZ = window[lastFrame + 5] * Config.MAX_SPEED;
            float tarSpeed = (float) Math.sqrt(tarVelX * tarVelX + tarVelZ * tarVelZ);
            
            if (tarSpeed > 0.25f) {
                // Boost от 1.0x до 1.5x при скорости 0.25-0.5 блока/тик
                float boost = 1.0f + Math.min(0.5f, tarSpeed * 0.8f);
                rawDyaw *= boost;
                rawDpitch *= boost;
                // Переклэмпить после boost
                rawDyaw = Mth.clamp(rawDyaw, -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
                rawDpitch = Mth.clamp(rawDpitch, -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
            }
            
            // EMA сглаживание для плавности: новое значение = alpha*raw + (1-alpha)*старое
            // Это убирает резкие скачки и делает наводку плавной как у человека
            smoothedDyaw = SMOOTH_ALPHA * rawDyaw + (1f - SMOOTH_ALPHA) * smoothedDyaw;
            smoothedDpitch = SMOOTH_ALPHA * rawDpitch + (1f - SMOOTH_ALPHA) * smoothedDpitch;
            
            // Дополнительное ограничение скорости изменения (rate limit)
            // Максимальное изменение за тик — 80% от MAX_YAW_RT для плавности
            float maxDelta = Config.MAX_YAW_RT * 0.8f;
            out[0] = Mth.clamp(smoothedDyaw, -maxDelta, maxDelta);
            out[1] = Mth.clamp(smoothedDpitch, -maxDelta, maxDelta);
        }

        // deadzone «луч внутри хитбокса»: гасим коррекцию только когда луч взгляда
        // реально пересекает AABB цели (точный ray-vs-AABB, а не cone-тест
        // look·toCenter — на бою 31° конус намного шире хитбокса и включал
        // deadzone даже вне цели, ослабляя наводку). Вне хитбокса коррекция идёт
        // полностью — удержать цель важнее идеального центра, но и дёргаться на
        // цели не надо.
        Vec3 eye = self.getEyePosition();
        Vec3 look = self.getLookAngle().normalize();
        lastAimInside = rayHitsAABB(eye, look, opp.getBoundingBox());
        if (lastAimInside) {
            float damp = 0.15f;
            out[0] *= damp;
            out[1] *= damp;
        }

        // порог удара — прежний мягкий cone-тест (≈31°), он шире строгого AABB.
        Vec3 center = new Vec3(opp.getX(), opp.getY() + opp.getBbHeight() * 0.5, opp.getZ());
        Vec3 toOpp = center.subtract(eye).normalize();
        float aimCos = (float) look.dot(toOpp);

        // усиление выхода прицеливания: модель имитирует маленькие поправки учителя
        // (П-регулятор с остаточной ошибкой ~25-35°). Gain доворачивает до центра;
        // направление всё равно даёт сеть/retrieval — это не скриптовая геометрия.
        // Gain применяем ТОЛЬКО по yaw: по тангажу в in-game retrieval (полноразмерное
        // KNN-окно) он вызывает уход тангажа в насыщение («смотрит вверх»). Тангаж и так
        // отрабатывался без gain в прошлой версии.
        out[0] = out[0] * AIM_GAIN;
        
        // В @silent режиме: обновляем серверный yaw/pitch, но НЕ трогаем локальный self.getYRot/XRot
        // Это позволяет камере оставаться неподвижной для зрителя/записи, но сервер получает пакеты с поворотами
        if (silentMode) {
            // Обновить серверные углы (для отправки пакетов)
            serverYaw = (float) Mth.wrapDegrees(serverYaw + Mth.clamp(out[0], -Config.MAX_YAW_RT, Config.MAX_YAW_RT));
            float pd = Mth.clamp(out[1], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
            if (Math.abs(serverPitch) > 80.0f && Math.signum(pd) == Math.signum(serverPitch)) {
                pd = 0f;
            }
            serverPitch = (float) Mth.clamp(serverPitch + pd, -90.0F, 90.0F);
            
            // Цели для интерполяции остаются серверными (но локальная камера не будет двигаться)
            tgtYaw = serverYaw;
            tgtPitch = serverPitch;
            aimTarget = false; // отключаем smoothAimFrame — локальная камера не движется
        } else {
            // Обычный режим: сеть (или retrieval) лишь СТАВИТ ЦЕЛЬ (tgtYaw/tgtPitch); 
            // сам плавный доворот между кадрами делает smoothAimFrame() (Mth.rotLerp) — без дёрганья.
            tgtYaw = (float) Mth.wrapDegrees(self.getYRot() + Mth.clamp(out[0], -Config.MAX_YAW_RT, Config.MAX_YAW_RT));
            // анти-клинч: гасим систематический drift взгляда в rail (±80°). Если выход
            // сети/retrieval толкает pitch ЕЩЁ дальше в ту же сторону, что и текущий
            // угол — это bias данных, а не цель; не даём заклинить взгляд в потолок/пол.
            float pd = Mth.clamp(out[1], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
            float pitchNow = self.getXRot();
            if (Math.abs(pitchNow) > 80.0f && Math.signum(pd) == Math.signum(pitchNow)) {
                pd = 0f;
            }
            tgtPitch = (float) Mth.clamp(pitchNow + pd, -90.0F, 90.0F);
            aimTarget = true;
        }

        double dist = Math.sqrt(self.distanceToSqr(opp));

        // Движение отключено: бот только целится и бьёт, ходить — вручную.
        // out[2]/out[3] из модели намеренно игнорируем (коррекцию движения убрали).
        lastFwd = 0f; lastStrafe = 0f;

        // --- Атака: сеть хочет (out[4]>0.5), заряд полный, цель в радиусе И взгляд
        //     реально в хитбокс (AIM_COS=0.85 ≈31°). Порог — только «когда бить»,
        //     прицел держит сама сеть, поэтому диких взмахов в сторону нет. ---
        if (out[Config.ATTACK_CH] > 0.5f && self.getAttackStrengthScale(0.0f) >= 1.0f
                && dist <= Config.REACH && opp != null) {
            if (aimCos > Config.AIM_COS) {
                // лёгкий разброс, чтобы не бить всё время в одну точку
                self.setYRot(self.getYRot() + (jitter.nextFloat() - 0.5f) * 2.0f);
                self.setXRot(self.getXRot() + (jitter.nextFloat() - 0.5f) * 2.0f);
                client.gameMode.attack(self, opp);
            }
        }
        if (out[Config.BLOCK_CH] > 0.5f && self.getMainHandItem().getItem() instanceof net.minecraft.world.item.ShieldItem && !self.isUsingItem()) {
            self.startUsingItem(InteractionHand.MAIN_HAND);
        } else if (out[Config.BLOCK_CH] <= 0.5f && self.isUsingItem()) {
            self.stopUsingItem();
        }
    }

    /** Пересекает ли луч (origin + t*dir, t>=0) AABB цели. Собственный slab-тест
     *  (entity AABB в MC axis-aligned). Точный «луч в хитбоксе» вместо cone-теста
     *  look·toCenter, который включал deadzone даже вне цели. */
    private static boolean rayHitsAABB(Vec3 origin, Vec3 dir, AABB box) {
        double tmin = -1e9, tmax = 1e9;
        double[] o = {origin.x, origin.y, origin.z};
        double[] d = {dir.x, dir.y, dir.z};
        double[] lo = {box.minX, box.minY, box.minZ};
        double[] hi = {box.maxX, box.maxY, box.maxZ};
        for (int i = 0; i < 3; i++) {
            if (Math.abs(d[i]) < 1e-8) {
                if (o[i] < lo[i] || o[i] > hi[i]) return false;
            } else {
                double t1 = (lo[i] - o[i]) / d[i];
                double t2 = (hi[i] - o[i]) / d[i];
                if (t1 > t2) { double tmp = t1; t1 = t2; t2 = tmp; }
                tmin = Math.max(tmin, t1);
                tmax = Math.min(tmax, t2);
                if (tmin > tmax) return false;
            }
        }
        return tmax >= 0.0;
    }

    /** Плавный доворот взгляда между тиками (вызывается каждый кадр). Решение о
     *  том, КУДА смотреть, принимает сеть (tgtYaw/tgtPitch); здесь лишь экспоненциальная
     *  интерполяция к цели, независимая от FPS (через tickDelta). yaw — Mth.rotLerp
     *  (корректно обходит ±180°), pitch — Mth.lerp. */
    public void smoothAimFrame(LocalPlayer self, float tickDelta) {
        if (!enabled || !aimTarget || self == null) return;
        float f = 1.0f - (float) Math.pow(1.0f - AIM_PER_TICK, Math.max(0.0f, tickDelta));
        self.setYRot(Mth.rotLerp(f, self.getYRot(), tgtYaw));
        self.setXRot(Mth.lerp(f, self.getXRot(), tgtPitch));
    }

    private static LivingEntity nearestOpponent(Minecraft client, LocalPlayer self) {
        LivingEntity best = null;
        double bestD = 16.0 * 16.0;
        for (Entity e : client.level.entitiesForRendering()) {
            if (!(e instanceof LivingEntity le) || e == self) continue;
            if (!(le instanceof Player) && !(le instanceof Monster)) continue;
            if (!le.isAlive()) continue;
            double d = self.distanceToSqr(le);
            if (d < bestD) { bestD = d; best = le; }
        }
        return best;
    }
}
