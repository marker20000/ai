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

    // Плавная наводка: MLP каждый тик ставит ЦЕЛЬ (tgtYaw/tgtPitch), а доворот
    // между тиками делает smoothAimFrame() через Mth.rotLerp — чтобы взгляд не
    // «дёргался» дискретно раз в тик. РЕШЕНИЕ всё равно за сетью, это лишь
    // механическая интерполяция (как сглаживание мыши).
    private float tgtYaw = 0f;
    private float tgtPitch = 0f;
    private boolean aimTarget = false;
    private boolean lastAimInside = false; // флаг «луч в хитбоксе» для дебага
    private static final float AIM_PER_TICK = 0.8f; // доля оставшегося пути за тик (независимо от FPS)
    private static final float AIM_GAIN = 2.5f;      // усиление выхода прицеливания (доводит до центра)

    public boolean isEnabled() { return enabled; }
    public boolean isModelLoaded() { return model != null; }
    public boolean isRecAim() { return useRecAim; }
    public boolean isRecAimLoaded() { return recAim != null; }

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
            LocalPlayer p = client.player;
            if (p != null) { prevYaw = p.getYRot(); prevPitch = p.getXRot(); }
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
        float[] s = StateVector.collect(self, opp, prevOpp, prevYaw, prevPitch, false);
        float pitchDiff = StateVector.pitchDiffTo(self, opp);

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
        // запоминаем состояние до применения действия — для признаков скорости на след. тике
        prevYaw = yawAtF;
        prevPitch = pitchAtF;
        prevOpp = oppAtF;
        if (dbg++ % 20 == 0) {
            String oodInfo = (useRecAim && recAim != null)
                ? String.format("ood=%d knnD=%.3f", recAim.isOod() ? 1 : 0, recAim.getLastDist())
                : "ood=n/a";
            System.out.println(String.format("[pvpbot] цель=%s d=%.1f dyaw=%.3f dpitch=%.3f pitchDiff=%.2f inHit=%d %s atk=%.2f blk=%.2f",
                opp.getName().getString(), Math.sqrt(self.distanceToSqr(opp)), out[0], out[1], pitchDiff,
                lastAimInside ? 1 : 0, oodInfo, out[4], out[5]));
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
            out[0] = Mth.clamp(ra[0], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
            out[1] = Mth.clamp(ra[1], -Config.MAX_YAW_RT, Config.MAX_YAW_RT);
        }

        // deadzone «луч внутри хитбокса»: если уже целимся в цель, гасим коррекцию,
        // чтобы не раскачиваться на цели (хаос знаков у цели из логов). Вне хитбокса
        // коррекция идёт полностью — удержать цель важнее идеального центра.
        Vec3 eye = self.getEyePosition();
        Vec3 center = new Vec3(opp.getX(), opp.getY() + opp.getBbHeight() * 0.5, opp.getZ());
        Vec3 look = self.getLookAngle().normalize();
        Vec3 toOpp = center.subtract(eye).normalize();
        float aimCos = (float) look.dot(toOpp);
        lastAimInside = aimCos > Config.AIM_COS;
        if (lastAimInside) {
            float damp = 0.15f;
            out[0] *= damp;
            out[1] *= damp;
        }

        // усиление выхода прицеливания: модель имитирует маленькие поправки учителя
        // (П-регулятор с остаточной ошибкой ~25-35°). Gain доворачивает до центра;
        // направление всё равно даёт сеть/retrieval — это не скриптовая геометрия.
        // Gain применяем ТОЛЬКО по yaw: по тангажу в in-game retrieval (полноразмерное
        // KNN-окно) он вызывает уход тангажа в насыщение («смотрит вверх»). Тангаж и так
        // отрабатывался без gain в прошлой версии.
        out[0] = out[0] * AIM_GAIN;
        // Сеть (или retrieval) лишь СТАВИТ ЦЕЛЬ (tgtYaw/tgtPitch); сам плавный доворот
        // между кадрами делает smoothAimFrame() (Mth.rotLerp) — без дёрганья. ---
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
