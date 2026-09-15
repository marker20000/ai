package dev.pvpbot.bot;

import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.tags.ItemTags;
import net.minecraft.world.item.AxeItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.phys.Vec3;
import net.minecraft.util.Mth;

import dev.pvpbot.config.Config;

/**
 * Извлечение 29 нормализованных признаков состояния (строго по docs/feature_spec.md).
 * self — локальный игрок (он же «бот» в режиме записи/инференса),
 * opp — противник, prevOpp — позиция противника на предыдущем тике.
 */
public final class StateVector {
    private StateVector() {}

    public static float[] collect(LocalPlayer self, LivingEntity opp,
                                  Vec3 prevOpp, float prevSelfYaw, float prevSelfPitch,
                                  boolean attackedThisTick) {
        float[] f = new float[Config.FEATURE_DIM];

        Vec3 selfEye = self.getEyePosition();
        double oppX = opp.getX(), oppY = opp.getY(), oppZ = opp.getZ();
        double oppCenterY = oppY + opp.getBbHeight() * 0.5;

        double dx = oppX - self.getX();
        double dy = oppCenterY - selfEye.y;
        double dz = oppZ - self.getZ();
        double dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        double hdist = Math.sqrt(dx * dx + dz * dz);

        double desiredYaw = Math.toDegrees(Math.atan2(-dx, dz));
        double desiredPitch = Math.toDegrees(Math.atan2(-dy, hdist)); // MC: положительный pitch = ВНИЗ (look.y = -sin(pitch)); цель выше (dy>0) => pitch<0
        double yawDiff = Mth.wrapDegrees(desiredYaw - self.getYRot());
        double pitchDiff = desiredPitch - self.getXRot();

        Vec3 oppVel = (prevOpp == null) ? Vec3.ZERO
                : new Vec3(oppX - prevOpp.x, oppY - prevOpp.y, oppZ - prevOpp.z);

        double yawRad = Math.toRadians(self.getYRot());
        // базис совпадает с MC: вперёд = (-sin yaw, cos yaw), вправо = (-cos yaw, -sin yaw)
        double fX = -Math.sin(yawRad), fZ = Math.cos(yawRad);
        double rX = -Math.cos(yawRad), rZ = -Math.sin(yawRad);
        Vec3 vel = self.getDeltaMovement();
        double fwdSpeed = vel.x * fX + vel.z * fZ;
        double strafeSpeed = vel.x * rX + vel.z * rZ;

        double yawRate = Mth.wrapDegrees(self.getYRot() - prevSelfYaw) / Config.MAX_YAW_RT;
        double pitchRate = (self.getXRot() - prevSelfPitch) / Config.MAX_YAW_RT;

        ItemStack main = self.getMainHandItem();
        boolean sword = main.is(ItemTags.SWORDS);
        boolean axe = main.getItem() instanceof AxeItem;
        boolean crit = !self.onGround() && vel.y < 0.0;
        boolean ground = self.onGround();
        boolean water = self.isInWater();

        int p = 0;
        f[p++] = (float) (dx / Config.MAX_DIST);
        f[p++] = (float) (dy / Config.MAX_DIST);
        f[p++] = (float) (dz / Config.MAX_DIST);
        f[p++] = (float) (oppVel.x / Config.MAX_SPEED);
        f[p++] = (float) (oppVel.y / Config.MAX_SPEED);
        f[p++] = (float) (oppVel.z / Config.MAX_SPEED);
        f[p++] = (float) (Math.min(dist, Config.MAX_DIST) / Config.MAX_DIST);
        f[p++] = (float) (yawDiff / 180.0);
        f[p++] = (float) (pitchDiff / 90.0);
        f[p++] = (float) (fwdSpeed / Config.MAX_SPEED);
        f[p++] = (float) (strafeSpeed / Config.MAX_SPEED);
        f[p++] = (float) (self.getYRot() / 180.0);
        f[p++] = (float) (self.getXRot() / 90.0);
        f[p++] = (float) Math.max(-1.0, Math.min(1.0, yawRate));
        f[p++] = (float) Math.max(-1.0, Math.min(1.0, pitchRate));
        f[p++] = self.getAttackStrengthScale(0.0f); // 0..1, 1 — готово
        f[p++] = sword ? 1f : 0f;
        f[p++] = axe ? 1f : 0f;
        f[p++] = (!sword && !axe) ? 1f : 0f;
        f[p++] = crit ? 1f : 0f;
        f[p++] = ground ? 1f : 0f;
        f[p++] = (!ground && !water) ? 1f : 0f;
        f[p++] = water ? 1f : 0f;
        f[p++] = (float) (self.getHealth() / Config.MAX_HP);
        f[p++] = (float) (self.getArmorValue() / Config.MAX_ARMOR);
        f[p++] = (float) (opp.getHealth() / Config.MAX_HP);
        f[p++] = (float) (opp.getArmorValue() / Config.MAX_ARMOR);
        // ориентация цели относительно взгляда игрока: скорость цели, спроецированная
        // на базис «вперёд/вправо» игрока. Аналог rec_*.json tarForward/tarSide (там ~0,
        // здесь — реальное движение цели: идёт ли она на меня или вбок). Признаки 27,28.
        f[p++] = (float) ((oppVel.x * fX + oppVel.z * fZ) / Config.MAX_SPEED);
        f[p++] = (float) ((oppVel.x * rX + oppVel.z * rZ) / Config.MAX_SPEED);
        // aim_center: насколько взгляд в центре хитбокса цели.
        // 1 = точно в центре, 0 = на краю хитбокса (cos=AIM_COS — порог удара), -1 = ~180° от цели.
        // Даёт сети плавный сигнал «остаток ошибки прицела» (признак 29).
        Vec3 svLook = self.getLookAngle();
        Vec3 svToOpp = new Vec3(oppX - selfEye.x, oppCenterY - selfEye.y, oppZ - selfEye.z).normalize();
        double svAimCos = svLook.dot(svToOpp);
        f[p++] = (float) Math.max(-1.0, Math.min(1.0, (svAimCos - Config.AIM_COS) / (1.0 - Config.AIM_COS)));
        return f;
    }

    /** Ошибка прицеливания по pitch (desiredPitch − текущий pitch) — ровно как в
     *  collect(), но без сборки всей 30-меры. Нужно для дебага bias наводки
     *  (сравнения out[1] сети/retrieval и реальной ошибки) в BotController. */
    public static float pitchDiffTo(LocalPlayer self, LivingEntity opp) {
        Vec3 selfEye = self.getEyePosition();
        double oppX = opp.getX(), oppY = opp.getY(), oppZ = opp.getZ();
        double oppCenterY = oppY + opp.getBbHeight() * 0.5;
        double dx = oppX - self.getX();
        double dy = oppCenterY - selfEye.y;
        double dz = oppZ - self.getZ();
        double hdist = Math.sqrt(dx * dx + dz * dz);
        double desiredPitch = Math.toDegrees(Math.atan2(-dy, hdist));
        return (float) (desiredPitch - self.getXRot());
    }

    /** Целевые действия игрока на этот тик (для imitation learning). */
    public static float[] target(LocalPlayer self,
                                 float prevSelfYaw, float prevSelfPitch, boolean attackedThisTick) {
        float[] t = new float[Config.TARGET_DIM];
        t[0] = Mth.wrapDegrees(self.getYRot() - prevSelfYaw);
        t[1] = self.getXRot() - prevSelfPitch;
        var ip = self.input.keyPresses;
        float mf = (ip.forward() ? 1f : 0f) - (ip.backward() ? 1f : 0f);
        float ms = (ip.left() ? 1f : 0f) - (ip.right() ? 1f : 0f);
        t[2] = Math.max(-1f, Math.min(1f, mf));
        t[3] = Math.max(-1f, Math.min(1f, ms));
        t[4] = attackedThisTick ? 1f : 0f;
        t[5] = self.isBlocking() ? 1f : 0f;
        return t;
    }
}
