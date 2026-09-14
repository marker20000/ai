package dev.pvpbot.config;

import java.nio.file.Files;
import java.nio.file.Path;

import net.fabricmc.loader.api.FabricLoader;

/** Пути и глобальные константы мода (совпадают с docs/feature_spec.md). */
public final class Config {
    public static final Path DIR = FabricLoader.getInstance().getConfigDir().resolve("pvpbot");
    public static final Path MODEL_JSON = DIR.resolve("model.json");
    public static final Path MODEL_BIN = DIR.resolve("model.bin");
    public static final Path RECORD_FILE = DIR.resolve("record.jsonl");

    // размерности / нормализация — бьются с training/pvp_bot/config.py
    public static final int WINDOW = 16;
    public static final int FEATURE_DIM = 30;
    public static final int INPUT_DIM = FEATURE_DIM * WINDOW; // 480
    public static final int TARGET_DIM = 6;
    public static final int ATTACK_CH = 4;
    public static final int BLOCK_CH = 5;

    public static final float MAX_DIST = 8.0f;
    public static final float REACH = 3.0f; // дистанция удара (как в sim/physics.py)
    // порог прицеливания: cos макс. угла отклонения взгляда от цели.
    // Явная наводка (BotController) держит взгляд вплотную к центру хитбокса, поэтому
    // можно держать ворота тайтно: 0.85 ≈ 31°. Бот бьёт только когда реально смотрит
    // в цель, а не «в сторону на 53°» (из-за чего казалось, что хитбокс в 2 раза больше).
    public static final float AIM_COS = 0.85f;
    public static final float MAX_SPEED = 0.35f;
    public static final float MAX_YAW_RT = 22.5f;
    public static final float MAX_HP = 20.0f;
    public static final float MAX_ARMOR = 20.0f;
    public static final float MOVE_SPEED = 0.28f;

    public static void ensureDir() {
        try { Files.createDirectories(DIR); } catch (Exception ignored) {}
    }

    private Config() {}
}
