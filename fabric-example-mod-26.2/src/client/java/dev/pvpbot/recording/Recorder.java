package dev.pvpbot.recording;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.LivingEntity;
import net.minecraft.world.entity.monster.Monster;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.phys.Vec3;

import dev.pvpbot.bot.StateVector;
import dev.pvpbot.config.Config;

/**
 * Запись дуэлей в JSONL (config/pvpbot/record_NNNN.jsonl).
 * Одна строка на тик: {"state":[27], "target":[6]}.
 * Порядок признаков — строго по docs/feature_spec.md.
 * Сегментация на эпизоды (от удара до удара) выполняется позже конвертером.
 */
public final class Recorder {
    private boolean recording = false;
    private final List<float[]> states = new ArrayList<>();
    private final List<float[]> targets = new ArrayList<>();
    private float prevYaw = 0f;
    private float prevPitch = 0f;
    private Vec3 prevOpp = null;
    private boolean attacked = false;
    private int tick = 0;

    public boolean isRecording() { return recording; }

    public void start(Minecraft client) {
        recording = true;
        states.clear();
        targets.clear();
        tick = 0;
        attacked = false;
        LocalPlayer self = client.player;
        if (self != null) { prevYaw = self.getYRot(); prevPitch = self.getXRot(); }
        prevOpp = null;
        Config.ensureDir();
    }

    public void onAttack() { attacked = true; }

    public void onTick(Minecraft client) {
        if (!recording) return;
        LocalPlayer self = client.player;
        if (self == null || client.level == null) return;

        LivingEntity opp = nearestOpponent(client, self);
        if (opp == null) { prevOpp = null; return; }

        float[] state = StateVector.collect(self, opp, prevOpp, prevYaw, prevPitch, attacked);
        float[] tgt = StateVector.target(self, prevYaw, prevPitch, attacked);
        states.add(state);
        targets.add(tgt);

        prevYaw = self.getYRot();
        prevPitch = self.getXRot();
        prevOpp = new Vec3(opp.getX(), opp.getY(), opp.getZ());
        attacked = false;
        tick++;
    }

    public void stop(Minecraft client) {
        if (!recording) return;
        recording = false;
        if (states.isEmpty()) return;
        Config.ensureDir();
        Path out = nextRecordPath();
        try (BufferedWriter bw = Files.newBufferedWriter(out, StandardCharsets.UTF_8)) {
            for (int i = 0; i < states.size(); i++) {
                bw.write("{\"state\":[");
                bw.write(floats(states.get(i)));
                bw.write("],\"target\":[");
                bw.write(floats(targets.get(i)));
                bw.write("]}");
                bw.newLine();
            }
        } catch (IOException e) {
            throw new RuntimeException("не удалось записать " + out, e);
        }
        states.clear();
        targets.clear();
    }

    /** Каждая сессия — отдельный файл record_NNNN.jsonl (чтобы эпизоды не сливались). */
    private static Path nextRecordPath() {
        int max = 0;
        try (var stream = Files.list(Config.DIR)) {
            for (Path p : (Iterable<Path>) stream::iterator) {
                String name = p.getFileName().toString();
                if (name.startsWith("record_") && name.endsWith(".jsonl")) {
                    try {
                        int n = Integer.parseInt(name.substring(7, name.length() - 6));
                        if (n > max) max = n;
                    } catch (NumberFormatException ignored) {}
                }
            }
        } catch (IOException ignored) {}
        return Config.DIR.resolve(String.format("record_%04d.jsonl", max + 1));
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

    private static String floats(float[] a) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < a.length; i++) {
            if (i > 0) sb.append(',');
            // Locale.US — иначе на локали с запятой вместо точки (ru_RU) String.format
            // пишет «0,0689» и JSON становится невалидным (парсер падает).
            sb.append(a[i] == 0f ? "0" : String.format(Locale.US, "%.5g", a[i]));
        }
        return sb.toString();
    }
}
