package dev.pvpbot.bot;

import java.util.ArrayList;
import java.util.List;

import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.entity.Entity.RemovalReason;
import net.minecraft.world.entity.EntityType;
import net.minecraft.world.entity.monster.zombie.Zombie;

/**
 * Тренировочные «фейковые противники» для теста наводки/трекинга в одиночной
 * игре. Это обычные Zombie с выключенным ИИ и бессмертием, которые летают по
 * орбите вокруг игрока, создавая движущуюся цель. Бот целится в них как в
 * обычных Monster (см. BotController.nearestOpponent).
 *
 * НЕ скриптовая логика боя — только передвижение манекена; решение «куда
 * смотреть и бить» по-прежнему за MLP/retrieval.
 */
public final class DummyOpponents {
    private static final List<Zombie> dummies = new ArrayList<>();
    private static int tick = 0;

    public static boolean active() { return !dummies.isEmpty(); }
    public static int count() { return dummies.size(); }

    public static void spawn(Minecraft client, int n) {
        clear(client);
        LocalPlayer p = client.player;
        if (p == null || client.level == null) return;
        for (int i = 0; i < n; i++) {
            Zombie z = new Zombie(
                    (EntityType<Zombie>) (Object) BuiltInRegistries.ENTITY_TYPE.getValue(
                            Identifier.fromNamespaceAndPath("minecraft", "zombie")),
                    client.level);
            z.setNoAi(true);          // не бегает и не атакует сам
            z.setInvulnerable(true);  // не умирает от ударов бота (тест долгий)
            z.setSilent(true);        // без звуков зомби
            double ang = (i / (double) Math.max(1, n)) * Math.PI * 2.0;
            double r = 3.0 + i * 0.8;
            z.setPos(p.getX() + Math.cos(ang) * r, p.getY() + 1.0, p.getZ() + Math.sin(ang) * r);
            z.setYRot((float) (ang * 180.0 / Math.PI + 90.0f));
            client.level.addEntity(z);
            dummies.add(z);
        }
        System.out.println("[pvpbot] заспавнено фейковых противников: " + dummies.size());
    }

    public static void clear(Minecraft client) {
        for (Zombie z : dummies) {
            if (z != null && z.isAlive() && client != null && client.level != null) {
                client.level.removeEntity(z.getId(), RemovalReason.DISCARDED);
            }
        }
        dummies.clear();
    }

    /** Двигаем манекенов по орбите с вертикальным покачиванием — движущаяся цель. */
    public static void tick(Minecraft client) {
        if (dummies.isEmpty() || client.player == null) return;
        tick++;
        LocalPlayer p = client.player;
        double t = tick * 0.05;
        for (int i = 0; i < dummies.size(); i++) {
            Zombie z = dummies.get(i);
            if (!z.isAlive()) { dummies.remove(i); i--; continue; }
            double r = 3.0 + i * 0.8;
            double ang = t * (0.6 + i * 0.35) + i * 2.0;
            double x = p.getX() + Math.cos(ang) * r;
            double zc = p.getZ() + Math.sin(ang) * r;
            double y = p.getY() + 1.0 + Math.sin(t * 1.3 + i) * 0.6;
            z.setPos(x, y, zc);
            z.setYRot((float) (ang * 180.0 / Math.PI + 90.0f));
        }
    }
}
