package dev.pvpbot;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.message.v1.ClientSendMessageEvents;
import net.fabricmc.fabric.api.client.rendering.v1.hud.HudElementRegistry;
import net.fabricmc.fabric.api.client.rendering.v1.hud.VanillaHudElements;
import net.fabricmc.fabric.api.event.player.AttackEntityCallback;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.world.InteractionResult;

import dev.pvpbot.bot.BotController;
import dev.pvpbot.bot.DummyOpponents;
import dev.pvpbot.recording.Recorder;

/**
 * Клиентский мод. Команды через чат (префикс @, не отправляются в сервер):
 *   @start      — начать запись дуэли
 *   @stop       — остановить и сохранить record_NNNN.jsonl
 *   @bot        — вкл/выкл инференс обученной модели
 *   @bot on     — вкл инференс
 *   @bot off    — выкл инференс
 */
public final class PvpBotMod implements ClientModInitializer {
    public static final Recorder RECORDER = new Recorder();
    public static final BotController BOT = new BotController();

    @Override
    public void onInitializeClient() {
        ClientSendMessageEvents.ALLOW_CHAT.register((message) -> {
            String m = message.trim();
            Minecraft client = Minecraft.getInstance();
            if (m.toLowerCase().startsWith("@rec")) {
                String sub = m.substring(4).trim().toLowerCase();
                if (sub.equals("on")) BOT.setRecAim(true, client);
                else if (sub.equals("off")) BOT.setRecAim(false, client);
                else BOT.setRecAim(!BOT.isRecAim(), client);
                client.player.sendSystemMessage(Component.literal("Retrieval-аим: " + (BOT.isRecAim() ? "ВКЛ" : "ВЫКЛ")
                        + (BOT.isRecAim() && !BOT.isRecAimLoaded() ? " · ИНДЕКС НЕ ЗАГРУЖЕН!" : "")));
                return false;
            }
            if (m.toLowerCase().startsWith("@dummy")) {
                String sub = m.substring(6).trim().toLowerCase();
                if (sub.equals("off")) {
                    DummyOpponents.clear(client);
                    client.player.sendSystemMessage(Component.literal("Фейковые противники: ВЫКЛ"));
                } else {
                    int n = 2;
                    try { if (!sub.isEmpty()) n = Math.max(1, Math.min(4, Integer.parseInt(sub))); } catch (Exception ignored) {}
                    DummyOpponents.spawn(client, n);
                    client.player.sendSystemMessage(Component.literal("Фейковые противники: ВКЛ x" + DummyOpponents.count()));
                }
                return false;
            }
            if (m.equalsIgnoreCase("@start")) {
                RECORDER.start(client);
                return false; // не отправлять в чат
            }
            if (m.equalsIgnoreCase("@stop")) {
                RECORDER.stop(client);
                return false;
            }
            if (m.equalsIgnoreCase("@bot on")) {
                BOT.setEnabled(true, client);
                client.player.sendSystemMessage(Component.literal("Бот: ВКЛ" + (BOT.isModelLoaded() ? " · модель загружена" : " · МОДЕЛЬ НЕ ЗАГРУЖЕНА!")));
                return false;
            }
            if (m.equalsIgnoreCase("@bot off")) {
                BOT.setEnabled(false, client);
                client.player.sendSystemMessage(Component.literal("Бот: ВЫКЛ"));
                return false;
            }
            if (m.equalsIgnoreCase("@bot")) {
                BOT.setEnabled(!BOT.isEnabled(), client);
                client.player.sendSystemMessage(Component.literal("Бот: " + (BOT.isEnabled() ? "ВКЛ" : "ВЫКЛ") + (BOT.isModelLoaded() ? " · модель загружена" : " · МОДЕЛЬ НЕ ЗАГРУЖЕНА!")));
                return false;
            }
            return true;
        });

        // фиксируем атаки локального игрока для записи флага attack
        AttackEntityCallback.EVENT.register((player, level, hand, entity, hitResult) -> {
            if (player == Minecraft.getInstance().player) RECORDER.onAttack();
            return InteractionResult.PASS;
        });

        ClientTickEvents.END_CLIENT_TICK.register((client) -> {
            RECORDER.onTick(client);
            DummyOpponents.tick(client);
            BOT.onTick(client);
        });

        // Плавный доворот прицела между кадрами (Mth.rotLerp в BotController).
        // HudElement вызывается каждый кадр; tickDelta берём из DeltaTracker.
        HudElementRegistry.attachElementBefore(
                VanillaHudElements.CROSSHAIR,
                Identifier.withDefaultNamespace("pvpbot_aim"),
                (graphics, deltaTracker) -> BOT.smoothAimFrame(
                        Minecraft.getInstance().player,
                        deltaTracker.getGameTimeDeltaPartialTick(false)));
    }
}
