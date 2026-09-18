package dev.pvpbot.mixin;

import net.minecraft.client.multiplayer.MultiPlayerGameMode;
import net.minecraft.network.protocol.game.ServerboundMovePlayerPacket;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyVariable;
import dev.pvpbot.PvpBotMod;

/**
 * Mixin для перехвата отправки пакетов поворота камеры на сервер.
 * При @silent режиме подменяет локальные yaw/pitch на серверные из BotController.
 */
@Mixin(ServerboundMovePlayerPacket.Rot.class)
public class MixinServerboundMovePlayerPacketRot {
    
    @ModifyVariable(
        method = "<init>(FFZZ)V",
        at = @At("HEAD"),
        ordinal = 0,
        argsOnly = true
    )
    private static float modifyYaw(float yaw) {
        if (PvpBotMod.BOT.isSilentMode() && PvpBotMod.BOT.isEnabled()) {
            return PvpBotMod.BOT.getServerYaw();
        }
        return yaw;
    }
    
    @ModifyVariable(
        method = "<init>(FFZZ)V",
        at = @At("HEAD"),
        ordinal = 1,
        argsOnly = true
    )
    private static float modifyPitch(float pitch) {
        if (PvpBotMod.BOT.isSilentMode() && PvpBotMod.BOT.isEnabled()) {
            return PvpBotMod.BOT.getServerPitch();
        }
        return pitch;
    }
}
