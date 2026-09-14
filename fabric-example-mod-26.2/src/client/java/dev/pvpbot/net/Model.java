package dev.pvpbot.net;

import java.nio.file.Path;

import dev.pvpbot.config.Config;

/** MLP-инференс, реализованный вручную. Без библиотек. */
public final class Model {
    private final Weights w;

    public Model(Weights w) { this.w = w; }

    public static Model load(Path dir) {
        return new Model(Weights.load(dir));
    }

    public int inputDim() { return w.layerSizes[0]; }
    public int outputDim() { return w.layerSizes[w.layerSizes.length - 1]; }

    /** input.length должен равняться inputDim(). Возвращает TARGET_DIM значений. */
    public float[] forward(float[] input) {
        float[] a = input;
        for (int l = 0; l < w.W.length; l++) {
            float[][] wm = w.W[l];
            float[] bb = w.b[l];
            float[] out = new float[wm.length];
            for (int r = 0; r < wm.length; r++) {
                float sum = bb[r];
                float[] row = wm[r];
                for (int c = 0; c < row.length; c++) sum += row[c] * a[c];
                out[r] = Activations.apply(w.activations[l], sum);
            }
            a = out;
        }
        // sigmoid на каналах атаки/блока (логиты)
        a[Config.ATTACK_CH] = Activations.sigmoid(a[Config.ATTACK_CH]);
        a[Config.BLOCK_CH] = Activations.sigmoid(a[Config.BLOCK_CH]);
        return a;
    }
}
