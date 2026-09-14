package dev.pvpbot.net;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Retrieval-аим: вместо MLP ищем в датасете друга ближайшее окно состояния
 * (относительно себя) и берём записанные дельты прицеливания (dyaw, dpitch).
 *
 * Это imitation через retrieval, а не скриптовая геометрия: бот «повторяет»,
 * как целился человек, опираясь на реальные записи. Решение (куда смотреть)
 * по-прежнему дано данными, сеть тут не при чём — это и есть то, что просил
 * пользователь («на условии датасета искал ближайшую траекторию»).
 *
 * Формат rec_index.bin (little-endian, как в Weights.fromBinary):
 *   "PVRI" (4 байта), int32 M, int32 D, float32 W[M*D], float32 Y[M*2].
 */
public final class RecAim {
    private static final int K = 5; // число соседей

    private final int M;
    private final int D;
    private final float[][] W; // [M][D] окна состояния
    private final float[][] Y; // [M][2] dyaw, dpitch учителя

    private RecAim(int M, int D, float[][] w, float[][] y) {
        this.M = M;
        this.D = D;
        this.W = w;
        this.Y = y;
    }

    public static RecAim load(Path dir) {
        Path p = dir.resolve("rec_index.bin");
        if (!Files.exists(p)) {
            System.out.println("[pvpbot] rec_index.bin не найден в " + dir + " — retrieval-аим недоступен");
            return null;
        }
        try {
            byte[] data = Files.readAllBytes(p);
            ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            byte[] magic = new byte[4];
            buf.get(magic);
            if (magic[0] != 'P' || magic[1] != 'V' || magic[2] != 'R' || magic[3] != 'I')
                throw new IOException("битый magic rec_index.bin");
            int m = buf.getInt();
            int d = buf.getInt();
            float[][] w = new float[m][d];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < d; j++) w[i][j] = buf.getFloat();
            float[][] y = new float[m][2];
            for (int i = 0; i < m; i++) {
                y[i][0] = buf.getFloat();
                y[i][1] = buf.getFloat();
            }
            System.out.println("[pvpbot] rec_index.bin загружен: M=" + m + " D=" + d);
            return new RecAim(m, d, w, y);
        } catch (IOException e) {
            System.out.println("[pvpbot] rec_index.bin не загружен: " + e);
            return null;
        }
    }

    public boolean loaded() { return W != null && M > 0; }

    /** Возвращает [dyaw, dpitch] — взвешенное по расстоянию среднее k ближайших. */
    public float[] aim(float[] window) {
        // bestD отсортирован по возрастанию (индекс 0 = ближайший), bestD[K-1] = худший.
        float[] bestD = new float[K];
        int[] bestI = new int[K];
        for (int k = 0; k < K; k++) {
            bestD[k] = Float.MAX_VALUE;
            bestI[k] = -1;
        }
        for (int i = 0; i < M; i++) {
            float d2 = 0f;
            for (int j = 0; j < D; j++) {
                float diff = window[j] - W[i][j];
                d2 += diff * diff;
            }
            if (d2 < bestD[K - 1]) {
                bestD[K - 1] = d2;
                bestI[K - 1] = i;
                // всплываем вверх, восстанавливая порядок по возрастанию
                for (int t = K - 2; t >= 0; t--) {
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
            dyaw += w * Y[bestI[k]][0];
            dpitch += w * Y[bestI[k]][1];
            wsum += w;
        }
        if (wsum > 0f) {
            dyaw /= wsum;
            dpitch /= wsum;
        }
        return new float[]{dyaw, dpitch};
    }
}
