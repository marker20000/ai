package dev.pvpbot.net;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import dev.pvpbot.config.Config;
import net.minecraft.util.Mth;

/**
 * Retrieval-аим: вместо MLP ищем в датасете ближайшее окно состояния и берём
 * записанные дельты прицеливания (dyaw, dpitch).
 *
 * Решение (куда смотреть) по-прежнему даётся ДАННЫМИ — это imitation через
 * retrieval, а не скриптовая геометрия (как и требовал пользователь). Просто
 * расстояние теперь считается НЕ по всему 480-мерному окну, а только по
 * aim-признакам (dist, yaw_diff, pitch_diff, tar_fwd, tar_side, aim_center):
 * иначе ошибка прицела «тонет» среди 480 признаков и KNN стабильно выбирает
 * один и тот же сосед даже при большой ошибке (залипание dpitch). См. анализ Srafd.
 *
 * OOD-фоллбэк: если ближайший сосед слишком далеко (состояние вне распределения
 * датасета), retrieval не заслуживает доверия. Вместо {0,0} (что клинит взгляд:
 * состояние не меняется → снова OOD → вечно {0,0}) возвращаем пропорциональную
 * коррекцию по СОБСТВЕННОЙ ошибке прицеливания из окна — recovery-контроллер,
 * направление даёт само состояние, не скриптовая геометрия.
 *
 * Формат rec_index.bin (little-endian):
 *   "PVRI" (4 байта), int32 M, int32 D (=WINDOW*FEAT_SEL.length),
 *   float32 W[M*D], float32 Y[M*2].
 */
public final class RecAim {
    private static final int K = 5; // число соседей

    // Только признаки, реально связанные с наводкой (индексы в FEATURE_DIM=30).
    private static final int[] FEAT_SEL = {6, 7, 8, 27, 28, 29}; // dist, yaw_diff, pitch_diff, tar_fwd, tar_side, aim_center
    // Веса расстояния по aim-признакам: доминируют ошибка прицеливания
    // (yaw_diff, pitch_diff) и aim_center — сосед выбирается по ПОХОЖЕЙ ОШИБКЕ
    // ПРИЦЕЛА, а не по случайным признакам состояния. Совпадает с FEAT_W в aim_knn.py.
    private static final float[] AIM_W = {0.5f, 4.0f, 4.0f, 1.0f, 1.0f, 2.0f}; // dist, yaw_diff, pitch_diff, tar_fwd, tar_side, aim_center
    // Проекция 480-мерного окна (WINDOW*FEATURE_DIM) -> 96-мерный aim-вектор.
    private static final int[] COLS;
    static {
        COLS = new int[Config.WINDOW * FEAT_SEL.length];
        int p = 0;
        for (int w = 0; w < Config.WINDOW; w++)
            for (int f : FEAT_SEL)
                COLS[p++] = w * Config.FEATURE_DIM + f;
    }

    // Временное затухание по кадрам окна: недавние кадры важнее старых, чтобы
    // retrieval быстрее реагировал на резкие изменения (прыжок/flick/смена
    // направления), не теряя пользу истории для плавного tracking. Старый кадр —
    // малый вес, новый — 1.0. Сами признаки не меняются, rec_index.bin
    // пересобирать не надо (веса только в метрике расстояния).
    private static final float[] TEMP_W;
    static {
        final int W = Config.WINDOW;
        TEMP_W = new float[W];
        for (int w = 0; w < W; w++)
            TEMP_W[w] = 0.15f + 0.85f * (w / (float) (W - 1));
    }

    // Во сколько раз типичная ближайшая дистанция должна быть превышена, чтобы
    // состояние считалось вне распределения (OOD). Подбирается опытным путём.
    private static final float OOD_FACTOR = 5.0f;

    private final int M;
    private final int D;
    private final float[][] W; // [M][D] окна состояния (уже спроецированные)
    private final float[][] Y; // [M][2] dyaw, dpitch учителя
    private final float oodThreshold; // квадрат расстояния, выше = OOD
    private boolean lastOod = false;   // OOD-флаг последнего aim() (для дебага)
    private float lastDist = 0f;       // расстояние до ближайшего соседа (для дебага)
    // Последние K соседей (индекс + квадрат расстояния) для диагностики: видно,
    // согласованны ли соседи по dyaw/dpitch, или K=5 усредняет противоречивые
    // решения в ≈0. Заполняется в aim() независимо от OOD-ветки.
    private final int[] lastNbI = new int[K];
    private final float[] lastNbD = new float[K];

    private RecAim(int M, int D, float[][] w, float[][] y, float oodThreshold) {
        this.M = M;
        this.D = D;
        this.W = w;
        this.Y = y;
        this.oodThreshold = oodThreshold;
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
            if (d != COLS.length)
                throw new IOException("rec_index.bin имеет D=" + d + ", ожидалось " + COLS.length +
                        " (пересоберите через aim_knn.py --export)");
            float[][] w = new float[m][d];
            for (int i = 0; i < m; i++)
                for (int j = 0; j < d; j++) w[i][j] = buf.getFloat();
            float[][] y = new float[m][2];
            for (int i = 0; i < m; i++) {
                y[i][0] = buf.getFloat();
                y[i][1] = buf.getFloat();
            }
            float ood = computeOodThreshold(w);
            System.out.println("[pvpbot] rec_index.bin загружен: M=" + m + " D=" + d + " oodThreshold=" + ood);
            return new RecAim(m, d, w, y, ood);
        } catch (IOException e) {
            System.out.println("[pvpbot] rec_index.bin не загружен: " + e);
            return null;
        }
    }

    /** Адаптивная оценка порога OOD: берём выборку строк индекса, для каждой —
     * минимальную квадрат-дистанцию до ДРУГОЙ строки (типичная близость внутри
     * распределения), 90-й перцентиль умножаем на фактор. Дешёвая и масштабируемая. */
    private static float computeOodThreshold(float[][] w) {
        int n = w.length;
        if (n < 2) return Float.MAX_VALUE;
        int sample = Math.min(n, 512);
        List<Float> near = new ArrayList<>(sample);
        for (int s = 0; s < sample; s++) {
            int i = (s * n) / sample; // равномерная выборка
            float best = Float.MAX_VALUE;
            for (int j = 0; j < n; j++) {
                if (j == i) continue;
                float d2 = dist2(w[i], w[j]);
                if (d2 < best) best = d2;
            }
            near.add(best);
        }
        float[] arr = new float[near.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = near.get(i);
        Arrays.sort(arr);
        float base = arr[(int) (0.90 * (arr.length - 1))];
        return base * OOD_FACTOR;
    }

    public boolean loaded() { return W != null && M > 0; }
    public boolean isOod() { return lastOod; }
    public float getLastDist() { return lastDist; }
    public float getOodThreshold() { return oodThreshold; }

    /** Диагностика top-K соседей: индекс, расстояние, yawDiff соседа (° из признака
     *  7 последнего кадра окна) и учительские dyaw/dpitch. Если соседи противоречат
     *  друг другу (одни вправо, другие влево), K=5 усредняет решение в ≈0 — бот
     *  «теряет цель». Читается в BotController каждые 20 тиков при @rec. */
    public String getNeighborDebug() {
        final int yawProj = (Config.WINDOW - 1) * FEAT_SEL.length + 1; // позиция yaw_diff(7) в 96-мерной проекции
        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < K; k++) {
            int i = lastNbI[k];
            if (i < 0) continue;
            float d = (float) Math.sqrt(lastNbD[k]);
            sb.append(String.format("#%d d=%.3f yawDiff=%+.1f dyaw=%+.2f dpitch=%+.2f  ",
                k + 1, d, W[i][yawProj] * 180f, Y[i][0], Y[i][1]));
        }
        return sb.toString();
    }

    /** Взвешенное (по AIM_W) квадратичное расстояние между двумя 96-мерными
     *  окнами: доминируют признаки ошибки прицеливания. */
    private static float dist2(float[] a, float[] b) {
        float d2 = 0f;
        for (int w = 0; w < Config.WINDOW; w++) {
            int base = w * FEAT_SEL.length;
            for (int f = 0; f < FEAT_SEL.length; f++) {
                float diff;
                if (FEAT_SEL[f] == 7) {
                    // yaw_diff (f[7]=yawDiff/180) — циклический признак: +179° и -179°
                    // физически в 2° друг от друга, но евклидова разница ~358°. Сворачиваем
                    // по окружности, иначе KNN на границе ±180° рвёт соседей (критично для
                    // recovery на 180°). pitch_diff (f[8]) циклическим НЕ является.
                    diff = Mth.wrapDegrees((a[base + f] - b[base + f]) * 180.0f) / 180.0f;
                } else {
                    diff = a[base + f] - b[base + f];
                }
                d2 += AIM_W[f] * TEMP_W[w] * diff * diff;
            }
        }
        return d2;
    }

    /** Возвращает [dyaw, dpitch] — взвешенное по расстоянию среднее k ближайших.
     *  window — полное 480-мерное окно; проецируется на aim-признаки. */
    public float[] aim(float[] window) {
        float[] pw = new float[D];
        for (int k = 0; k < D; k++) pw[k] = window[COLS[k]];

        float[] bestD = new float[K];
        int[] bestI = new int[K];
        for (int k = 0; k < K; k++) {
            bestD[k] = Float.MAX_VALUE;
            bestI[k] = -1;
        }
        for (int i = 0; i < M; i++) {
            float d2 = dist2(pw, W[i]);
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
        // сохраняем K соседей для диагностики (до OOD-ветки, чтобы были доступны в обоих случаях)
        for (int k = 0; k < K; k++) { lastNbI[k] = bestI[k]; lastNbD[k] = bestD[k]; }
        // OOD: ближайший сосед слишком далеко — retrieval недостоверен. Вместо
        // {0,0} (это клинит взгляд: состояние не меняется → снова OOD → вечно
        // {0,0}) возвращаем пропорциональную коррекцию по СОБСТВЕННОЙ ошибке
        // прицеливания из окна (признаки 7/8 = yawDiff/pitchDiff). Это
        // recovery-контроллер: направление даёт само состояние, не скрипт.
        if (bestD[0] > oodThreshold) {
            lastOod = true;
            lastDist = (float) Math.sqrt(bestD[0]);
            int last = (Config.WINDOW - 1) * Config.FEATURE_DIM;
            float yawDiffDeg = window[last + 7] * 180f;   // f[7] = yawDiff/180
            float pitchDiffDeg = window[last + 8] * 90f;  // f[8] = pitchDiff/90
            float k = 0.25f;
            return new float[]{
                Mth.clamp(yawDiffDeg * k, -Config.MAX_YAW_RT, Config.MAX_YAW_RT),
                Mth.clamp(pitchDiffDeg * k, -Config.MAX_YAW_RT, Config.MAX_YAW_RT)
            };
        }
        float dyaw = 0f, dpitch = 0f, wsum = 0f;
        for (int k = 0; k < K; k++) {
            if (bestI[k] < 0) continue;
            float wgt = (float) (1.0 / (Math.sqrt(bestD[k]) + 1e-3));
            dyaw += wgt * Y[bestI[k]][0];
            dpitch += wgt * Y[bestI[k]][1];
            wsum += wgt;
        }
        if (wsum > 0f) {
            dyaw /= wsum;
            dpitch /= wsum;
        }
        lastOod = false;
        lastDist = (float) Math.sqrt(bestD[0]);
        return new float[]{dyaw, dpitch};
    }
}
