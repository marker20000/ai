package dev.pvpbot.net;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Загрузка весов MLP из JSON или бинаря (см. docs/weight_format.md).
 * Парсится вручную, без сторонних библиотек сериализации.
 *
 * W[layer] : float[out][in] (row-major, строка = выходной нейрон)
 * b[layer] : float[out]
 */
public final class Weights {
    public final int[] layerSizes;
    public final String[] activations;
    public final float[][][] W;
    public final float[][] b;

    private Weights(int[] layerSizes, String[] activations, float[][][] w, float[][] b) {
        this.layerSizes = layerSizes;
        this.activations = activations;
        this.W = w;
        this.b = b;
    }

    public static Weights load(Path dir) {
        Path bin = dir.resolve("model.bin");
        Path json = dir.resolve("model.json");
        if (Files.exists(bin)) return fromBinary(bin);
        if (Files.exists(json)) return fromJson(json);
        throw new IllegalStateException("ни model.bin, ни model.json не найдены в " + dir);
    }

    // ---------------- JSON (ручной парсер) ----------------
    public static Weights fromJson(Path path) {
        try {
            byte[] bytes = Files.readAllBytes(path);
            JsonValue root = JsonValue.parse(new String(bytes, StandardCharsets.UTF_8));
            Map<String, Object> m = root.asObject();

            Map<String, Object> arch = (Map<String, Object>) m.get("arch");
            List<Object> ls = (List<Object>) arch.get("layer_sizes");
            int[] sizes = new int[ls.size()];
            for (int i = 0; i < sizes.length; i++) sizes[i] = ((Number) ls.get(i)).intValue();
            List<Object> acts = (List<Object>) arch.get("activations");
            String[] activations = new String[acts.size()];
            for (int i = 0; i < activations.length; i++) activations[i] = acts.get(i).toString();

            List<Object> layers = (List<Object>) m.get("layers");
            int n = layers.size();
            float[][][] W = new float[n][][];
            float[][] b = new float[n][];
            for (int i = 0; i < n; i++) {
                Map<String, Object> layer = (Map<String, Object>) layers.get(i);
                List<Object> wRows = (List<Object>) layer.get("W");
                int r = wRows.size();
                int c = ((List<Object>) wRows.get(0)).size();
                float[][] w = new float[r][c];
                for (int rr = 0; rr < r; rr++) {
                    List<Object> row = (List<Object>) wRows.get(rr);
                    for (int cc = 0; cc < c; cc++) w[rr][cc] = ((Number) row.get(cc)).floatValue();
                }
                List<Object> bb = (List<Object>) layer.get("b");
                float[] bv = new float[bb.size()];
                for (int k = 0; k < bv.length; k++) bv[k] = ((Number) bb.get(k)).floatValue();
                W[i] = w;
                b[i] = bv;
            }
            return new Weights(sizes, activations, W, b);
        } catch (IOException e) {
            throw new RuntimeException("не удалось прочитать " + path, e);
        }
    }

    // ---------------- Бинарь ----------------
    // Python пишет little-endian (struct "<"); Java DataInputStream/Buffer по
    // умолчанию BIG_ENDIAN, поэтому читаем через ByteBuffer с LITTLE_ENDIAN.
    public static Weights fromBinary(Path path) {
        try {
            byte[] data = Files.readAllBytes(path);
            ByteBuffer buf = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);

            byte[] magic = new byte[4];
            buf.get(magic);
            if (magic[0] != 'P' || magic[1] != 'V' || magic[2] != 'P' || magic[3] != 'W')
                throw new IOException("битый magic бинаря");
            int version = buf.get() & 0xFF;
            if (version != 1) throw new IOException("неподдерживаемая версия " + version);

            int numActs = buf.getInt();
            String[] activations = new String[numActs];
            for (int i = 0; i < numActs; i++) {
                int len = buf.get() & 0xFF;
                byte[] name = new byte[len];
                buf.get(name);
                activations[i] = new String(name, StandardCharsets.US_ASCII);
            }
            int numLayers = buf.getInt();
            float[][][] W = new float[numLayers][][];
            float[][] b = new float[numLayers][];
            int[] sizes = null;
            for (int i = 0; i < numLayers; i++) {
                int rows = buf.getInt();
                int cols = buf.getInt();
                float[][] w = new float[rows][cols];
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++) w[r][c] = buf.getFloat();
                float[] bv = new float[rows];
                for (int r = 0; r < rows; r++) bv[r] = buf.getFloat();
                W[i] = w; b[i] = bv;
                if (sizes == null) sizes = new int[]{cols, rows};
                else sizes = append(sizes, rows);
            }
            return new Weights(sizes, activations, W, b);
        } catch (IOException e) {
            throw new RuntimeException("не удалось прочитать " + path, e);
        }
    }

    private static int[] append(int[] a, int v) {
        int[] out = new int[a.length + 1];
        System.arraycopy(a, 0, out, 0, a.length);
        out[a.length] = v;
        return out;
    }

    /** Минимальный JSON-парсер: объекты, массивы, числа, строки, true/false/null. */
    private static final class JsonValue {
        final Object value;
        JsonValue(Object v) { this.value = v; }
        @SuppressWarnings("unchecked")
        Map<String, Object> asObject() { return (Map<String, Object>) value; }

        static JsonValue parse(String s) {
            Parser p = new Parser(s);
            p.ws();
            return new JsonValue(p.value());
        }

        static final class Parser {
            final String s; int i;
            Parser(String s) { this.s = s; this.i = 0; }
            void ws() { while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++; }
            Object value() {
                ws();
                char c = s.charAt(i);
                if (c == '{') return object();
                if (c == '[') return array();
                if (c == '"') return string();
                if (c == 't' || c == 'f') return bool();
                if (c == 'n') { i += 4; return null; }
                return number();
            }
            Map<String, Object> object() {
                i++; Map<String, Object> m = new LinkedHashMap<>();
                ws();
                if (s.charAt(i) == '}') { i++; return m; }
                while (true) {
                    ws(); String key = string();
                    ws(); i++; // ':'
                    Object v = value();
                    m.put(key, v);
                    ws();
                    if (s.charAt(i) == ',') { i++; continue; }
                    i++; // '}'
                    return m;
                }
            }
            List<Object> array() {
                i++; List<Object> list = new ArrayList<>();
                ws();
                if (s.charAt(i) == ']') { i++; return list; }
                while (true) {
                    list.add(value());
                    ws();
                    if (s.charAt(i) == ',') { i++; continue; }
                    i++; // ']'
                    return list;
                }
            }
            String string() {
                i++; StringBuilder sb = new StringBuilder();
                while (i < s.length() && s.charAt(i) != '"') {
                    char c = s.charAt(i++);
                    if (c == '\\') {
                        char e = s.charAt(i++);
                        if (e == 'n') sb.append('\n');
                        else if (e == 't') sb.append('\t');
                        else if (e == '"') sb.append('"');
                        else if (e == '\\') sb.append('\\');
                        else sb.append(e);
                    } else sb.append(c);
                }
                i++; // closing quote
                return sb.toString();
            }
            Object number() {
                int start = i;
                while (i < s.length()) {
                    char c = s.charAt(i);
                    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
                        i++;
                    else break;
                }
                return Double.parseDouble(s.substring(start, i));
            }
            boolean bool() {
                if (s.startsWith("true", i)) { i += 4; return true; }
                i += 5; return false;
            }
        }
    }
}
