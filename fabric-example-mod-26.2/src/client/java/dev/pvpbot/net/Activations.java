package dev.pvpbot.net;

/** Активации, реализованные чистой Java (без библиотек). */
public final class Activations {
    public static float relu(float x) { return Math.max(0f, x); }
    public static float tanh(float x) { return (float) Math.tanh(x); }
    public static float sigmoid(float x) {
        if (x >= 0f) return (float) (1.0 / (1.0 + Math.exp(-x)));
        float e = (float) Math.exp(x);
        return e / (1.0f + e);
    }
    public static float linear(float x) { return x; }

    public static float apply(String name, float x) {
        switch (name) {
            case "relu":   return relu(x);
            case "tanh":   return tanh(x);
            case "sigmoid":return sigmoid(x);
            default:       return linear(x);
        }
    }

    private Activations() {}
}
