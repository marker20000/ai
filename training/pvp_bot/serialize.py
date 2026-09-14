"""Сериализация весов MLP в JSON и бинарь. Без сторонних библиотек.

Формат бинаря (little-endian, float = IEEE-754 single):
  [0:4]   magic "PVPW"
  [4:5]   version (uint8) = 1
  [5:9]   num_acts (uint32)
           для каждой активации: 1 byte len + ascii-имя
  [...]    num_layers (uint32)
           для каждого слоя:
             rows (uint32), cols (uint32)
             W: rows*cols * float32 (row-major)
             b: rows * float32
См. docs/weight_format.md.
"""
import struct
import json
import sys
import numpy as np

from .mlp import MLP

MAGIC = b"PVPW"
VERSION = 1


# ---------------- JSON ----------------
def to_json(model, path, meta=None):
    data = {
        "format": "pvpbot-weights-v1",
        "arch": {
            "type": "mlp",
            "layer_sizes": model.sizes,
            "activations": model.acts,
        },
        "layers": [
            {"W": w.tolist(), "b": b.tolist()} for w, b in zip(model.W, model.b)
        ],
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("format") == "pvpbot-weights-v1", "неизвестный формат JSON"
    m = MLP(data["arch"]["layer_sizes"], data["arch"]["activations"])
    for i, layer in enumerate(data["layers"]):
        m.W[i] = np.array(layer["W"], dtype=np.float64)
        m.b[i] = np.array(layer["b"], dtype=np.float64)
    return m


# ---------------- Бинарь ----------------
def to_binary(model, path):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<B", VERSION))
        f.write(struct.pack("<I", len(model.acts)))
        for name in model.acts:
            nb = name.encode("ascii")
            f.write(struct.pack("<B", len(nb)))
            f.write(nb)
        f.write(struct.pack("<I", len(model.W)))
        for w, b in zip(model.W, model.b):
            r, c = w.shape
            f.write(struct.pack("<II", r, c))
            f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
            f.write(np.ascontiguousarray(b, dtype="<f4").tobytes())
    return path


def from_binary(path):
    with open(path, "rb") as f:
        raw = f.read()
    assert raw[:4] == MAGIC, "битый magic бинаря"
    off = 4
    version = struct.unpack_from("<B", raw, off)[0]
    off += 1
    assert version == VERSION, f"неподдерживаемая версия {version}"
    (num_acts,) = struct.unpack_from("<I", raw, off)
    off += 4
    acts = []
    for _ in range(num_acts):
        (ln,) = struct.unpack_from("<B", raw, off)
        off += 1
        acts.append(raw[off:off + ln].decode("ascii"))
        off += ln
    (num_layers,) = struct.unpack_from("<I", raw, off)
    off += 4
    sizes = []
    W, b = [], []
    for _ in range(num_layers):
        r, c = struct.unpack_from("<II", raw, off)
        off += 8
        w = np.frombuffer(raw, dtype="<f4", count=r * c, offset=off).reshape(r, c).astype(np.float64)
        off += r * c * 4
        bb = np.frombuffer(raw, dtype="<f4", count=r, offset=off).astype(np.float64)
        off += r * 4
        W.append(w)
        b.append(bb)
        if sizes:
            sizes.append(r)
        else:
            sizes = [c, r]
    m = MLP(sizes, acts)
    m.W, m.b = W, b
    return m


def convert(in_path, out_path):
    """Загрузить из любого формата и сохранить в нужный."""
    if in_path.endswith(".json"):
        m = from_json(in_path)
    else:
        m = from_binary(in_path)
    if out_path.endswith(".json"):
        return to_json(m, out_path)
    return to_binary(m, out_path)


if __name__ == "__main__":
    # usage: python -m pvp_bot.serialize in.json --to-bin out.bin
    if len(sys.argv) < 3 or sys.argv[2] != "--to-bin":
        print("usage: python -m pvp_bot.serialize <in.json|.bin> --to-bin <out.bin>")
        print("       python -m pvp_bot.serialize <in.bin> --to-json <out.json>")
        sys.exit(1)
    inp, outp = sys.argv[1], sys.argv[3]
    convert(inp, outp)
    print(f"wrote {outp}")
