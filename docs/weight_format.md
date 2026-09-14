# Формат обмена весами (Python -> Java)

Два формата, несущих одни и те же данные. JSON — основной (читаемый,
валидируемый), бинарь — компактный для быстрого инференса. Оба парсятся
**вручную**, без сторонних библиотек сериализации.

## Архитектура внутри файла

```
layer_sizes = [432, 256, 128, 64, 6]   # вход, скрытые, выход
activations = ["relu","relu","tanh","linear"]
# слой i: матрица W[i] shape (layer_sizes[i+1], layer_sizes[i]) + вектор b[i] shape (layer_sizes[i+1],)
```

`W[i]` хранится построчно (row-major): строка = выходной нейрон, столбец =
входной. Умножение: `z = W @ a + b`, где `a` — вектор-столбец входа слоя.

## JSON (`model.json`)

```json
{
  "format": "pvpbot-weights-v1",
  "arch": {
    "type": "mlp",
    "layer_sizes": [432, 256, 128, 64, 6],
    "activations": ["relu", "relu", "tanh", "linear"]
  },
  "layers": [
    { "W": [[...], [...]], "b": [...] },
    ...
  ],
  "meta": { "trained_on": "duels-001", "val_loss": 0.0123 }
}
```

- `W` — список списков float (row-major).
- `b` — список float длины `layer_sizes[i+1]`.

## Бинарь (`model.bin`)

Все числа little-endian. float — IEEE-754 single (4 байта).

```
offset  size   поле
0       4      magic = "PVPW" (ASCII)
4       1      version = 1 (uint8)
5       4      num_acts (uint32)         ; число de/активаций
9       ...    для каждой активации:
                 1  byte   len (uint8)
                 len bytes ascii-имя ("relu"/"tanh"/"sigmoid"/"linear")
...     4      num_layers (uint32)
...     для каждого слоя:
            4  uint32  rows
            4  uint32  cols
            rows*cols*4  float32  W (row-major)
            rows*4       float32  b
```

Java-парсер (`Weights.readBinary`) и Python-сериализатор (`serialize.to_binary`)
обязаны совпадать побайтово. Рекомендуется после экспорта прогнать
`python -m pvp_bot.serialize model.json --to-bin model.bin` и сравнить
восстановленные веса.

## Обратная совместимость

При добавлении слоёв/активаций меняется только `layer_sizes`/`activations` и
число слоёв. Структура файла не меняется.
