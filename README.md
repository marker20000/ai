# PvP-bot — обучаемый NPC для тренировки PvP в Minecraft

ИИ-управляемый противник, обучающийся повторять стиль игрока (imitation learning)
с последующей лёгкой эволюционной дошлифовкой. Математика сети реализована
**вручную** (без PyTorch/TensorFlow/sklearn).

- **Обучение:** Python + numpy (матричная математика, forward/backward сами).
- **Рантайм:** Java-клиентский мод под Fabric (инференс чистой Java).
- **Данные:** сначала ручная запись дуэлей через мод, потом headless-симуляция
  (6 ботов по парам учатся друг на друге).

## Быстрый старт (Python)

```bash
cd training
pip install -r requirements.txt

# Прогнать весь конвейер на синтетике (без реальных данных):
python train.py selftest

# Обучить imitation-модель на датасете (npz с states/targets):
python train.py train --data data/dataset.npz --epochs 40 --out model.json

# Экспорт в бинарь для мода:
python -m pvp_bot.serialize model.json --to-bin model.bin

# Самообучение 6 ботов (headless-симуляция):
python selfplay.py --generations 20 --pop 12 --out champion.json
```

## Быстрый старт (Java / Fabric-мод)

```bash
cd mod
./gradlew build        # требует Fabric MDK / интернета при первой сборке
```

Положи `model.json` (или `model.bin`) в папку, указанную в `Config.java`
(по умолчанию `config/pvpbot/model.json`). В игре мод:

- **Режим записи** (`record=true`): пишет тики дуэлей в
  `config/pvpbot/record.jsonl` в формате `feature_spec.md`.
- **Режим бота** (`bot=true`): на каждый client-tick собирает окно состояния,
  прогоняет сеть и управляет локальным игроком (камера/движение/атака).

## Контракты (читать обязательно при правках)

- `docs/feature_spec.md` — размерности входа/выхода и нормализация.
- `docs/weight_format.md` — формат файла весов (JSON + бинарь).

Любое изменение в одной реализации должно синхронно править другую.
