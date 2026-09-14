# Архитектура проекта PvP-bot

```
ai bot/
├── README.md
├── docs/
│   ├── feature_spec.md     # контракт признаков (Python <-> Java)
│   ├── weight_format.md    # формат весов (JSON + бинарь)
│   └── architecture.md     # этот файл
│
├── training/               # Python: обучение + эволюция (numpy/чистые списки)
│   ├── requirements.txt
│   ├── pvp_bot/
│   │   ├── config.py        # FEATURE_DIM, WINDOW, INPUT_DIM, TARGET_DIM, норм-константы
│   │   ├── activations.py  # relu/tanh/sigmoid/linear + градиенты
│   │   ├── mlp.py           # MLP: forward/backward вручную, get/set_flat
│   │   ├── lstm.py          # (опц.) ручной LSTM/GRU + BPTT
│   │   ├── optim.py         # SGD / Adam вручную
│   │   ├── loss.py          # MSE (континуум) + BCE (attack/block)
│   │   ├── serialize.py     # JSON + бинарь весов
│   │   ├── dataset.py       # загрузка, окно, синтетика
│   │   ├── trainer.py       # цикл обучения, train/val
│   │   └── evolution.py     # мутации весов + отбор (без RL-бэкпропа)
│   ├── sim/
│   │   ├── physics.py       # углы, движение, hit-детект
│   │   └── arena.py         # headless-бой бот-против-бота
│   ├── train.py             # CLI: train / selftest / evolve / export
│   └── selfplay.py          # 6 ботов по парам, самообучение
│
└── mod/                    # Java: Fabric клиентский мод
    ├── build.gradle, settings.gradle, gradle.properties
    ├── src/main/resources/fabric.mod.json
    └── src/main/java/dev/pvpbot/
        ├── PvpBotMod.java          # ModInitializer, регистрация тиков
        ├── config/Config.java      # пути весов, флаги rec/bot
        ├── recording/Recorder.java  # сбор датасета из дуэлей
        ├── net/Activations.java     # активации (чистая Java)
        ├── net/Weights.java         # парсер JSON/бинарь (вручную)
        ├── net/Model.java           # MLP-инференс вручную
        ├── bot/StateVector.java     # извлечение 27 признаков из MC
        └── bot/BotController.java   # применение действий к игроку
```

## Поток данных

```
[Ручная запись дуэлей] --(JSONL)--> dataset
                                        |
[Синтетика / self-play sim] --(эпизоды)-> dataset
                                        |
                                        v
                              training/train.py  (imitation)
                                        |
                                  model.json/.bin
                                        |
                                        v
                   mod/net/Weights.java  <-- загружает веса
                                        |
                                        v
              BotController (на каждый client-tick):
              StateVector -> окно -> Model.forward -> dyaw/dpitch/fwd/strafe/attack/block
                                        |
                                        v
                       применяет к локальному игроку (камера/движ/атака)
```

## Этапы (по ТЗ)

1. **Imitation Learning** — `train.py train` на записанном/синтетическом датасете.
2. **Эволюционная дошлифовка** — `selfplay.py`: 6 ботов по парам, мутации весов,
   отбор по `(урон нанесённый - полученный)`. Без бэкпропа через reward.
3. **Интеграция** — Fabric-мод инференс на каждый тик + клэмп скорости поворота
   (`MAX_YAW_RT`), чтобы бот не был «снайпером».
