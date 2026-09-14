# Папка с данными и весами

- `dataset.npz` — датасет для imitation learning (`states`: (E,T,27), `targets`: (E,T,6)).
  Собирается рекордером из мода (`Recorder.java`) или генерируется симуляцией.
- `model.json` / `model.bin` — обученные веса (формат см. `docs/weight_format.md`).
- `champion.json` — результат эволюции (`train.py evolve`).
- `selfplay_champion.json` — результат self-play 6 ботов (`selfplay.py`).

Формат одной записи дуэли (JSONL от рекордера):
```json
{"tick": 123, "state": [/* 27 норм. признаков */], "target": [/* 6 целей */]}
```
Порядок признаков — строго по `docs/feature_spec.md`.
