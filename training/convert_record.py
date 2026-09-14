"""Конвертер записей мода (JSONL) в датасет для обучения (npz).

Каждая сессия @start/@stop сохраняется модом в отдельный файл
record_NNNN.jsonl. Каждый такой файл — один эпизод (от удара до удара,
или вся дуэль целиком, если сегментацию делать негде). Здесь каждый файл
считается одним эпизодом.

Использование:
  python convert_record.py --src ../mod/config/pvpbot --out data/dataset.npz
  python convert_record.py   # ищет config/pvpbot/record_*.jsonl и пишет data/dataset.npz
"""
import argparse
import glob
import json
import os

import numpy as np


MAX_YAW_RT = 22.5  # совпадает с config.MAX_YAW_RT: бот может повернуть не больше этого за тик

import re
# Целое (возможно со знаком) — используется как целая часть десятичной дроби
# в сломанном формате (ru_RU: «0,0689» вместо «0.0689»).
_INT = re.compile(r"^-?\d+$")

# Дробная часть (после десятичной запятой). Допускает научную нотацию: для очень
# маленьких значений String.format("%.5g", ...) пишет «3,5789e-05» (на ru_RU),
# что после split по запятой даёт токены «3» и «5789e-05».
_FRAC = re.compile(r"^\d+([eE][-+]?\d+)?$")

# Значения, которые могут быть записаны ЦЕЛИКОМ как целое (без дробной части).
# Нормализованные признаки лежат в [-1,1] (dx/dy/dz могут быть чуть больше, до ~2),
# поэтому «чистое» целое вне этого диапазона (напр. 9) — всегда артефакт и
# трактуется как дробная часть предыдущего числа (0,9 = 0.9, а не [0, 9]).
_STANDALONE = {"-2": -2.0, "-1": -1.0, "-0": 0.0, "0": 0.0, "1": 1.0, "2": 2.0}


def _parse_arr(s, k):
    """Парс одного массива, устойчивый к локализованным ЗАПЯТЫМ вместо точки
    (баг Recorder.java на ru_RU: писал «-0,0689»).

    ДП ищет разбиение токенов РОВНО на k чисел, где каждое число — либо
    «чистое» целое из _STANDALONE, либо пара «целое,дробные_цифры». Это
    однозначно восстанавливает формат: смежные флаги 0/1 читаются как
    отдельные числа, а 0,0689 — как десятичная дробь. Среди разбиений
    выбирается то, где меньше значений вне диапазона [-1.5, 1.5] (отсекает
    артефакты вроде 9.0). Возвращает список из k float или None."""
    inner = s.strip()
    if inner.startswith("["):
        inner = inner[1:]
    if inner.endswith("]"):
        inner = inner[:-1]
    toks = inner.split(",")
    if not toks or toks == [""]:
        return None
    N = len(toks)
    # reaches[i][count] = (prev_i, span, penalty)
    reaches = [dict() for _ in range(N + 1)]
    reaches[0][0] = (-1, None, 0)
    for i in range(N + 1):
        for c, (prev, span, pen) in list(reaches[i].items()):
            # вариант S: самостоятельное целое
            if i < N and toks[i] in _STANDALONE:
                v = _STANDALONE[toks[i]]
                np_ = pen + (1 if abs(v) > 1.5 else 0)
                j = i + 1
                cur = reaches[j].get(c + 1)
                if cur is None or np_ < cur[2]:
                    reaches[j][c + 1] = (i, ("S", toks[i]), np_)
            # вариант D: целая часть + дробная (десятичная запятая)
            if i < N and _INT.match(toks[i]) and i + 1 < N and _FRAC.match(toks[i + 1]) and toks[i + 1] != "0":
                v = float(toks[i] + "." + toks[i + 1])
                np_ = pen + (1 if abs(v) > 1.5 else 0)
                j = i + 2
                cur = reaches[j].get(c + 1)
                if cur is None or np_ < cur[2]:
                    reaches[j][c + 1] = (i, ("D", toks[i], toks[i + 1]), np_)
    if k not in reaches[N]:
        return None
    vals = []
    i, c = N, k
    while i > 0:
        prev, span, _ = reaches[i][c]
        if span[0] == "S":
            vals.append(_STANDALONE[span[1]])
            c -= 1
        else:
            vals.append(float(span[1] + "." + span[2]))
            c -= 1
        i = prev
    vals.reverse()
    return vals


def _parse_line(line):
    try:
        obj = json.loads(line)
        if len(obj["state"]) == 27 and len(obj["target"]) == 6:
            return obj
        return None
    except ValueError:
        # старые записи с запятыми-разделителями: вытаскиваем массивы вручную
        si = line.find('state":[') + len('state":[')
        ti = line.find('target":[') + len('target":[')
        se = line.find("]", si)
        te = line.find("]", ti)
        s = _parse_arr(line[si:se + 1], 27)
        t = _parse_arr(line[ti:te + 1], 6)
        if s is None or t is None:
            return None
        return {"state": s, "target": t}


def load_episodes(src):
    episodes = []
    files = sorted(glob.glob(os.path.join(src, "record_*.jsonl")))
    for f in files:
        states, targets = [], []
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = _parse_line(line)
                if row is None:
                    print(f"[convert] WARN {os.path.basename(f)}: пропущена строка "
                          f"(не удалось разобрать)")
                    continue
                s, t = row["state"], row["target"]
                # пропускаем битые строки (локальные запятые иногда неоднозначны)
                if len(s) != 27 or len(t) != 6:
                    print(f"[convert] WARN {os.path.basename(f)}: пропущена строка "
                          f"(state={len(s)} target={len(t)}, ожидалось 27/6)")
                    continue
                states.append(s)
                targets.append(t)
        if not states:
            continue
        S = np.array(states, dtype=np.float64)
        T = np.array(targets, dtype=np.float64)
        # цели прицеливания не могут превышать MAX_YAW_RT за тик (бот клэмпит выход),
        # поэтому обрезаем выбросы (бывали -159° из-за переползания через ±180).
        T[:, 0] = np.clip(T[:, 0], -MAX_YAW_RT, MAX_YAW_RT)
        T[:, 1] = np.clip(T[:, 1], -MAX_YAW_RT, MAX_YAW_RT)
        episodes.append((S, T))
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None,
                    help="папка с record_*.jsonl (по умолчанию config/pvpbot)")
    ap.add_argument("--out", default="data/dataset.npz")
    args = ap.parse_args()

    src = args.src
    if src is None:
        candidates = [
            os.path.join("config", "pvpbot"),
            os.path.join(os.path.dirname(__file__), "..", "mod", "config", "pvpbot"),
            os.path.join("..", "mod", "config", "pvpbot"),
            os.path.join(os.path.dirname(__file__), "..", "fabric-example-mod-26.2", "run", "config", "pvpbot"),
        ]
        for c in candidates:
            if glob.glob(os.path.join(c, "record_*.jsonl")):
                src = c
                break
    if src is None:
        src = "config/pvpbot"

    eps = load_episodes(src)
    if not eps:
        print(f"[convert] не найдено record_*.jsonl в {src}")
        return

    states = np.array([e[0] for e in eps], dtype=object)
    targets = np.array([e[1] for e in eps], dtype=object)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, states=states, targets=targets)
    ticks = sum(e[0].shape[0] for e in eps)
    print(f"[convert] эпизодов={len(eps)} тиков={ticks} -> {args.out}")


if __name__ == "__main__":
    main()
