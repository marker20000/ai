"""Обучение бота: нейроэволюция (два бота дерутся, победитель размножается).

Запуск:
    python train_bot.py
и введи число поколений (эпох). Каждые REPORT_EVERY поколений печатается
отчёт: что улучшилось (победы, убийства, урон) и что ухудшилось (полученный
урон) относительно прошлого отчёта. В конце чемпион сохраняется как
model.bin для клиентского мода.

Можно и без ввода (для фона):
    python train_bot.py --gens 300 --report 100
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np

from pvp_bot.mlp import MLP
from pvp_bot import serialize
from pvp_bot.evolution import mutate, scrub_policy
from sim.arena import run, make_policy
from sim import physics as phys
from selfplay import build_base

# ====== НАСТРОЙКИ (можно менять) ======
DEFAULT_GENS = 300
REPORT_EVERY = 10       # печатать отчёт каждые N поколений
POP = 6                 # число ботов в лиге
T = 150                 # тиков на бой (достаточно для серии ударов, эпоха быстрее)
SIGMA = 0.04            # разброс мутации (крупнее — лучше исследование прицеливания)
SCRUB_MATCHES = 3       # матчей с эталонным ботом для каждого бота (якорь от дрейфа)
SCRUB_W = 3.0           # вес очков против эталона: держим отбор на «реальный бой», а не на дегенератов лиги
EVAL_MATCHES = 25       # матчей для оценки чемпиона на отчёте (стабильнее оценка)
OUT_JSON = "data/selfplay_champion.json"
OUT_BIN = "data/selfplay_champion.bin"
# куда класть model.bin для клиентского мода (@bot):
MOD_MODEL_BIN = r"C:\Users\user2\Desktop\ai bot\fabric-example-mod-26.2\run\config\pvpbot\model.bin"
# ======================================


def evaluate(champ_policy, opp_policy, n, seed0):
    wr = kills = dd = dt = sc = 0.0
    for i in range(n):
        r = run(champ_policy, opp_policy, T=T, seed=seed0 + i * 131, fair=True)
        if r["winner"] == "a":
            wr += 1
        kills += r["kill_a"]
        dd += r["dmg_a"]
        dt += r["taken_a"]
        sc += r["score_a"]
    return dict(win_rate=wr / n, kills=kills / n,
                dmg_dealt=dd / n, dmg_taken=dt / n, score=sc / n)


def fmt_delta(name, cur, prev, better_up=True, nd=2):
    if prev is None:
        return f"{name}: {cur:.{nd}f}"
    d = cur - prev
    if abs(d) < 1e-9:
        tag = "(как есть)"
    elif (d > 0) == better_up:
        tag = f"(+{d:.{nd}f} лучше)"
    else:
        tag = f"({d:.{nd}f} хуже)"
    return f"{name}: {cur:.{nd}f} {tag}"


def report(gen, elapsed, best_score, m, prev):
    print(f"\n=== поколение {gen}  (время {elapsed:.0f}s) ===")
    print(f"  лучший счёт лиги: {best_score:.1f}")
    print("  чемпион против эталонного бота:")
    pw = None if prev is None else prev["win_rate"] * 100
    pk = None if prev is None else prev["kills"]
    pdd = None if prev is None else prev["dmg_dealt"]
    pdt = None if prev is None else prev["dmg_taken"]
    psc = None if prev is None else prev["score"]
    print("   " + fmt_delta("победы %", m["win_rate"] * 100, pw, True, 0))
    print("   " + fmt_delta("убийств/бой", m["kills"], pk, True, 2))
    print("   " + fmt_delta("урон нанесённый", m["dmg_dealt"], pdd, True, 1))
    print("   " + fmt_delta("урон полученный", m["dmg_taken"], pdt, False, 1))
    print("   " + fmt_delta("счёт", m["score"], psc, True, 1))


def train(generations, report_every=REPORT_EVERY, seed=42):
    print(f"Обучаем {generations} поколений, отчёт каждые {report_every}.")

    rng = np.random.default_rng(seed)
    base = build_base(None)          # затравка: imitation на синтетике
    sizes, acts = base.sizes, base.acts
    flat0 = base.get_flat()

    bots = [base.copy()]
    for _ in range(POP - 1):
        m = MLP(sizes, acts)
        m.set_flat(mutate(flat0, 0.03, rng))
        bots.append(m)

    best_flat = flat0.copy()       # лучший реальный боец (по счёту против эталона)
    best_score = float("-inf")     # лучший комбинированный счёт (для отбора)
    best_league = float("-inf")    # лучший чисто лиговый счёт (инфо)
    best_scr = float("-inf")
    prev = None
    t0 = time.time()

    for gen in range(1, generations + 1):
        # Курс прицеливания: на ранних поколениях ворота широкие (достаточно
        # «лицом к врагу»), к концу — узкие (взгляд строго в хитбокс). Это заставляет
        # нейроэволюцию сначала научиться поворачиваться к цели, а потом — держать
        # прицел, иначе поздние поколения не получают награду за урон.
        frac = (gen - 1) / max(1, generations - 1)
        # Курс прицеливания: старт — широкие ворота (достаточно «лицом к врагу»),
        # финал — 0.85 (≈31°), совпадает с ИГРОВЫМ гейтом AIM_COS. Это заставляет
        # нейроэволюцию довести наводку до ПЛОТНОЙ (иначе поздние поколения не получают
        # урон — ворота узкие), вместо «рыхлого» конуса ~53°, из-за которого в игре
        # казалось, что бот бьёт «как попало / хитбокс в 2 раза больше».
        phys.AIM_COS = 0.65 + 0.20 * frac    # 0.65 -> 0.85
        phys.AIM_COS_LO = 0.20 + 0.20 * frac  # 0.20 -> 0.40

        league = [0.0] * POP
        for i in range(POP):
            for j in range(i + 1, POP):
                res = run(make_policy(bots[i]), make_policy(bots[j]), T=T, seed=gen * 31 + i * 7 + j)
                league[i] += res["score_a"]
                league[j] += res["score_b"]
        # якорь от дрейфа: каждый бот ещё бьётся с эталонным ботом (scrub),
        # чтобы отбор шёл не только против своих, но и против реального бойца.
        scores = [0.0] * POP
        scr = [0.0] * POP
        for i in range(POP):
            s = 0.0
            for k in range(SCRUB_MATCHES):
                r = run(make_policy(bots[i]), scrub_policy, T=T, seed=gen * 97 + i * 13 + k)
                # Отбор по РЕАЛЬНОМУ урону по эталону, а не по «просто не проиграть»:
                # иначе эволюция учит убегать/стоять, и бот в игре не бьёт
                # (жалоба «делает только первый удар»). −0.3·полученный — мягкий штраф,
                # чтобы давить, а не просто обмениваться уроном.
                s += r["dmg_a"] - 0.3 * r["taken_a"]
            scr[i] = s
            scores[i] = league[i] + SCRUB_W * s
        order = sorted(range(POP), key=lambda k: -scores[k])
        if scores[order[0]] > best_score:
            best_score = scores[order[0]]
        if league[order[0]] > best_league:
            best_league = league[order[0]]
        # чемпион для экспорта — лучший реальный боец (по счёту против эталона)
        bi = int(np.argmax(scr))
        if scr[bi] > best_scr:
            best_scr = scr[bi]
            best_flat = bots[bi].get_flat().copy()

        # выживают топ-3, остальные — усреднение двух случайных выживших + мутация
        survivors = [bots[order[k]].copy() for k in range(3)]
        new_bots = survivors[:]
        while len(new_bots) < POP:
            a = survivors[rng.integers(0, 3)]
            b = survivors[rng.integers(0, 3)]
            child = MLP(sizes, acts)
            child.set_flat((a.get_flat() + b.get_flat()) / 2.0 + mutate(np.zeros_like(flat0), SIGMA, rng))
            new_bots.append(child)
        bots = new_bots

        if gen % report_every == 0:
            champ = MLP(sizes, acts)
            champ.set_flat(best_flat)
            m = evaluate(make_policy(champ), scrub_policy, EVAL_MATCHES, seed0=gen * 1000)
            report(gen, time.time() - t0, best_league, m, prev)
            prev = m
            # периодически сохраняем лучшего (по нанесённому урону) чемпиона в мод,
            # чтобы принудительное завершение (таймаут) не теряло прогресс.
            try:
                serialize.to_binary(champ, OUT_BIN)
                os.makedirs(os.path.dirname(MOD_MODEL_BIN), exist_ok=True)
                import shutil as _sh
                _sh.copyfile(OUT_BIN, MOD_MODEL_BIN)
            except Exception:
                pass

    champ = MLP(sizes, acts)
    champ.set_flat(best_flat)
    serialize.to_json(champ, OUT_JSON)
    serialize.to_binary(champ, OUT_BIN)
    print(f"\n[готово] чемпион: {OUT_JSON}")
    print(f"[готово] бинарь для мода: {OUT_BIN}")
    try:
        import shutil
        os.makedirs(os.path.dirname(MOD_MODEL_BIN), exist_ok=True)
        shutil.copyfile(OUT_BIN, MOD_MODEL_BIN)
        print(f"[готово] model.bin скопирован в мод: {MOD_MODEL_BIN}")
    except Exception as e:
        print(f"[внимание] не удалось скопировать в мод: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=None, help="число поколений (иначе спросим)")
    ap.add_argument("--report", type=int, default=REPORT_EVERY)
    args = ap.parse_args()
    generations = args.gens
    if generations is None:
        try:
            raw = input(f"Сколько поколений (эпох) обучать? [={DEFAULT_GENS}]: ")
            generations = int(raw) if raw.strip() else DEFAULT_GENS
        except (EOFError, ValueError):
            generations = DEFAULT_GENS
    train(generations, args.report)


if __name__ == "__main__":
    main()
