"""CLI для обучения и экспорта модели.

Режимы:
  selftest            — прогнать весь конвейер на синтетике (без игры)
  train --data X.npz  — imitation learning на датасете
  evolve              — эволюционная дошлифовка поверх базовой модели
  export              — конвертировать веса json<->bin
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from pvp_bot.mlp import MLP
from pvp_bot import dataset, trainer, serialize, evolution


def _ensure_data_dir():
    os.makedirs(os.path.join(os.path.dirname(__file__), "data"), exist_ok=True)


def selftest():
    _ensure_data_dir()
    print("[selftest] генерация синтетики...")
    S, T = dataset.make_toy_dataset(n_episodes=40, T=60, seed=1)
    X, Y = dataset.dataset_windows(S, T)
    print(f"[selftest] X={X.shape} Y={Y.shape} params={MLP().num_params()}")

    print("[selftest] обучение imitation (Adam)...")
    model = MLP(seed=3)
    trainer.train(model, X, Y, epochs=30, batch_size=128, lr=0.002, verbose=True)

    print("[selftest] сохранение + round-trip проверка формата...")
    serialize.to_json(model, "data/toy_model.json")
    # JSON хранит float64 -> точное совпадение
    m_json = serialize.from_json("data/toy_model.json")
    assert m_json.sizes == model.sizes and m_json.acts == model.acts
    assert (m_json.get_flat() == model.get_flat()).all(), "JSON round-trip не точен"
    # бинарь хранит float32 -> допустима погрешность, проверяем allclose + предсказания
    serialize.to_binary(model, "data/toy_model.bin")
    m_bin = serialize.from_binary("data/toy_model.bin")
    assert m_bin.sizes == model.sizes and m_bin.acts == model.acts
    assert np.allclose(m_bin.get_flat(), model.get_flat(), atol=1e-2), "бинарь слишком неточен"
    assert np.allclose(m_bin.predict(X[:5].T), model.predict(X[:5].T), atol=1e-2)
    print("[selftest] OK: модель обучена, веса конвертируются туда-обратно (JSON точно, бинарь allclose)")

    print("[selftest] эволюционная дошлифовка (несколько поколений)...")
    champ = evolution.evolve(model, generations=6, pop=8, matches=2, T=160, seed=5)
    serialize.to_json(champ, "data/champion.json")
    print("[selftest] champion сохранён в data/champion.json")


def train(data, epochs, out, lr):
    _ensure_data_dir()
    S, Tgt = dataset.load_npz(data)
    X, Y = dataset.dataset_windows(S, Tgt)
    print(f"загружено X={X.shape} Y={Y.shape}")
    model = MLP(seed=7)
    trainer.train(model, X, Y, epochs=epochs, lr=lr, verbose=True)
    serialize.to_json(model, out)
    serialize.to_binary(model, out.replace(".json", ".bin"))
    print(f"модель сохранена: {out}")


def evolve(base, out, generations, pop):
    _ensure_data_dir()
    if base and os.path.exists(base):
        model = serialize.from_json(base) if base.endswith(".json") else serialize.from_binary(base)
    else:
        S, Tgt = dataset.make_toy_dataset(n_episodes=40, T=60, seed=1)
        X, Y = dataset.dataset_windows(S, Tgt)
        model = MLP(seed=3)
        trainer.train(model, X, Y, epochs=20, lr=0.002, verbose=False)
    champ = evolution.evolve(model, generations=generations, pop=pop, matches=3, T=200, seed=11)
    serialize.to_json(champ, out)
    print(f"чемпион эволюции сохранён: {out}")


def main():
    ap = argparse.ArgumentParser(description="PvP-bot training CLI")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("selftest")
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--out", default="data/model.json")
    p.add_argument("--lr", type=float, default=0.001)
    p = sub.add_parser("evolve")
    p.add_argument("--base", default=None)
    p.add_argument("--out", default="data/champion.json")
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--pop", type=int, default=12)
    p = sub.add_parser("export")
    p.add_argument("inp")
    p.add_argument("outp")

    args = ap.parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "train":
        train(args.data, args.epochs, args.out, args.lr)
    elif args.cmd == "evolve":
        evolve(args.base, args.out, args.generations, args.pop)
    elif args.cmd == "export":
        serialize.convert(args.inp, args.outp)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
