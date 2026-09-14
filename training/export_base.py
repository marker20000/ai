"""Выгрузить обученную imitation-базу в model.bin для клиентского мода.

Используется, когда нужно быстро «починить» поведение в игре (например,
после смены конвенции наклона/pitch), не прогоняя полную эволюцию.
База уже умеет наводиться (mean_aim ~0.9 в arena) и бить в зоне досягаемости.
"""
import os
import sys
import shutil

sys.path.insert(0, os.path.dirname(__file__))

from selfplay import build_base
from pvp_bot import serialize

MOD_MODEL_BIN = r"C:\Users\user2\Desktop\ai bot\fabric-example-mod-26.2\run\config\pvpbot\model.bin"
OUT_BIN = "data/selfplay_champion.bin"


def main():
    print("[export_base] обучаем imitation-базу на синтетике...")
    m = build_base(None)
    os.makedirs("data", exist_ok=True)
    serialize.to_binary(m, OUT_BIN)
    print(f"[export_base] бинарь базы: {OUT_BIN}")

    os.makedirs(os.path.dirname(MOD_MODEL_BIN), exist_ok=True)
    shutil.copyfile(OUT_BIN, MOD_MODEL_BIN)
    print(f"[export_base] model.bin для мода: {MOD_MODEL_BIN}")


if __name__ == "__main__":
    main()
