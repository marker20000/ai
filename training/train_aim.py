"""Отдельная тренировка АИМА (наводки).

Цель — научить модель ПЛАВНО и устойчиво поворачиваться на игрока без
раскачки pitch (дрейф вверх/вниз), который был при обучении на «мгновенный
захват» (K=1). Использует датасет с мягким усилением (pvp_bot/dataset.AIM_GAIN)
и геометрию «глаз vs центр хитбокса», совпадающую с модом.

Пайплайн переиспользует нейроэволюцию из train_bot.run (база уже обучена
демпфированному аиму, эволюция с curriculum-воротами AIM_COS докручивает бой),
и кладёт итоговый model.bin туда, откуда его читает клиентский мод.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import train_bot


def main():
    # база строится внутри train (build_base) — теперь с демпфированным аимом.
    # train() сам сохраняет чемпиона в data/selfplay_champion.bin и копирует
    # его в model.bin мода (MOD_MODEL_BIN), поэтому лишних копий не нужно.
    train_bot.train(generations=40, report_every=10, seed=7)
    print(f"[train_aim] model.bin в моде: {train_bot.MOD_MODEL_BIN}")


if __name__ == "__main__":
    main()
