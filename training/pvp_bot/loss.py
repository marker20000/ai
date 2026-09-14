"""Функция потерь: MSE для непрерывных величин + BCE для флагов атаки/блока.

pred, target — формы (D,) или (B, D), где D = TARGET_DIM.
Каналы CONTINUOUS = [dyaw, dpitch, fwd, strafe] (linear-выход, MSE).
Каналы LOGIT      = [attack, block] (logits, sigmoid+BCE).
"""
import numpy as np

from .activations import sigmoid
from . import config


def _to_batch(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
        target = target.reshape(1, -1)
    return pred, target


def loss_and_grad(pred, target, cont_weight=1.0, bce_weight=1.0, pos_weight=None, sample_weight=None):
    pred, target = _to_batch(pred, target)
    B = pred.shape[0]
    dout = np.zeros_like(pred)

    if sample_weight is None:
        sw = np.ones(B)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64)
        if sw.sum() > 0:
            sw = sw * B / sw.sum()  # нормируем к B, чтобы масштаб потерь не поплыл
    swc = sw[:, None]

    cont = config.CONTINUOUS
    diff = pred[:, cont] - target[:, cont]
    mse = np.mean(sw * np.sum(diff ** 2, axis=1))
    dout[:, cont] = (2.0 / B) * diff * cont_weight * swc

    logit = config.LOGIT
    if pos_weight is None:
        pw = np.ones(len(logit))
    else:
        pw = np.asarray(pos_weight, dtype=np.float64)[logit]
    p = sigmoid(pred[:, logit])
    p = np.clip(p, 1e-7, 1.0 - 1e-7)
    tgt = target[:, logit]
    # взвешенный BCE: положительные примеры (атака/блок) редки в демо, без
    # взвешивания сеть вырождается в константу «никогда не атакуй».
    bce_per = pw * (tgt * np.log(p) + (1.0 - tgt) * np.log(1.0 - p))  # (B, nlogit)
    bce = -np.mean(swc * bce_per)
    dout[:, logit] = ((1.0 - tgt) * p - pw * tgt * (1.0 - p)) / B * bce_weight * swc

    total = cont_weight * mse + bce_weight * bce
    return total, dout


def evaluate(pred, target, cont_weight=1.0, bce_weight=1.0, pos_weight=None, sample_weight=None):
    loss, _ = loss_and_grad(pred, target, cont_weight, bce_weight, pos_weight, sample_weight)
    return loss
