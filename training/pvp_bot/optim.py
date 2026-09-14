"""Оптимизаторы, реализованные вручную (без библиотек)."""
import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, model, grads_W, grads_b):
        for i in range(len(model.W)):
            model.W[i] -= self.lr * grads_W[i]
            model.b[i] -= self.lr * grads_b[i]


class Adam:
    """Ручная реализация Adam: накопление moving average градиентов."""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.t = 0
        self.mW = self.mb = self.vW = self.vb = None

    def _init(self, model):
        self.mW = [np.zeros_like(w) for w in model.W]
        self.mb = [np.zeros_like(b) for b in model.b]
        self.vW = [np.zeros_like(w) for w in model.W]
        self.vb = [np.zeros_like(b) for b in model.b]

    def step(self, model, grads_W, grads_b):
        if self.mW is None:
            self._init(model)
        self.t += 1
        for i in range(len(model.W)):
            self.mW[i] = self.b1 * self.mW[i] + (1 - self.b1) * grads_W[i]
            self.mb[i] = self.b1 * self.mb[i] + (1 - self.b1) * grads_b[i]
            self.vW[i] = self.b2 * self.vW[i] + (1 - self.b2) * (grads_W[i] ** 2)
            self.vb[i] = self.b2 * self.vb[i] + (1 - self.b2) * (grads_b[i] ** 2)
            mWhat = self.mW[i] / (1 - self.b1 ** self.t)
            mbhat = self.mb[i] / (1 - self.b1 ** self.t)
            vWhat = self.vW[i] / (1 - self.b2 ** self.t)
            vbhat = self.vb[i] / (1 - self.b2 ** self.t)
            model.W[i] -= self.lr * mWhat / (np.sqrt(vWhat) + self.eps)
            model.b[i] -= self.lr * mbhat / (np.sqrt(vbhat) + self.eps)
