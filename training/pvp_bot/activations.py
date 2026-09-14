"""Активации и их производные. Чистый numpy, никаких ML-фреймворков."""
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def relu_grad(x):
    return (x > 0).astype(np.float64)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2


def sigmoid(x):
    # стабильная версия, чтобы не было переполнения
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    neg = ~pos
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def linear(x):
    return x


def linear_grad(x):
    return np.ones_like(x, dtype=np.float64)


_ACT = {
    "relu": (relu, relu_grad),
    "tanh": (tanh, tanh_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "linear": (linear, linear_grad),
}


def forward(name, x):
    return _ACT[name][0](x)


def grad(name, x):
    return _ACT[name][1](x)
