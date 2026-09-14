"""Многослойный перцептрон. Forward/backward реализованы вручную.

Соглашение о формах:
  - вход слоя a: shape (n_in, B) для батча или (n_in,) для одного примера
  - веса W[i]:   shape (n_out, n_in), row-major (строка = выходной нейрон)
  - z = W @ a + b[:, None]   (b broadcast по батчу)
  - h = activation(z)

Backpropagation считается по стандартным формулам матричного дифференцирования.
"""
import numpy as np

from .activations import forward as _act_fwd
from .activations import grad as _act_grad
from . import config


class MLP:
    def __init__(self, layer_sizes=None, activations=None, seed=0):
        layer_sizes = list(layer_sizes or config.LAYER_SIZES)
        activations = list(activations or config.ACTIVATIONS)
        assert len(activations) == len(layer_sizes) - 1
        self.sizes = layer_sizes
        self.acts = activations
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            # He-инициализация
            scale = np.sqrt(2.0 / n_in)
            self.W.append(rng.standard_normal((n_out, n_in)) * scale)
            self.b.append(np.zeros(n_out, dtype=np.float64))

    # ---- forward ----
    def forward(self, x):
        """x: (n_in,) или (n_in, B). Возвращает (out, cache)."""
        a = np.asarray(x, dtype=np.float64)
        cache = []
        for i in range(len(self.W)):
            z = self.W[i] @ a + self.b[i][:, None] if a.ndim == 2 else self.W[i] @ a + self.b[i]
            h = _act_fwd(self.acts[i], z)
            cache.append((a.copy(), z, h))
            a = h
        return a, cache

    # ---- backward ----
    def backward(self, dout, cache):
        """dout: градиент по выходу сети, форма как у выхода.
        Возвращает (grads_W, grads_b) — списки градиентов по слоям."""
        batched = dout.ndim == 2
        delta = dout.astype(np.float64)
        gW = [np.zeros_like(w) for w in self.W]
        gb = [np.zeros_like(b) for b in self.b]
        for i in reversed(range(len(self.W))):
            a_prev, z, _h = cache[i]
            g = _act_grad(self.acts[i], z)
            dz = delta * g
            if batched:
                gW[i] = dz @ a_prev.T
                gb[i] = dz.sum(axis=1)
                delta = self.W[i].T @ dz
            else:
                gW[i] = np.outer(dz, a_prev)
                gb[i] = dz
                delta = self.W[i].T @ dz
        return gW, gb

    # ---- инференс с применением sigmoid к logit-каналам ----
    def predict(self, x):
        out, _ = self.forward(x)
        out = out.copy()
        for idx in config.LOGIT:
            out[idx] = _act_fwd("sigmoid", out[idx])
        return out

    # ---- плоский вектор весов (для эволюции) ----
    def get_flat(self):
        parts = []
        for w, b in zip(self.W, self.b):
            parts.append(w.ravel())
            parts.append(b.ravel())
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

    def set_flat(self, flat):
        flat = np.asarray(flat, dtype=np.float64)
        idx = 0
        for i in range(len(self.W)):
            nw = self.W[i].size
            self.W[i] = flat[idx:idx + nw].reshape(self.W[i].shape)
            idx += nw
            nb = self.b[i].size
            self.b[i] = flat[idx:idx + nb]
            idx += nb
        return self

    def copy(self):
        m = MLP(self.sizes, self.acts)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        return m

    def num_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))
