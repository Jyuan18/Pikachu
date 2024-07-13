import numpy as np

class RNN:
    def __init__(self, wx, wh, b):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]

    def forward(self, x, h_prev):
        wx, wh ,b = self.params
        t = x @ wx + h_prev @ wh + b
        h_next = np.tanh(t)
        self.cache = (h_next, h_prev, x)
        return h_next

    def backward(self, dh_next):
        wx, wh, b = self.params
        h_next, h_prev, x = self.cache
        dt = dh_next * (1 - h_next ** 2)
        dh_prev = dt @ wh.T
        dwh = h_prev.T @ dt
        dwx = x.T @ dt
        dx = dt @ wx.T
        db = np.sum(dt, axis=0)
        self.grads[0][...] = dwx
        self.grads[1][...] = dwh
        self.grads[2][...] = db
        return dx, dh_prev

class TimeRNN:
    def __init__(self, wx, wh, b, stateful=False):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.stateful = stateful
        self.layers = None
        self.h, self.dh = None, None

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        wx, wh, b = self.params
        N, T, D = xs.shape
        D, H = wx.shape
        hs = np.empty((N, T, H), dtype='f')
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        self.layers = []
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        wx, wh, b = self.params
        N, T, H = dhs.shape
        D, H = wx.shape
        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

class Embedding:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.idx = None

    def forward(self, idx):
        w, = self.params
        self.idx = idx
        out = w[idx]
        return out

    def backward(self, dout):
        dw, = self.grads
        dw[...] = 0
        np.add.at(dw, self.idx, dout)
        return None

class TimeEmbedding:
    def __init__(self, w):
        self.params = [w]
        self.grads = [np.zeros_like(w)]
        self.layers = None
        self.w = w

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.w.shape
        out = np.empty((N, T, D), dtype='f')
        self.layers = []
        for t in range(T):
            layer = Embedding(self.w)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out

    def backwars(self, dout):
        N, T, D = dout.shape
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]
        self.grads[0][...] = grad
        return None



