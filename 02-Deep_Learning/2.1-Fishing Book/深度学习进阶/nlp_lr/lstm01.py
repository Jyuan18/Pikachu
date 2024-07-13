import numpy as np


class LSTM:
    def __init__(self, wx, wh, b):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        wx, wh, b = self.params
        N, H = h_prev.shape


        A = x @ wx + h_prev @ wh + b
        f = sigmoid(A[:, :H])
        g = np.tanh(A[:, H:2*H])
        i = sigmoid(A[:, 2*H:3*H])
        o = sigmoid(A[:, 3*H:4*H])

        c_next = c_prev * f + g * i
        h_next = o * np.tanh(c_next)
        self.cache = (x, h_prev, c_prev, f, g, i, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        wx, wh, b = self.params
        x, h_prev, c_prev, f, g, i, o, c_next = self.cache
        tanh_c_next = np.tanh(c_next)

        # do = dh_next * tanh_c_next
        ds = dh_next * o * (1 - tanh_c_next ** 2) + dc_next

        dc_prev = ds * f

        df = ds * c_prev
        dg = ds * i
        di = ds * g
        d0 = dh_next * tanh_c_next

        dA = np.hstack((df, dg, di, do))

        dwh = h_prev.T @ dA
        dwx = x.T @ dA
        db = np.sum(dA, axis=0)

        self.grads[0][...] = dwx
        self.grads[1][...] = dwh
        self.grads[2][...] = db

        dx = dA @ wx.T
        dh_prev = dA.T @ wh.T
        return dx, dc_prev, dh_prev

class lstm:
    def __init__(self, wx, wh, b):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.cacahe = None

    def forward(self, x, h_prev, c_prev):
        wx, wh, b = self.params
        N, H = h_prev.shape
        A = np.dot(x, wx) + np.dot(h_prev, wh) + b
        f = sigmoid(A[:, :H])
        g = np.tanh(A[:, H:2*H])
        i = sigmoid(A[:, 2*H:3*H])
        o = sigmoid(A[:, 3*H:])
        c_next = c_prev * f + g * i
        h_next = o * np.tanh(c_next)
        self.cache = (x, h_prev, c_prev, f, g, i, o, c_next, h_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        wx, wh, b = self.params
        x, h_prev, c_prev, f, g, i, o, c_next, h_next = self.cache
        tanh_c_next = np.tanh(c_next)
        ds = dc_next + dh_next * o * (1 - tanh_c_next ** 2)
        dc_prev = ds * f

        df = ds * c_prev
        dg = ds * i
        di = ds * g
        do = dh_next * tanh_c_next

        dA = np.hstack((df, dg, di, do))
        dwx = x.T @ dA
        dx = dA @ wx.T
        dwh = h_prev.T @ dA
        dh_prev = dA @ wh.T
        db = np.sum(dA, axis=0)
        self.grads[0][...] = dwx
        self.grads[1][...] = dwh
        self.grads[2][...] = db
        return dx, dh_prev, dc_prev

class TimeLSTM:
    def __init__(self, wx, wh, b, stateful=False):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.c, self.dh = None, None, None
        self.stateful = stateful

    def forward(self, xs):
        wx, wh, b = self.params
        N, T, D = xs.shape
        H = wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        wx, wh, b = self.params
        N, T, H = dhs.shape
        D = wx.shape[0]
        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None

class GRU:
    def __init__(self, wx, wh, b):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, h_prev, x):
        wx, wh, b = self.params
        H = wh.shape[0]
        wxz, wxr, wxh = wx[:, :H], wx[:, H:2*H], wx[:, 2*H:]
        whz, whr, whh = wh[:, :H], wh[:, H:2*H], wh[:, 2*H:]
        bz, br, bh = b[:H], b[H:2*H], b[2*H:]
        z = sigmoid(np.dot(x, wxz) + np.dot(h_prev, whz) + bz)
        r = sigmoid(np.dot(x, wxr) + np.dot(h_prev, whr) + br)
        h_hat = np.tanh(np.dot(x, wxh) + np.dot(r*h_prev, whh) + bh)
        h_next = (1-z)*h_prev + z*h_hat
        self.cache = (x, h_prev, z, r, h_hat)
        return h_next

    def backward(self, dh_next):
        wx, wh, b = self.params
        H = wh.shape[0]
        wxz, wxr, wxh = wx[:, :H], wx[:, H:2*H], wx[:, 2*H:]
        whz, whr, whh = wh[:, :H], wh[:, H:2*H], wh[:, 2*H:]
        x, h_prev, z, r, h_hat = self.cache
        dh_hat = dh_next * z
        dh_prev = dh_next * (1-z)

        dt = dh_hat * (1 - h_hat ** 2)
        dbh = np.sum(dt, axis=0)
        dwhh = np.dot((r*h_prev).T, dt)
        dhr = np.dot(dt, whh.T)
        dwxh = np.dot(x.T, dt)
        dx = np.dot(dt, wxh.T)
        dh_prev += r * dhr

        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1 - z)
        dbz = np.sum(dt, axis=0)
        dwhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, whz.T)
        dwxz = np.dot(x.T, dt)
        dx += np.dot(dt, wxz.T)

        dr = dhr * h_prev
        dt = dr * r * (1-r)
        dbr = np.sum(dt, axis=0)
        dwhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, whr.T)
        dwxr = np.dot(x.T, dt)
        dx += np.dot(dt, wxr.T)

        self.dwx = np.hstack((dwxz, dwxr, dwxh))
        self.dwh = np.hstack((dwhz, dwhr, dwhh))
        self.db = np.hstack((dbz, dbr, dbh))
        self.grads[0][...] = self.dwx
        self.grads[1][...] = self.dwh
        self.grads[2][...] = self.db
        return dx, dh_prev











