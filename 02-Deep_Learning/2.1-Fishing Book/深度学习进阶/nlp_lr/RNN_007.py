import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x


class SGD:
    '''
    随机梯度下降法（Stochastic Gradient Descent）
    '''

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class RNN:
    def __init__(self, wx, wh, b):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        """
        x: NxD, wx: DxH
        h: NxH, wh: HxH
        b: 1xH
        """
        wx, wh, b = self.params
        t = np.dot(x, wx) + np.dot(h_prev, wh) + b
        h_next = np.tanh(t)
        self.cache = (h_prev, h_next, x)
        return h_next

    def backward(self, dh_next):
        wx, wh, b = self.params
        h_prev, h_next, x = self.cache
        dt = dh_next * (1 - h_next ** 2)
        dwx = x.T @ dt
        dx = dt @ wx.T
        dh_prev = dt @ wh.T
        dwh = h_prev.T @ dt
        db = np.sum(dt, axis=0)
        self.grads[0][...] = dwx
        self.grads[1][...] = dwh
        self.grads[2][...] = db
        return dx, dh_prev


class TimeRNN:
    def __init__(self, wx, wh, b, stateful):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx), np.zeros_like(wh), np.zeros_like(b)]
        self.layers = None
        self.stateful = stateful
        self.h, self.dh = None, None

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None

    def forward(self, xs):
        """
        xs: NxTxD  wx: DxH
        hs: NxTxH  wh: HxH
        self.h: NxH
        b: 1xH
        """
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
            for i, j in enumerate(layer.grads):
                grads[i] += j
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs


class LSTM:
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


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        np.add.at(dW, self.idx, dout)
        return None


class TimeEmbedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        N, T, D = x.shape
        W, b = self.params

        rx = x.reshape(N*T, -1)
        out = np.dot(rx, W) + b
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N*T, -1)
        rx = x.reshape(N*T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 在监督标签为one-hot向量的情况下
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 按批次大小和时序大小进行整理（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # 与ignore_label相应的数据将损失设为0
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # 与ignore_label相应的数据将梯度设为0

        dx = dx.reshape((N, T, V))

        return dx


class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        # 初始化权重
        embed_w = (rn(V, D) / 100).astype('f')
        rnn_wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_w = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # 生成层
        self.layers = [
            TimeEmbedding(embed_w),
            TimeRNN(rnn_wx, rnn_wh, rnn_b, stateful=True),
            TimeAffine(affine_w, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]
        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, tx):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, tx)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()


def read_data(file, num=None):
    with open(file, "rb") as f:
        word_2_index, index_2_word = pickle.load(f)
    return word_2_index, index_2_word


if __name__ == "__main__":
    # 设定超参数
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5  # Truncated BPTT的时间跨度大小
    lr = 0.1
    max_epoch = 100

    # 读入训练数据（缩小了数据集）
    # corpus, word_to_id, id_to_word = ptb.load_data('train')
    word_to_id, id_to_word = read_data(os.path.join("..", "自然语言处理", "dataset", "ptb.vocab.pkl"))
    corpus = np.load(os.path.join("..", "自然语言处理", "dataset", "ptb.train.npy"))
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1]  # 输入
    ts = corpus[1:]  # 输出（监督标签）
    data_size = len(xs)
    print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

    # 学习用的参数
    max_iters = data_size // (batch_size * time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []

    # 生成模型
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    # 计算读入mini-batch的各笔样本数据的开始位置
    jump = (corpus_size - 1) // batch_size
    offsets = [i * jump for i in range(batch_size)]

    for epoch in range(max_epoch):
        for iter in range(max_iters):
            # 获取mini-batch
            batch_x = np.empty((batch_size, time_size), dtype='i')
            batch_t = np.empty((batch_size, time_size), dtype='i')
            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    batch_x[i, t] = xs[(offset + time_idx) % data_size]
                    batch_t[i, t] = ts[(offset + time_idx) % data_size]
                time_idx += 1

            # 计算梯度，更新参数
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # 各个epoch的困惑度评价
        ppl = np.exp(total_loss / loss_count)
        print('| epoch %d | perplexity %.2f'
              % (epoch + 1, ppl))
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0

    # 绘制图形
    x = np.arange(len(ppl_list))
    plt.plot(x, ppl_list, label='train')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()








