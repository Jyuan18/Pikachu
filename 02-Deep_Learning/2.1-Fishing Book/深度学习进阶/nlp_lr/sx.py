import os
import pickle
import numpy as np
from tqdm import tqdm


def read_data(file, num=None):
    with open(file, "rb") as f:
        word_2_index, index_2_word = pickle.load(f)
    return word_2_index, index_2_word


class MyDataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class MyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.idx = range(len(self.dataset))

    def __iter__(self):
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= len(self.dataset):
            raise StopIteration
        batch_x, batch_y = [], []
        for _ in range(self.batch_size):
            if self.cursor >= len(self.dataset):
                break
            batch_data = self.dataset[self.cursor]
            batch_x.append(batch_data[0])
            batch_y.append(batch_data[1])
            self.cursor += 1
        return np.array(batch_x), np.array(batch_y)


class Module(object):
    def __init__(self):
        self.params = []
        self.models = []

    def forward(self, *args):
        raise NotImplementedError

    def backward(self, *args):
        raise NotImplementedError

    @property
    def all_params(self):
        for m in self.models:
            self.params.extend(m.all_params)
        return self.params

    def __call__(self, *args):
        return self.forward(*args)

    def __setattr__(self, key, value):
        object.__setattr__(self, key, value)
        if isinstance(value, Parameters):
            self.params.append(value)
        elif isinstance(value, Module):
            self.models.append(value)


class Module1(object):
    def __init__(self):
        self.info = "Module:\n"
        self.params = []

    def __repr__(self):
        return self.info


class Parameters1:
    def __init__(self, value):
        self.value = value
        self.delta = np.zeros(value.shape)


class Linear(Module1):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.info += f"**  Linear"
        self.weight = Parameters1(np.random.normal(size=(input_features, output_features)))
        self.bias = Parameters1(np.zeros((1, output_features)))
        self.params.append(self.weight)
        self.params.append(self.bias)

    def forward(self, x):
        res = x @ self.weight.value + self.bias.value
        self.x = x
        return res

    def backward(self, G):
        self.weight.delta += self.x.T @ G
        self.bias.delta += np.sum(G, axis=0)
        delta_x = G @ self.weight.value.T
        return delta_x







class Parameters:
    def __init__(self,data):
        self.data = data
        self.grad = np.zeros_like(self.data)


class Embedding(Module):
    def __init__(self, w):
        super(Embedding, self).__init__()
        self.w = Parameters(w)
        self.idx = None

    def forward(self, idx):
        self.idx = idx
        out = self.w.data[idx]
        return out

    def backward(self, dout):
        self.w.grad = np.zeros(self.w.data.shape)
        np.add.at(self.w.grad, self.idx, dout)


class TimeEmbedding(Module):
    def __init__(self, w):
        super(TimeEmbedding, self).__init__()
        self.w = Parameters(w)
        self.layers = None

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.w.data.shape
        out = np.empty((N, T, D), dtype='f')
        self.layers = []
        for t in range(T):
            layer = Embedding(self.w.data)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
        return out

    def backward(self, dout):
        N, T, D = dout.shape
        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.w.grad
        self.w.grad += grad


class RNN(Module):
    def __init__(self, wx, wh, b):
        self.w = Parameters(wx)
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


def softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    x = x - max_x
    ex = np.exp(x)
    sum_ex = np.sum(ex, axis=1, keepdims=True)
    result = ex / sum_ex
    result = np.clip(result, 1e-10, 1e10)
    return result


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
    def __init__(self, vocab):
        pass

class ModuleList:
    def __init__(self,layers):
        self.layers = layers

    def forward(self,x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self,G):
        for layer in self.layers[::-1]:
            G = layer.backward(G)

    def __repr__(self):
        info = ""
        for layer in self.layers:
            info += layer.info
            info += "\n"

        return info


class Model:
    def __init__(self):
        self.model_list = ModuleList(
            [
                Linear(784, 256),
                ReLU(),
                Conv2D(256, 10),
                Tanh(),
                Softmax()
            ])

    def forward(self,x,label=None):
        pre = self.model_list.forward(x)

        if label is not None:
            self.label = label
            loss = - np.mean(label * np.log(pre))
            return loss
        else:
            return np.argmax(pre,axis=-1)

    def backward(self):
        self.model_list.backward(self.label)

    def __repr__(self):
        return self.model_list.__repr__()

    def parameters(self):
        all_parameters = []
        for layer in self.model_list.layers:
            all_parameters.extend(layer.params)

        return all_parameters


if __name__ == "__main__":
    
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100
    time_size = 5
    lr = 0.1
    max_epoch = 1
    shuffle = False

    word_2_index, index_2_word = read_data(os.path.join("..", "自然语言处理", "dataset", "ptb.vocab.pkl"))
    corpus = np.load(os.path.join("..", "自然语言处理", "dataset", "ptb.train.npy"))
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1]
    ts = corpus[1:]
    data_size = len(xs)
    print(f"corpus_size:{corpus_size}, vocab_size:{vocab_size}")

    max_iters = data_size // (batch_size * time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0

    rest = len(xs) // time_size
    xs = xs[:rest * time_size].reshape(-1, time_size)
    ts = ts[:rest * time_size].reshape(-1, time_size)
    train_dataset = MyDataset(xs, ts)
    train_dataloader = MyDataLoader(train_dataset, batch_size, shuffle)

    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    opt = SGD(lr)
    opt = MSGD(model.parameters(), lr=lr)

    for epoch in range(max_epoch):
        for batch_x, batch_y in tqdm(train_dataloader):
            loss = model.forward(batch_x, batch_y)
            loss.backward()
            opt.step()
            opt.zero_gard()

        print(loss)
