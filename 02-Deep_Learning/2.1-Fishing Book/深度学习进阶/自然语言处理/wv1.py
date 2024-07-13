import collections
import numpy as np
import pickle
import os
from tqdm import tqdm


def read_data(file, num=None):
    with open(file, "rb") as f:
        word_2_index, index_2_word = pickle.load(f)
    return word_2_index, index_2_word


def create_contexts_target(corpus, window_size):
    target = corpus[window_size:-window_size]
    contexts = []
    for i in range(window_size, len(corpus)-window_size):
        cs = []
        for j in range(-window_size, window_size+1):
            if j == 0:
                continue
            cs.append(corpus[j+i])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


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
        self.grads[0][...] = dw
        return None


class EmbeddingDot:
    def __init__(self, w):
        self.embed = Embedding(w)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_w = self.embed.forward(idx)
        out = np.sum(target_w * h, axis=1)
        self.cache = (h, target_w)
        return out

    def backward(self, dout):
        h, target_w = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_w = dout * h
        self.embed.backward(dtarget_w)
        dh = dout * target_w
        return dh


def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y + 1e-7)) / len(t)


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y, self.t = None, None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        self.loss = cross_entropy_error(np.c_[1-self.y, self.y], np.c_[1-self.t, self.t])
        return self.loss

    def backward(self, dout):
        dx = (self.y - self.t) * dout / len(self.t)
        return dx


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)
        for i in range(batch_size):
            p = self.word_p.copy()
            target_idx = target[i]
            p[target_idx] = 0
            p /= p.sum()
            negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        return negative_sample


class NegativeSamplingLoss:
    def __init__(self, w, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(w) for _ in range(sample_size + 1)]
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
        return dh


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        w_in = np.random.normal(size=(V, H))
        w_out = np.random.normal(size=(V, H))

        self.in_layers = []
        for _ in range(2 * window_size):
            layer = Embedding(w_in)
            self.in_layers.append(layer)

        self.out_layers = NegativeSamplingLoss(w_out, corpus, power=0.75, sample_size=5)

        layers = self.in_layers + [self.out_layers]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        self.wordvec = w_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.out_layers.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.out_layers.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i] ** 2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)


if __name__ == "__main__":
    # 设定超参数
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # 读入数据
    word_2_index, index_2_word = read_data(os.path.join("dataset", "ptb.vocab.pkl"))
    corpus = np.load(os.path.join("dataset", "ptb.train.npy"))
    vocab_size = len(word_2_index)
    contexts, target = create_contexts_target(corpus, window_size)

    # 生成模型
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    # trainer = Trainer(model, optimizer)
    #
    # # 训练与评估
    # trainer.fit(contexts, target, max_epoch, batch_size)
    # trainer.plot()
    data_size = len(corpus)
    max_iters = data_size // batch_size
    eval_interval = 20
    total_loss = 0
    loss_count = 0

    for epoch in range(max_epoch):
        for iters in tqdm(range(max_iters)):
            batch_x = contexts[iters*batch_size:(iters+1)*batch_size]
            batch_t = target[iters*batch_size:(iters+1)*batch_size]
            loss = model.forward(batch_x, batch_t)
            model.backward()
            params, grads = model.params, model.grads
            optimizer.update(params, grads)
            total_loss += loss
            loss_count += 1
        print(total_loss)

