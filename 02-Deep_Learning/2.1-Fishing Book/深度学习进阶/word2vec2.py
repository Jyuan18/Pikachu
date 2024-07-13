import collections
import numpy as np
import os
import pickle
import time
import matplotlib.pyplot as plt


# 读取数据
def read_data(file):
    with open(file, "rb") as f:
        word_2_index, index_2_word = pickle.load(f)
    return word_2_index, index_2_word


# 预处理输入和标签
def create_contexts_target(corpus, window_size=1):
    target = corpus[window_size:-window_size]
    contexts =[]
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size+1):
            if t == 0:
                continue
            cs.append(corpus[idx+t])
        contexts.append(cs)
    return np.array(contexts), np.array(target)


# 模型前置准备
# 交叉熵
def cross_entropy_error(y, t):
    # y是神经网络的输出, t为标签
    # 检查 y 是否为一维数组，如果是，将 t 和 y 转换为二维数组
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)
    # 如果监督标签 t 是 one-hot 向量，将其转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
    # 计算批次大小 batch_size
    batch_size = y.shape[0]
    # 使用 numpy 数组索引和 np.sum 函数计算交叉熵损失
    # 首先，使用 y[np.arange(batch_size), t] 选择每个样本的正确预测概率
    # 然后，取对数并求和
    # 最后，除以 batch_size 得到平均损失值
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

# 损失函数 sigmoidwithloss
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []   # 该层没有参数，因此 params 和 grads 都是空列表
        self.loss = None    # 记录该层的损失值
        self.y = None       # sigmoid 函数的输出
        self.t = None       # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))    # 计算 sigmoid 函数的输出
        # 将 sigmoid 函数的输出作为输入，计算交叉熵损失
        # np.c_[1 - self.y, self.y] 是 numpy 的一种数组拼接方式，用于将两个二维数组按列拼接起来。
        # 具体来说，它将两个二维数组按列方向拼接成一个新的二维数组，其中第一个数组的列数为 0，
        # 元素为 1 减去 sigmoid 函数的输出；第二个数组的列数为 1，元素为 sigmoid 函数的输出。
        # 因此，np.c_[1 - self.y, self.y] 的形状为 (batch_size, 2)
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        # 根据反向传播算法，计算该层的梯度
        dx = (self.y - self.t) * dout / batch_size
        return dx


# embedding 层
class Embedding:
    def __init__(self, w):
        self.params = [w]               # 该层的参数是一个词向量矩阵 w
        self.grads = [np.zeros_like(w)] # 该层的梯度是一个和 w 形状相同的零矩阵
        self.idx = None                 # 该层的输入是一个整数序列，idx 记录了该序列

    def forward(self, idx):
        w, = self.params                 # 获取参数 w
        self.idx = idx                    # 记录输入 idx
        out = w[idx]                      # 将输入序列中的整数转换为对应的词向量
        return out

    def backward(self, dout):
        dw, = self.grads                # 获取梯度 dw
        dw[...] = 0                     # 将梯度矩阵清零
        np.add.at(dw, self.idx, dout)   # 根据输入序列中的整数索引，累加词向量的梯度
        return None


# embedding dot层
class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)         # Embedding 层用于将整数序列转换为词向量矩阵
        self.params = self.embed.params  # 该层的参数和梯度与 Embedding 层相同
        self.grads = self.embed.grads
        self.cache = None                 # 缓存 h 和 target_W，用于计算梯度

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)  # 将整数序列 idx 转换为词向量矩阵
        out = np.sum(target_W * h, axis=1)  # 计算输入向量 h 与词向量矩阵的点积
        self.cache = (h, target_W)          # 缓存 h 和 target_W，用于计算梯度
        return out

    def backward(self, dout):
        h, target_W = self.cache                  # 获取缓存的 h 和 target_W
        dout = dout.reshape(dout.shape[0], 1)      # 将 dout 变形为 (batch_size, 1) 的形状

        dtarget_W = dout * h                       # 计算词向量矩阵的梯度
        self.embed.backward(dtarget_W)             # 将梯度传递给 Embedding 层
        dh = dout * target_W                       # 计算输入向量 h 的梯度
        return dh


# UnigramSampler 类，用于对语料库进行负采样
class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        # 初始化采样器，保存采样大小和词汇表大小
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        # 统计语料库中每个单词出现的次数，并计算每个单词出现的概率
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
            negative_sample[i,:] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        return negative_sample


# 计算负采样损失
"""
在初始化过程中，先使用语料库和平滑因子 power 初始化 UnigramSampler，然后初始化多个 SigmoidWithLoss 和 EmbeddingDot 层，
并将它们的参数和梯度存储在 self.params 和 self.grads 中。在前向传播过程中，先使用 UnigramSampler 获取负样本，然后对正样本和负样本
分别进行正向传播和损失计算。在正例的正向传播中，使用 EmbeddingDot 层计算正样本的得分，并使用 SigmoidWithLoss 层计算正样本的损失。
在负例的正向传播中，对于每个负样本，使用 EmbeddingDot 层计算负样本的得分，并使用 SigmoidWithLoss 层计算负样本的损失。
将正例和负例的损失相加，得到最终的损失值。在反向传播过程中，先对所有的 SigmoidWithLoss 层进行反向传播，得到它们的输入梯度，
然后对所有的 EmbeddingDot 层进行反向传播，得到它们的输入梯度，并将它们相加，得到最终的输入梯度。最后返回输入梯度 dh。
"""
class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例的正向传播
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 负例的正向传播
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

# Adam优化器
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

# CBOW模型
"""
该类实现了 CBOW 模型，用于进行词向量的训练。在初始化过程中，根据词汇表大小 vocab_size、隐藏层大小 hidden_size 和
窗口大小 window_size 初始化输入层的权重 W_in 和输出层的权重 W_out。然后使用 Embedding 层初始化输入层，
使用 NegativeSamplingLoss 层初始化输出层，并将它们存储在 self.in_layers 和 self.ns_loss 中。
最后将所有的权重和梯度整理到列表中，并将单词的分布式表示 W_in 设置为成员变量 self.word_vecs。
在前向传播过程中，首先将上下文单词的分布式表示通过输入层转换为隐藏层的表示，然后求取平均值作为最终的隐藏层表示 h。
接着将 h 和目标单词的标签 target 传递给输出层进行计算，得到损失值 loss。在反向传播过程中，先对输出层进行反向传播，
得到它们的输入梯度，然后将梯度值除以输入层的个数，以便将梯度平均到每个输入层中。接着对每个输入层进行反向传播，
得到它们的输入梯度，最后返回 None。
"""
class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')  # 10000x100
        W_out = 0.01 * np.random.randn(V, H).astype('f')

        # 生成层
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)  # 使用Embedding层
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 将所有的权重和梯度整理到列表中
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None


"""这是一个梯度裁剪函数，用于限制梯度的范数，避免梯度爆炸问题。具体来说，输入参数 grads 是一个包含各个参数的梯度的列表，max_norm 是裁剪后的梯度范数的最大值。

首先，该函数计算梯度的范数，即将各个梯度平方和的平方根作为梯度的范数。然后，计算一个比率，它将 max_norm 与梯度范数之和的比值作为裁剪的比率。如果该比率小于 1，则说明梯度范数超过了 max_norm，需要进行梯度裁剪。具体地，将每个梯度乘以裁剪比率，从而使梯度范数不超过 max_norm。
"""
def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


"""
这是一个用于整合共享权重的函数。在神经网络中，有时会有一些层共享同样的权重，例如在卷积神经网络中，不同的卷积层可能会共享同样的卷积核。这些共享权重在计算梯度时会产生重复，因此需要将它们整合为一个权重，从而减少计算量。

具体来说，该函数遍历参数列表 params，查找共享权重，如果找到则将它们的梯度累加起来，并将它们整合为一个参数。在这里，共享权重指的是同一个参数对象，共享偏置指的是具有相同形状和数值的参数，而共享 gamma 指的是具有相同形状和数值的 gamma。函数返回整合后的参数列表和梯度列表。
"""
def remove_duplicate(params, grads):
    """
    将共享权重的参数整合为一个，并累加它们的梯度。
    """
    params, grads = params[:], grads[:]  # 复制参数和梯度列表
    while True:
        find_flg = False  # 标记是否找到共享权重
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 如果找到共享权重，则将它们的梯度累加并整合为一个参数
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 如果找到共享偏置，则将它们的梯度累加并整合为一个偏置
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                        params[i].shape == params[j].shape and np.all(params[i] == params[j]):
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 如果找到共享 gamma 则将它们的梯度累加并整合为一个 gamma
                elif params[i].ndim == 1 and params[j].ndim == 1 and \
                        params[i].shape == params[j].shape and np.all(params[i] == params[j]):
                    grads[i] += grads[j]
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads


# trainer
"""
该类实现了神经网络模型的训练。在初始化过程中，存储模型 model 和优化器 optimizer，以及损失列表 loss_list、
评价间隔 eval_interval 和当前 epoch current_epoch。在 fit 方法中，首先计算数据集大小 data_size 和
每个 batch 的大小 batch_size，然后根据最大 epoch max_epoch 和 batch 数量计算最大迭代次数 max_iters。
接着将模型和优化器存储到局部变量 model 和 optimizer 中，并初始化 total_loss 和 loss_count 为 0。
在每个 epoch 中，首先打乱数据集，然后对每个 batch 进行循环训练。对于每个 batch，首先将输入和目标数据
划分为 batch_x 和 batch_t，然后计算损失 loss，并对模型进行反向传播，计算梯度并更新参数。在更新参数之前，
可以使用 remove_duplicate 函数将共享的权重整合为 1 个，以便减少计算量。此外，还可以使用 max_grad 参数来
限制梯度的范数，以避免梯度爆炸的问题。最后统计 total_loss 和 loss_count。在每个评价间隔 eval_interval 中，
计算平均损失，并打印出当前 epoch、迭代次数、时间和损失值。然后将平均损失添加到 loss_list 中，并将 total_loss 和 loss_count 重置为 0。
在每个 epoch 结束后，将当前 epoch current_epoch 加 1。plot 方法用于绘制损失曲线图。在该方法中，首先创建 x 轴上的刻度值 x，然后
根据评价间隔 eval_interval 构建 x 轴上的标签。如果设置了 y 轴的范围 ylim，则将其应用于绘图。
最后使用 matplotlib 库中的 plot 函数将损失列表 loss_list 绘制为折线图，并设置 x 轴和 y 轴的标签。
"""
class Trainer:
    def __init__(self, model, optimizer):
        self.model = model  # 存储神经网络模型
        self.optimizer = optimizer  # 存储优化器
        self.loss_list = []  # 存储损失值列表
        self.eval_interval = None  # 评价间隔
        self.current_epoch = 0  # 当前 epoch

    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x)  # 数据集大小
        max_iters = data_size // batch_size  # 最大迭代次数
        self.eval_interval = eval_interval  # 设置评价间隔
        model, optimizer = self.model, self.optimizer
        total_loss = 0  # 总损失
        loss_count = 0  # 损失次数

        start_time = time.time()  # 记录训练开始时间
        for epoch in range(max_epoch):  # 对每个 epoch 进行循环
            # 打乱数据集
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):  # 对每个 batch 进行循环
                batch_x = x[iters*batch_size:(iters+1)*batch_size]  # 获取当前 batch 的输入数据
                batch_t = t[iters*batch_size:(iters+1)*batch_size]  # 获取当前 batch 的目标数据

                # 前向传播计算损失，反向传播计算梯度并更新参数
                loss = model.forward(batch_x, batch_t)  # 前向传播计算损失
                model.backward()  # 反向传播计算梯度
                params, grads = remove_duplicate(model.params, model.grads)  # 将共享的权重整合为1个
                if max_grad is not None:
                    clip_grads(grads, max_grad)  # 限制梯度范数
                optimizer.update(params, grads)  # 更新参数
                total_loss += loss  # 统计总损失
                loss_count += 1  # 统计损失次数

                # 评价
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count  # 计算平均损失
                    elapsed_time = time.time() - start_time  # 计算训练时间
                    print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))  # 将平均损失添加到损失列表中
                    total_loss, loss_count = 0, 0  # 重置总损失和损失次数

            self.current_epoch += 1  # epoch 加 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))  # 构建 x 轴刻度值
        if ylim is not None:
            plt.ylim(*ylim)  # 如果设置了 y 轴范围，则应用于绘图
        plt.plot(x, self.loss_list, label='train')  # 绘制损失曲线图
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')  # 设置 x 轴标签
        plt.ylabel('loss')  # 设置 y 轴标签
        plt.show()  # 显示损失曲线图


if __name__ == "__main__":
    # 超参数
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # 加载数据
    word_2_index, index_2_word = read_data(os.path.join("dataset", "ptb.vocab.pkl"))
    corpus = np.load(os.path.join("dataset", "ptb.train.npy"))
    vocab_size = len(word_2_index)
    contexts, target = create_contexts_target(corpus, window_size)

    # 模型定义
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    opt = Adam()

    # 训练
    trainer = Trainer(model, opt)
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()

    # # 保存
    # word_vecs = model.word_vecs
    # params = {}
    # params['word_vecs'] = word_vecs.astype(np.float16)
    # params['word_to_id'] = word_2_index
    # params['id_to_word'] = index_2_word
    # pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
    # with open(pkl_file, 'wb') as f:
    #     pickle.dump(params, f, -1)
