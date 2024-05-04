"""_summary_
1.简介
    facebook ai团队开源的针对聚类和相似性搜索库,为稠密向量提供高效相似度搜索和聚类,支持十亿级别向量的搜索
    特性:
        支持相似度检索和聚类
        支持多种索引方式;
        支持cpu和GPU计算;
        支持python和c++调用
2.安装
    cpu版
        conda install faiss-cpu -c pytorch
        pip install faiss-cpu --no-cache
    GPU版
        conda install faiss-gpu cudatoolkit=12.0 -c pytorch
3.使用
    步骤:
        构建向量库,对已知数据进行向量,最终以矩阵的形式表示
        为矩阵选择合适的index,将第一步得到的矩阵add到index中
        search最终结果
        
        构建索引index
        根据不同索引的特性,对索引进行训练
        add 添加 xb数据到索引
        针对xq进行search操作
    
    索引基类Index(针对稠密向量)
        IndexFlatL2\IndexFlatIP\IndexHNSWFlat\IndexIVFFlat\
        IndexLSH\IndexScalarQuantizer\IndexPQ\IndexIVFScalarQuantizer\
        IndexIVFPQ\IndexIVFPQR

"""

# 使用案例
import numpy as np
import faiss

d = 64  # 向量维度dimension
nb = 100000  # 向量库中的向量数量 database size
nq = 10000  # 查询向量的数量nb of queries

# 构建向量库
# 向量库。生成随机向量，并调整第一个元素以防止所有向量都相同
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
print(type(xb))
print(xb[:2])
# 查询向量。同样地，生成随机向量并进行调整
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

# 关键步骤 build index构建索引
# 使用 faiss.IndexFlatL2 创建一个以 L2 距离为基准的索引,这意味着索引将计算向量间的欧氏距离
index = faiss.IndexFlatL2(d)
# 将向量库 xb 添加到刚创建的索引中
index.add(xb)

# 使用查询向量 xq 中的前5个向量搜索它们的最近邻（k=4），即每个查询向量返回4个最近的邻居
k = 4
D, I = index.search(xq[:5], k)
# I的维度是nq*4,代表距离每个query最近的k个数据的id
# D的维度是nq*4,代表距离每个query最近的k个数据的距离
print('D:', D, '\nI:', I)
"""
D: [[7.7808666 7.8058233 7.8452034 7.849343 ]
 [6.569394  6.905781  6.9702663 7.3386545]
 [6.5604625 6.6950636 6.7078743 6.8476715]
 [6.5510855 6.627158  6.8704214 7.049502 ]
 [6.6219087 6.660241  6.686224  6.722783 ]] 
I: [[ 239  464  450   65]
 [  16  132  419  337]
 [  57  928  201  234]
 [1006  505  475  188]
 [ 384   75  471 1065]]
 """