# coding = utf-8

import gensim
import pandas as pd
import re
import os
import numpy as np


# zs 一个句子, [词1, 词2, 词3......]
# xLen 期望得到的x的长度
def IIR4x(zi, xLen, IterNum=10, c=1, C2=1):
    ViewNum = len(zi) # View的个数
    WordLen = len(zi[0])  # 输入的词向量的长度
    Wv = []  # 对每一个View初始化矩阵
    x_0 = np.random.normal(0, 0.1, xLen)
    x = x_0

    for i in range(ViewNum):
        Wv.append(np.random.normal(0, 0.1, WordLen * xLen).reshape(WordLen, xLen))

    for iter in range(IterNum):
        # 获得Q
        Q = []
        for i in range(ViewNum):
            xi = np.matmul(Wv[i], x)
            ri_norm = np.linalg.norm(xi-zi[i], 1) ** 2
            Q.append(1.0 / (c ** 2 + ri_norm))

        # 更新矩阵
        mC2I = np.eye(xLen, xLen) * ViewNum * C2
        WQW = np.zeros([xLen, xLen])
        WQZ = np.zeros(xLen)
        for i in range(ViewNum):
            WT = Wv[i].T
            WTQ = Q[i] * WT
            WQW += np.matmul(WTQ, Wv[i])
            WQZ += np.matmul(WTQ, zi[i])
        WQW += mC2I
        x = np.matmul(WQW.I, WQZ)

    return x


def IIR4W(xs, zs, C1=1, IterNum=10):
    ViewNum = len(zs[0])
    WordLen = len(zs[0][0])  # 输入的词向量的长度
    xLen = len(xs[0])
    n = len(zs)
    Ws =[]

    for i in range(ViewNum):
        Ws.append(np.random.normal(0, 0.1, WordLen * xLen).reshape(WordLen, xLen))

    for it in range(IterNum):
        for v in range(ViewNum):
            # 获得Q
            Q = []
            for i in range(n):
                xi = np.matmul(Ws[v], x)
                ri_norm = np.linalg.norm(xi - zs[i], 1) ** 2
                Q.append(ri_norm)

            XQX = np.zeros([xLen, xLen])
            nC1 = np.eye(xLen, xLen)
            for i in range(n):
                XQX += np.matmul(xs[i] * Q[i], xs[i].T)
            XQX += nC1
            XQX = XQX.I

            W = np.zeros([WordLen, xLen])
            for i in range(n):
                ZQX = np.matmul(zs[i] * Q[i], xs[i].T)
                W += np.matmul(ZQX, XQX)

            Ws[v] = W

    return Ws

def seq2vec(model, str_):
    li= []
    str_ = re.split(' ', str_)
    for item in str_:
        li.append(model[item])
    return li


if __name__ == '__main__':

    # word1 = [1, 2, 3, 4]
    # word2 = [4, 3, 2, 3]
    # sequence = [word1, word2]
    #
    # print(IIR4x(sequence, 4, IterNum=100))
    train_data = pd.read_csv('ISEAR/train.txt')

    X_test = pd.read_csv('ISEAR/test.txt').data
    Y_test = pd.read_csv('ISEAR/test_label.txt').label
    X_train = train_data.data
    Y_train = train_data.label

    # word2vec 特征
    save_path = 'word2vec_model/temp.w2v'
    if not os.path.isfile(save_path):
        w_train = [re.split(' ', item) for item in X_train]
        w_test = [re.split(' ', item) for item in X_test]
        w_train += w_test
        model = gensim.models.Word2Vec(w_train, min_count=1, size=20)
        model.save(save_path)
    else:
        model = gensim.models.Word2Vec.load(save_path)

    for item in X_train:
        seq1 = seq2vec(model, item)
