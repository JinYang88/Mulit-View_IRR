# coding = utf-8

import gensim
import pandas as pd
import re
import os
import numpy as np
from sklearn import datasets, neighbors, linear_model

# zs 一个句子, [词1, 词2, 词3......]
# xLen 期望得到的x的长度
def IIR4x(zs, xLen, xs, Ws, IterNum=10, c=1.0, C2=1.0):
    ViewNum = len(zs[0]) # View的个数
    WordLen = len(zs[0][0])  # 输入的词向量的长度
    n = len(zs)

    if xs == []:
        for i in range(n):
            xs.append(np.random.normal(0, 0.1, xLen))

    if Ws == []:
        for i in range(ViewNum):
            Ws.append(np.random.normal(0, 0.1, WordLen * xLen).reshape(WordLen, xLen))

    for j in range(n):
        x = xs[j]
        for iter in range(IterNum):
            # 获得Q
            Q = []
            for i in range(ViewNum):
                xi = np.matmul(Ws[i], x)
                ri_norm = np.linalg.norm(xi-zs[j][i], 1) ** 2
                Q.append(1.0 / (c ** 2 + ri_norm))
            # print(Q)
            # 更新矩阵
            mC2I = np.eye(xLen, xLen) * ViewNum * C2
            WQW = np.zeros([xLen, xLen])
            WQZ = np.zeros([xLen, 1])
            for i in range(ViewNum):
                WT = Ws[i].T
                WTQ = Q[i] * WT
                WQW += np.matmul(WTQ, Ws[i])
                WQZ += np.matmul(WTQ, np.array(zs[j][i]).reshape(WordLen, 1))
            WQW += mC2I
            WQW = np.mat(WQW)
            # print(WQW.I)
            x = np.matmul(WQW.I, WQZ).reshape([xLen, 1])
        xs[j] = x

    return xs


def IIR4W(xs, zs, Ws, C1=1.0, c=1.0, IterNum=10):
    ViewNum = len(zs[0])
    WordLen = len(zs[0][0])  # 输入的词向量的长度
    xLen = len(xs[0])
    n = len(xs)

    if Ws == []:
        for i in range(ViewNum):
            Ws.append(np.random.normal(1, 0.1, WordLen * xLen).reshape(WordLen, xLen))

    for v in range(ViewNum):
        for it in range(IterNum):
            # 获得Q
            Q = []
            for i in range(n):
                xi = np.matmul(Ws[v], xs[i])
                ri_norm = np.linalg.norm(xi - zs[i][v], 1) ** 2
                Q.append(1 / (c**2 + ri_norm))

            XQX = np.zeros([xLen, xLen])
            ZQX = np.zeros([WordLen, xLen])
            nC1 = np.eye(xLen, xLen) * C1 * n
            for i in range(n):
                XQX += np.matmul(xs[i] * Q[i], xs[i].T)
                ZQX += np.matmul(np.array(zs[i][v]).reshape([WordLen, 1]) * Q[i], xs[i].T)
            XQX += nC1
            XQX = np.mat(XQX)
            XQX = XQX.I

            Ws[v] = np.matmul(ZQX, XQX)

    return Ws


def getLoss(zs, xs, ws, c=1, C1=1.0, C2=1.0):
    ViewNum = len(zs[0])
    n = len(zs)

    loss = 0
    for v in range(ViewNum):
        for i in range(n):
            xi = np.matmul(ws[v], xs[i])
            ri_norm = np.linalg.norm(xi - zs[i][v], 1) ** 2
            loss += np.log(1 + ri_norm/c**2)

    wl = 0
    for v in range(ViewNum):
        wl += np.linalg.norm(ws[v], 1) ** 2

    xl = 0
    for i in range(n):
        xl += np.linalg.norm(xs[i], 1) ** 2

    return loss + C1 * wl + C2 * xl


def IIRITER(zs,C1=0.001, C2=0.001, c=1, xLen=10, IterNum=10):

    xs = IIR4x(zs, xLen, Ws=[], xs=[], C2=C2, IterNum=1, c=c)
    ws = IIR4W(xs, zs, Ws=[], C1=C1, IterNum=1, c=c)
    for iter in range(IterNum-1):
        if iter % 1 == 0:
            print(getLoss(zs, xs, ws, C1=C1, C2=C2, c=c))
        xs = IIR4x(zs, xLen, xs, ws, C2=C2, c=c)
        ws = IIR4W(xs, zs, Ws=ws, C1=C1, c=c)

    return ws, xs


def seq2vec(model, str_):
    li = []
    str_ = re.split(' ', str_)
    for item in str_:
        li.append(model[item])
    return li


if __name__ == '__main__':

    # word1 = np.array([-0.1, 0.2, 0.3, 0.4])
    # word2 = np.array([0.4, 0.3, 0.2, 0.33])
    # word3 = np.array([4, 5, 2, 1])
    # word4 = np.array([0, 1, 4, 3])
    # sequence = [word1, word2]
    # sequence2 = [word3, word4]
    # sequences = [sequence, sequence2]
    #
    # ws, xs = IIRITER(sequences, IterNum=10)

    # for idi, xi in enumerate(xs):
    #     for idv, wv in enumerate(ws):
    #         print(np.matmul(wv, xi), sequences[idi][idv])

    # print(x1)
    #
    # print(np.shape(x1))

    label_mapping = {
        'guilt': 0,
        'disgust': 1,
        'sadness': 2,
        'shame': 3,
        'anger': 4,
        'fear': 5,
        'joy': 6
    }

    dir = 'Small_ISEAR'
    train_data = pd.read_csv(dir + '/train.txt')
    Text_test = pd.read_csv(dir + '/test.txt').data
    Y_test = pd.read_csv(dir + '/test_label.txt').label
    Text_train = train_data.data
    Y_train = train_data.label
    Y_train = Y_train.map(label_mapping)
    Y_test = Y_test.map(label_mapping)

    wordlen = 10
    xlen = 10
    IterNum = 30

    # word2vec 特征
    save_path = 'word2vec_model/temp.w2v'
    if not os.path.isfile(save_path):
        w_train = [re.split(' ', item) for item in Text_train]
        w_test = [re.split(' ', item) for item in Text_test]
        w_train += w_test
        model = gensim.models.Word2Vec(w_train, min_count=1, size=wordlen)
        model.save(save_path)
    else:
        model = gensim.models.Word2Vec.load(save_path)

    X_train = []
    X_test = []
    for item in Text_train:
        seq = seq2vec(model, item)
        X_train.append(seq)
    for item in Text_test:
        seq = seq2vec(model, item)
        X_test.append(seq)

    X_all = X_train + X_test
    maxlen = max([len(item) for item in X_all])

    for idx in range(len(X_train)):
        while len(X_train[idx]) < maxlen:
            X_train[idx].append(np.zeros(wordlen))

    for idx in range(len(X_test)):
        while len(X_test[idx]) < maxlen:
            X_test[idx].append(np.zeros(wordlen))

    print('Start IIR')
    ws_test, x_test = IIRITER(X_test, IterNum=IterNum, xLen=xlen)
    print('------------')
    ws_train, x_train = IIRITER(X_train, IterNum=IterNum, xLen=xlen)

    x_train = np.array(x_train).reshape([len(Text_train), xlen])
    x_test = np.array(x_test).reshape([len(Text_test), xlen])

    logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logistic_model = logistic.fit(x_train, Y_train)

    print(logistic.score(x_test, Y_test))
