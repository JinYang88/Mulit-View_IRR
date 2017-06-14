# coding = utf-8

import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import pickle
import os
from sklearn import datasets, neighbors, linear_model
from scipy import sparse

def getvocabulary(text):
    voc = []
    for item in text:
        for word in re.split(' ', item.strip()):
            voc.append(word)
    return set(voc)

def getonehot(sample, voc):
    return sparse.coo_matrix([1 if sample.count(item) else 0 for item in voc])

# xLen 期望得到的x的长度
def IIR4x(zs, xLen, xs, Ws, IterNum=1, c=1.0, C2=1.0):
    ViewNum = len(zs[0]) # View的个数
    WordLen = len(zs[0][0])  # 输入的词向量的长度
    n = len(zs)
    start = time.time()

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

    end = time.time()
    print('x迭代一次耗时{}s'.format(round(end - start, 2)))

    return xs


def IIR4W(xs, zs, Ws, C1=1.0, c=1.0, IterNum=1):
    ViewNum = len(zs[0])
    WordLen = len(zs[0][0])  # 输入的词向量的长度
    xLen = len(xs[0])
    n = len(xs)

    if Ws == []:
        for i in range(ViewNum):
            Ws.append(np.random.normal(1, 0.1, WordLen * xLen).reshape(WordLen, xLen))

    for v in range(ViewNum):
        start = time.time()
        for it in range(IterNum):
            start = time.time()
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
            nws = np.matmul(ZQX, XQX)
            Ws[v] = nws
            end = time.time()
            print('W迭代一次耗时{}s'.format(round(end - start, 2)))

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

    start = time.time()
    xs = IIR4x(zs, xLen, Ws=[], xs=[], C2=C2, c=c)  # 更新x
    ws = IIR4W(xs, zs, Ws=[], C1=C1, c=c)  # 更新w
    end = time.time()
    print('迭代一次耗时{}s'.format(round(end-start, 2)))
    for iter in range(IterNum-1):
        if iter % 5 == 0:
            print(getLoss(zs, xs, ws, C1=C1, C2=C2, c=c))  # 输出loss
        xs = IIR4x(zs, xLen, xs, ws, C2=C2, c=c)  # 更新x
        ws = IIR4W(xs, zs, Ws=ws, C1=C1, c=c)  # 更新w

    return ws, xs



if __name__ == '__main__':
    all_start = time.time()

    label_mapping = {
        'guilt': 0,
        'disgust': 1,
        'sadness': 2,
        'shame': 3,
        'anger': 4,
        'fear': 5,
        'joy': 6
    }
    label_imapping = {
        0: 'guilt',
        1: 'disgust',
        2: 'sadness',
        3: 'shame',
        4: 'anger',
        5: 'fear',
        6: 'joy'
    }

    dir = 'AI_ISEAR'  # Small_ISEAR 是个小数据集, 用于测试程序的正确性
    train_data = pd.read_csv(dir + '/train.csv', encoding='utf-8')
    test_data = pd.read_csv(dir + '/test.csv', encoding='utf-8')
    Text_test = test_data.data
    Text_train = train_data.data
    Y_test = test_data.label
    Y_train = train_data.label
    x_len = 100

    Y_train = Y_train.map(label_mapping)
    Y_test = Y_test.map(label_mapping)

    Text_all = pd.concat([Text_test, Text_train])
    voc = getvocabulary(Text_all)
    print('词典构建结束')
    print('词典大小为{}'.format(len(voc)))
    print('耗时{}s'.format(round(time.time() - all_start, 2)))

    tfidf_vectorizer = TfidfVectorizer(max_df=1, min_df=1, vocabulary=voc)
    tfidf_train = tfidf_vectorizer.fit_transform(raw_documents=Text_train)
    tfidf_test = tfidf_vectorizer.fit_transform(raw_documents=Text_test)
    print('TFIDF构建结束')
    print('耗时{}s'.format(round(time.time() - all_start, 2)))

    # 构造训练集
    x_train = []
    x_test = []
    for i in range(len(Text_train)):
        # x_train.append([getonehot(Text_train[i], voc), tfidf_train[i].toarray()[0]])
        x_train.append([getonehot(Text_train[i], voc), tfidf_train[i]])
    for i in range(len(Text_test)):
        x_test.append([getonehot(Text_test[i], voc), tfidf_test[i].toarray()])
        # x_test.append([getonehot(Text_test[i], voc), tfidf_test[i].toarray()[0]])
    print('组合特征结束')
    print('耗时{}s'.format(round(time.time() - all_start, 2)))

    if os.path.isfile('var_save/vars'):
        ws_train, xs_train, ws_test, xs_test = pickle.load(open('var_save/vars', 'rb'))
    else:
        print('开始迭代')
        ws_train, xs_train = IIRITER(x_train, IterNum=2, xLen=x_len)
        ws_test, xs_test = IIRITER(x_test, IterNum=2, xLen=x_len)
        print('迭代结束')
        print('耗时{}s'.format(round(time.time() - all_start, 2)))
        pickle.dump([ws_train, xs_train, ws_test, xs_test], open('var_save/vars', 'wb'))

    print(np.matmul(ws_test[1], xs_test[0]).reshape([1, len(voc)]))
    print(x_test[0][1])

    xs_train = np.reshape(xs_train, [len(Text_train), x_len])
    xs_test = np.reshape(xs_test, [len(Text_test), x_len])

    logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)
    logistic_model = logistic.fit(xs_train, Y_train)

    print(logistic.score(xs_test, Y_test))
