# coding = utf-8

import numpy as np
import sys
import time

sys.path.append("H:\PythonProject\multi-learn\src\IRR.py")

# zs 一个句子, [词1, 词2, 词3......]
# xLen 期望得到的x的长度
def IIR4x(zs, xLen, xs, Ws, IterNum=1, c=1.0, C2=1.0):
    ViewNum = len(zs[0])  # View的个数
    n = len(zs)
    start = time.time()

    if xs == []:
        for i in range(n):
            xs.append(np.random.normal(0, 0.1, xLen))

    if Ws == []:
        for i in range(ViewNum):
            WordLen = len(zs[0][i])
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
                WordLen = len(zs[0][i])
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

    return xs


def IIR4W(xs, zs, Ws, C1=1.0, c=1.0, IterNum=1):
    ViewNum = len(zs[0])
    xLen = len(xs[0])
    n = len(xs)

    if Ws == []:
        for i in range(ViewNum):
            WordLen = len(zs[0][i])
            Ws.append(np.random.normal(1, 0.1, WordLen * xLen).reshape(WordLen, xLen))

    for v in range(ViewNum):
        start = time.time()
        WordLen = len(zs[0][v])
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
        if iter % 25 == 0:
            print(getLoss(zs, xs, ws, C1=C1, C2=C2, c=c))  # 输出loss
        xs = IIR4x(zs, xLen, xs, ws, C2=C2, c=c)  # 更新x
        ws = IIR4W(xs, zs, Ws=ws, C1=C1, c=c)  # 更新w

    return ws, xs
