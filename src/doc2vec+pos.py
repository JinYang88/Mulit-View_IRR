#coding=utf-8

import gensim
import os
import logging
import pandas as pd
import nltk
import pickle
import numpy as np
import IRR
import time
from sklearn import preprocessing

class d2v_pos():
    def __init__(self, xlen=100):
        self.label_mapping = {
            'guilt': 0,
            'disgust': 1,
            'sadness': 2,
            'shame': 3,
            'anger': 4,
            'fear': 5,
            'joy': 6
        }
        self.label_imapping = {
            0: 'guilt',
            1: 'disgust',
            2: 'sadness',
            3: 'shame',
            4: 'anger',
            5: 'fear',
            6: 'joy'
        }
        self.filetype = 'Small_ISEAR'  # Small_ISEAR 是个小数据集, 用于测试程序的正确性
        self.Text_train = ""
        self.Text_test = ""
        self.Y_test = ""
        self.Y_train = ""
        self.X_train = []
        self.X_test = []
        self.X_all = []
        self.docvecs = []
        self.xlen = xlen

    def read_data(self):
        self.Text_test = pd.read_csv('../{}/text_test.csv'.format(self.filetype), encoding='utf-8', names=['data'], dtype=str)
        self.Text_train = pd.read_csv('../{}/text_train.csv'.format(self.filetype), encoding='utf-8', names=['data'], dtype=str)
        self.Y_test = pd.read_csv('../{}/label_test.csv'.format(self.filetype), encoding='utf-8', names=['label'], dtype=str)
        self.Y_train = pd.read_csv('../{}/label_train.csv'.format(self.filetype), encoding='utf-8', names=['label'], dtype=str)
        self.Text_all = pd.read_csv('../{}/text_all.csv'.format(self.filetype), encoding='utf-8', names=['data'], dtype=str)

    def train_d2v(self):
        # 如果train过就不用train
        dumppath = '../VectorModel/{}_{}_vector.pk'.format(self.filetype, 'all')
        if os.path.isfile(dumppath):
            with open(dumppath, 'rb') as fr:
                self.docvecs = pickle.load(fr)
        else:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            sentences = gensim.models.doc2vec.TaggedLineDocument('../{}/text_all.csv'.format(self.filetype))
            model = gensim.models.Doc2Vec(sentences, size=100, window=5)
            model.save(dumppath)
            self.docvecs = model.docvecs
            with open(dumppath, 'wb') as out:
                pickle.dump(model.docvecs, out)
        min_max_scaler = preprocessing.MinMaxScaler()
        docvecs_0_1 = min_max_scaler.fit_transform(self.docvecs)
        self.X_all = [[line] for line in docvecs_0_1]
        # print(self.X_all[0])

    def get_pos(self):
        # 类型: 名词, 形容词, 动词, 感叹词, 副词
        # 数出以上类型在各个位置的数量
        part_num = 5

        type_pos = {
            'NN': 0,
            'JJ': 1 * part_num,
            'VB': 2 * part_num,
            'UH': 3 * part_num,
            'RB': 4 * part_num
        }
        all_pos_info = []
        for idx, line in self.Text_all.iterrows():
            pos_info = [0] * part_num * 5
            line = line['data']
            words = nltk.word_tokenize(line)
            word_tag = nltk.pos_tag(words)
            for idx, item in enumerate(word_tag):
                for type in type_pos:
                    if item[1].count(type):
                        position = int(part_num * (idx+1)/len(word_tag))
                        pos_info[type_pos[type] + position] += 1
            all_pos_info.append(np.array(pos_info, dtype=np.int))

        for idx, line in enumerate(all_pos_info):
            self.X_all[idx].append(line)

    def IRR(self):
        dumppath = '../var_save/IRR_res.pk'
        if os.path.isfile(dumppath):
            ws_all, xs_all = pickle.load(open(dumppath, 'rb'))
        else:
            print('开始迭代')
            all_start = time.time()
            ws_all, xs_all = IRR.IIRITER(self.X_all, IterNum=10, xLen=self.xlen, C1=0.000001, C2=0.000001, c=0.1)
            print('迭代结束')
            print('耗时{}s'.format(round(time.time() - all_start, 2)))
            # pickle.dump([ws_all, xs_all], open(dumppath, 'wb'))

        for idx, line in enumerate(xs_all):
            line = np.array(line).reshape([1, self.xlen])[0]
            if idx < len(self.Text_train):
                self.X_train.append(line)
            else:
                self.X_test.append(line)

        # print(ws_all[1])
        # print('合成结果:')
        # print(self.X_train[0])
        # print('第一个view原结果:')
        # print(self.X_all[0][0])
        # print('第一个view拆分结果:')
        # print(np.matmul(ws_all[0], self.X_train[0]))
        # print('第二个view原结果')
        # print(self.X_all[0][1])
        # print('第二个view拆分结果:')
        # print(np.matmul(ws_all[1], self.X_train[0]))

# model = d2v_pos()
# model.read_data()
# model.train_d2v()
# model.get_pos()
# words = nltk.word_tokenize('oh I wa busy')
# print(words)
# word_tag = nltk.pos_tag(words)
# print(word_tag)

if __name__ == '__main__':
    dp = d2v_pos()
    dp.read_data()
    dp.train_d2v()
    dp.get_pos()
    dp.IRR()