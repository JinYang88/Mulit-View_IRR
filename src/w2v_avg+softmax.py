# coding = utf-8

import gensim
import pandas as pd
import re
import os
import numpy as np
from sklearn import datasets, neighbors, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer

def getvocabulary(text):
    voc = []
    for item in text:
        for word in re.split(' ', item.strip()):
            voc.append(word)
    return set(voc)


# 将一句话转换成word2vec的首尾相接形式
def seq2vec(model, str_):
    li = []
    str_ = re.split(' ', str_)
    for item in str_:
        li.append(model[item])
    return li


# 将一句话转换成word2vec的均值
def seq2avg(model, str_):
    str_ = re.split(' ', str_.strip())
    try:
        temp = np.array(model[str_[0]])
    except KeyError:
        temp = np.random.normal(0, 0.1, 300)
    for item in str_[1:]:
        try:
            temp += model[item]
        except KeyError:
            temp += np.random.normal(0, 0.1, 300)
    temp = temp / len(str_)
    return temp


if __name__ == '__main__':
    label_mapping = {
        'guilt': 0,
        'disgust': 1,
        'sadness': 2,
        'shame': 3,
        'anger': 4,
        'fear': 5,
        'joy': 6
    }

    dir = 'ISEAR'  # Small_ISEAR 是个小数据集, 用于测试程序的正确性
    train_data = pd.read_csv(dir + '/train.csv', encoding='utf-8')
    test_data = pd.read_csv(dir + '/test.csv', encoding='utf-8')
    Text_test = test_data.data
    Y_test = test_data.label
    Text_train = train_data.data
    Y_train = train_data.label

    Y_train = Y_train.map(label_mapping)
    Y_test = Y_test.map(label_mapping)

    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.KeyedVectors.load_word2vec_format('./word2vec_model/GoogleNews-vectors-negative300.bin.gz',
                                                        binary=True)

    # Load simple self-trained Word2Vec model.
    # save_path = 'word2vec_model/temp.w2v'
    # if not os.path.isfile(save_path):  # 如果没有已经有训练好的word2vec模型
    #     w_train = [re.split(' ', item) for item in Text_train]
    #     w_test = [re.split(' ', item) for item in Text_test]
    #     w_train += w_test
    #     model = gensim.models.Word2Vec(w_train, min_count=1, size=300)
    #     model.save(save_path)
    # else:  # 如果有模型, 就直接把模型读出来
    #     model = gensim.models.Word2Vec.load(save_path)

    X_train = []
    for row in Text_train:
        X_train.append(seq2avg(model, row))

    X_test = []
    for row in Text_test:
        X_test.append(seq2avg(model, row))

    logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    logistic_model = logistic.fit(X_train, Y_train)

    print(logistic.score(X_test, Y_test))
