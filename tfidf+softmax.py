# coding = utf-8

import gensim
import pandas as pd
import re
import os
import numpy as np
from sklearn import datasets, neighbors, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

def getvocabulary(text):
    voc = []
    for item in text:
        for word in re.split(' ', item.strip()):
            voc.append(word)
    return set(voc)

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
    train_data = pd.read_csv(dir + '/train.txt')
    Text_test = pd.read_csv(dir + '/test.txt').data
    Y_test = pd.read_csv(dir + '/test_label.txt').label
    Text_train = train_data.data
    Y_train = train_data.label
    Y_train = Y_train.map(label_mapping)
    Y_test = Y_test.map(label_mapping)

    Text_all = pd.concat([Text_test, Text_train])

    voc = getvocabulary(Text_all)

    tfidf_vectorizer = TfidfVectorizer(max_df=1, min_df=0, vocabulary=voc)
    tfidf_train = tfidf_vectorizer.fit_transform(raw_documents=Text_train)
    tfidf_test = tfidf_vectorizer.fit_transform(raw_documents=Text_test)


    # print(np.append(tfidf_train[0].toarray(), np.array([1, 0, 1])))

    # logistic = linear_model.LogisticRegression(multi_class='multinomial', solver='lbfgs')
    # logistic_model = logistic.fit(tfidf_train, Y_train)
    #
    # print(logistic.score(tfidf_test, Y_test))
