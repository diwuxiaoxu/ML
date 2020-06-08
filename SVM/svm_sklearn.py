
# encoding=utf-8

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
from sklearn import svm

if __name__ == '__main__':
    print('prepare datasets...')
    # Iris数据集
    iris = datasets.load_iris()
    features = iris.data

    labels = iris.target



    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=0)
    time_2 = time.time()
    print('Start training...')
    clf = svm.SVC()  # svm class
    clf.fit(train_features, train_labels)  # training the svc model
    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)
