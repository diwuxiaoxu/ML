# -*- coding: utf-8 -*-
import random


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

""""AdaBoost 算法是利用前一轮的弱学习器的误差来更新样本权重值，然后一轮一轮的迭代；
我们先初始化第一棵回归树，使这个分界点让整体误差最小;
我们每生成一棵树之后，就将这棵树的每一条数据的损失函数（均方误差）的梯度求出来；
求出每个数据的负梯度之后，我们依据已有数据和每个数据的负梯度，生成一个新树出来，我们先将每个数据的负梯度当作新的数据的yi，这样就得到了一组新数据，也确定了新数据的空间划分。
然后再计算每一条数据的误差函数，取误差函数最小的那个点做为下一个分支切分点，这也就生成了一颗新树；
我们将新树加到原来那棵树上；
最后总和得到一棵树；
"""
if __name__ == '__main__':

    data = pd.read_csv('../data/train_binary.csv', header=0)

    data = data.values

    X = data[::, 1::]

    y = data[::, 0].reshape(-1, 1)

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# 模型训练，使用GBDT算法
    gbr = GradientBoostingClassifier(n_estimators=10, max_depth=2, learning_rate=0.1)

    gbr.fit(x_train, y_train)


    y_gbr = gbr.predict(x_test)

    acc_test = accuracy_score(y_test, y_gbr)

    print(acc_test)

