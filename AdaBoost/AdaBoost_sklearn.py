""""
二分类

（1）初始化训练数据（每个样本）的权值分布：如果有N个样本，则每一个训练的样本点最开始时都被赋予相同的权重：1/N。
（2）训练弱分类器。具体训练过程中，如果某个样本已经被准确地分类，那么在构造下一个训练集中，它的权重就被降低；相反，如果某个样本点没有被准确地分类，那么它的权重就得到提高。同时，得到弱分类器对应的话语权。然后，更新权值后的样本集被用于训练下一个分类器，整个训练过程如此迭代地进行下去。
（3）将各个训练得到的弱分类器组合成强分类器。各个弱分类器的训练过程结束后，分类误差率小的弱分类器的话语权较大，其在最终的分类函数中起着较大的决定作用，而分类误差率大的弱分类器的话语权较小，其在最终的分类函数中起着较小的决定作用。换言之，误差率低的弱分类器在最终分类器中占的比例较大，反之较小。

"""

# encoding=utf-8

import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.ensemble import AdaBoostClassifier

if __name__ == '__main__':
    print("Start read data...")
    time_1 = time.time()

    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values

    features = data[::, 1::]
    labels = data[::, 0]

    # 随机选取33%数据作为测试集，剩余为训练集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=0)

    time_2 = time.time()
    print('read data cost %f seconds' % (time_2 - time_1))

    print('Start training...')

    clf = AdaBoostClassifier(n_estimators=100, algorithm='SAMME.R')
    """
    base_estimator:基分类器，默认是决策树，在该分类器基础上进行boosting，理论上可以是任意一个分类器，但是如果是其他分类器时需要指明样本权重。

    n_estimators:基分类器提升（循环）次数，默认是50次，这个值过大，模型容易过拟合；值过小，模型容易欠拟合。

    learning_rate:学习率，表示梯度收敛速度，默认为1，如果过大，容易错过最优值，如果过小，则收敛速度会很慢；该值需要和n_estimators进行一个权衡，当分类器迭代次数较少时，学习率可以小一些，当迭代次数较多时，学习率可以适当放大。

    algorithm:boosting算法，也就是模型提升准则，有两种方式SAMME, 和SAMME.R两种，默认是SAMME.R，两者的区别主要是弱学习器权重的度量，前者是对样本集预测错误的概率进行划分的，后者是对样本集的预测错误的比例，即错分率进行划分的，默认是用的SAMME.R。
    """

    clf.fit(train_features, train_labels)

    time_3 = time.time()
    print('training cost %f seconds' % (time_3 - time_2))

    print('Start predicting...')
    test_predict = clf.predict(test_features)
    time_4 = time.time()
    print('predicting cost %f seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print("The accruacy score is %f" % score)
