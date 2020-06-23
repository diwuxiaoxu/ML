# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
"""
XGBClassifier(max_depth=3,
learning_rate=0.1, 
n_estimators=100, 
silent=True, 
objective='binary:logistic', 
booster='gbtree', n_jobs=1, 
nthread=None, gamma=0, 
min_child_weight=1, max_delta_step=0, 
subsample=1, colsample_bytree=1,
colsample_bylevel=1, reg_alpha=0,
reg_lambda=1, scale_pos_weight=1,
base_score=0.5, random_state=0,
seed=None, missing=None, **kwargs)

"""
if __name__ == '__main__':
    data = pd.read_csv('../data/train_binary.csv', header=0)

    data = data.values

    X = data[::, 1::]

    y = data[::, 0].reshape(-1, 1)

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

    model = XGBClassifier()

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    acc_test = accuracy_score(y_test,y_pred)

    print(acc_test)
