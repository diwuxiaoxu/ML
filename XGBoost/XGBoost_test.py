# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

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
