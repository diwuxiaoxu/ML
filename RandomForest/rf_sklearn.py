
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    data = pd.read_csv('../data/train_binary.csv', header=0)

    data = data.values

    X = data[::, 1::]

    y = data[::, 0].reshape(-1, 1)

    x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 0)

    model = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    acc_test = accuracy_score(y_test,y_pred)

    print(acc_test)

