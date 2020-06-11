
# from sklearn.ensemble import RandomForestRegressor
#
# from sklearn.datasets import load_iris
#
# iris = load_iris()
# # print iris#iris的４个属性是：萼片宽度　萼片长度　花瓣宽度　花瓣长度　标签是花的种类：setosa versicolour virginica
# print(iris['target'].shape)
#
# rf = RandomForestRegressor()  # 这里使用了默认的参数设置
#
# rf.fit(iris.data[:150], iris.target[:150])  # 进行模型的训练
#
# # 随机挑选两个预测不相同的样本
# instance = iris.data[[100, 109]]
# print(instance)
# rf.predict(instance[[0]])
# print('instance 0 prediction；', rf.predict(instance[[0]]))
# print('instance 1 prediction；', rf.predict(instance[[1]]))
#
# print(iris.target[100], iris.target[109])
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = make_blobs(n_samples=10000, n_features=10, centers=100,random_state=0)

clf = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())


clf = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())

clf = ExtraTreesClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
scores = cross_val_score(clf, X, y)
print(scores.mean())
