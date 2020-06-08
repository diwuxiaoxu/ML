import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

pca = PCA(n_components=2)

newX = pca.fit_transform(X)

print(newX)

# 返回所保留各个特征的方差百分比
print(pca.explained_variance_ratio_)
