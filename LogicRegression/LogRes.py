from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

iris = load_iris()

from sklearn.model_selection import train_test_split

X = iris.data[:, :2]

Y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(penalty='l2', solver='newton-cg', multi_class='multinomial')

#  LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='multinomial',
#           n_jobs=1, penalty='l2', random_state=None, solver='newton-cg',
#           tol=0.0001, verbose=0, warm_start=False)

lr.fit(x_train, y_train)

score = lr.score(x_test, y_test)

print(score)
