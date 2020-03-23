# sklearn LogisticRegression
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def create_data():
    iris = load_iris()
    X, y = np.array(iris.data), np.array(iris.target)
    return X[:100,0:2], y[:100]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
print(clf.coef_, clf.intercept_)
x_points = np.arange(4, 8)
y_ = -(clf.coef_[0][0] * x_points + clf.intercept_[0]) / clf.coef_[0][1]
plt.plot(x_points, y_)

plt.plot(X[:50, 0],X[:50,1],'bo',color='blue',label='0')
plt.plot(X[50:, 0],X[50:,1],'bo',color='orange',label='1')

plt.xlabel('speal length')
plt.ylabel('speal weight')
plt.legend()
plt.show()

