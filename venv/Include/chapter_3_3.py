# sklearn  knn 用法
from sklearn import datasets, neighbors

iris = datasets.load_iris()

X = iris.data
y = iris.target
knn_clf = neighbors.KNeighborsClassifier()
knn_clf.fit(X, y)
predict = knn_clf.predict(X)

accuracy = (y == predict).astype(int).mean()
print(accuracy)
