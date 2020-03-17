# knn2 建立kd树进行分类

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from math import sqrt

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width ', 'label']

data = np.array(df.iloc[:100, [0, 1, -1]])
train, test = train_test_split(data, test_size=0.4)
x0 = np.array([x0 for i, x0 in enumerate(train) if train[i][-1] == 0])
x1 = np.array([x1 for i, x1 in enumerate(train) if train[i][-1] == 1])


def show_train():
    plt.scatter(x0[:, 0], x0[:, 1], c='pink', label='[0]')
    plt.scatter(x1[:, 0], x1[:, 1], c='orange', label='[1]')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')


class Node:
    def __init__(self, data, depth=0, lchild=None, rchild=None):
        self.data = data
        self.depth = depth
        self.lchild = lchild
        self.rchild = rchild


class KdTree:
    def __init__(self):
        self.KdTree = None
        self.n = 0
        self.nearest = None

    def create(self, dataSet, depth=0):
        if len(dataSet) > 0:
            m, n = np.shape(dataSet)
            self.n = n - 1
            axis = depth % self.n
            mid = int(m / 2)
            dataSetcopy = sorted(dataSet, key=lambda x: x[axis])
            node = Node(dataSetcopy[mid], depth)
            if depth == 0:
                self.KdTree = node
            node.lchild = self.create(dataSetcopy[:mid], depth + 1)
            node.rchild = self.create(dataSetcopy[mid + 1:], depth + 1)
            return node
        return None

    # 先序遍历
    def preOrder(self, node):
        if node is not None:
            print(node.depth, node.data)
            self.preOrder(node.rchild)
            self.preOrder(node.lchild)

    def search(self, x, count=1):
        nearest = []
        for i in range(count):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)

        def recurve(node):
            if node is not None:
                axis = node.depth % self.n
                daxis = x[axis] - node.data[axis]
                if daxis < 0:
                    recurve(node.lchild)
                else:
                    recurve(node.rchild)

                dist = sqrt(sum(p1 - p2) ** 2 for p1, p2 in zip(x, node.data))
                for i, d in enumerate(self.nearest):
                    if d[0] < 0 or dist < d[0]:
                        self.nearest = np.insert(self.nearest, i, [dist, node], axis=0)
                        self.nearest = self.nearest[:-1]
                        break   # 这里一定要break

                n = list(self.nearest[:, 0]).count(-1)
