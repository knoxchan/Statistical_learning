# 朴素贝叶斯 基础方法

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from collections import Counter
import math


# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, :])
    return data[:, :-1], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


class NaiveBayes:
    def __init__(self):
        self.model = None

    @staticmethod  # 这里定义为成员函数也是可以的
    # 静态方法无需实例化
    # eg:   NaiveBayes.mean(X) 可以这样直接调用 方法不需要实例化
    # 同时也可以先实例化方法，再进行调用
    # 平均数
    def mean(X):
        return sum(X) / float(len(X))

    # 标准差(方差)
    def stdev(self, X):
        avg = self.mean(X)
        return math.sqrt(sum([pow(x - avg, 2) for x in X]) / float(len(X)))

    # 概率密度函数
    def guassian_probability(self, x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return exponent

    # 处理X_train
    def summarize(self, train_data):
        # 注意这里train_data 的用法
        summarize = [(self.mean(i), self.stdev(i)) for i in zip(*train_data)]
        return summarize

    # 分类求平均值和标准差
    def fit(self, X, y):
        labels = list(set(y))
        data = {label: [] for label in labels}
        for f, label in zip(X, y):
            data[label].append(f)
        self.model = {label: self.summarize(value) for label, value in data.items()}
        return 'GussianNB train done!'

    # 计算概率
    def calculate_probabilities(self, input_data):
        probabilities = {}
        for label, value in self.model.items():
            probabilities[label] = 1
            for i in range(len(value)):
                mean, stdev = value[i]
                probabilities[label] *= self.guassian_probability(input_data[i], mean, stdev)
            return probabilities

    # 类别预测
    def predict(self, X_test):
        # sorted 后之后是一个list[]
        label = sorted(self.calculate_probabilities(X_test).items(), key=lambda x: x[1])[-1][0]
        return label

    def score(self, X_test, y_test):
        right = 0
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right += 1

        return right / float(len(X_test))


model = NaiveBayes()
model.fit(X_train, y_train)
print(model.predict([4.4, 3.2, 1.3, 0.2]))
print(model.score(X_test,y_test))
