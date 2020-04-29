import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
from sklearn.datasets import load_iris, load_digits


# iris = sns.load_dataset('iris')
# sns.pairplot(iris,hue='species',size=1.5)
# plt.show()

# x_iris = iris.drop('species',axis=1)
# y_iris = iris['species']


# 实例1 简单线性回归
def example_01():
    from sklearn.linear_model import LinearRegression
    rng = np.random.RandomState(42)
    x = 10 * rng.rand(50)
    y = 2 * x - 1 + rng.rand(50)
    plt.scatter(x, y)

    model = LinearRegression(fit_intercept=True)
    X = x[:, np.newaxis]
    model.fit(X, y)
    w = model.coef_
    b = model.intercept_

    x2 = np.linspace(0, 10)
    y2 = w * x2 + b
    plt.plot(x2, y2)

    plt.show()


# 实例2 有监督学习示例：鸢尾花数据分类
def example_02():
    from sklearn.model_selection import train_test_split

    # 0 数据初始化
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length ', 'sepal width ', 'petal length ', 'petal width', 'label']
    X_iris, y_iris = df.drop('label', axis=1), df['label']
    X_train, X_test, y_train, y_test = train_test_split(X_iris, y_iris, random_state=1)

    from sklearn.naive_bayes import GaussianNB  # 1 选择模型类
    model = GaussianNB()  # 2 初始化模型
    model.fit(X_train, y_train)  # 3 用模型对数据进行拟合
    y_model = model.predict(X_test)  # 4 对新数据进行预测

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, y_model))  # 5 得到分数


# 实例3 无监督学习示例：鸢尾花数据分类
def example_03():
    # 0 数据初始化
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length ', 'sepal width ', 'petal length ', 'petal width', 'label']
    X_iris, y_iris = df.drop('label', axis=1), df['label']

    from sklearn.decomposition import PCA  # 1 选者模型类
    model = PCA(n_components=2)  # 2 设置超参数，初始化模型
    model.fit(X_iris)  # 3 拟合数据，这里不需要Y变量
    X_2D = model.transform(X_iris)  # 4 将数据转换成2维
    # print(X_2D)
    X = X_2D[:, 0]
    Y = X_2D[:, 1]
    plt.scatter(X, Y)
    plt.show()
    from sklearn.model_selection import cross_validate


# 实例4 应用 手写数字探索
def example_04():
    digits = load_digits()
    # fig, axes = plt.subplots(10, 10, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []},
    #                          gridspec_kw=dict(hspace=0.1, wspace=0.1))
    #
    # # axes.flat 一维迭代器
    # for i, ax in enumerate(axes.flat):
    #     ax.imshow(digits.images[i], cmap='binary')
    #     ax.text(0.05, 0.05, str(digits.target[i]), transform=ax.transAxes, color='green')
    # plt.show()

    X = digits.data
    y = digits.target

    from sklearn.manifold import Isomap
    iso = Isomap(n_components=2)
    iso.fit(X)
    data_projected = iso.transform(X)

    # plt.scatter(data_projected[:, 0], data_projected[:, 1], c=y, edgecolors='none', alpha=0.5,
    #             cmap=plt.cm.get_cmap('Spectral', 10))
    # plt.colorbar(label='digit label', ticks=range(10))
    # plt.clim(-0.5, 9.5)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_model = model.predict(X_test)

    # 量化评分
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_model, y_test))

    # 得到结果是85%的正确率 但是还是不知道哪里出了问题
    # 解决这个问题的方法就是打印混淆矩阵
    from sklearn.metrics import confusion_matrix

    mat = confusion_matrix(y_test, y_model)
    sns.heatmap(mat, square=True, annot=True, cbar=True)
    plt.xlabel('predict value')
    plt.ylabel('True value')
    plt.show()


# 实例5 skl验证曲线
def example_05():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

    # 创造一些数据
    def make_data(N, err=1.0, rseed=1):
        # 随机抽样数据
        rng = np.random.RandomState(rseed)
        X = rng.rand(N, 1) ** 2
        y = 10 - 1. / (X.ravel() + 0.1)
        if err > 0:
            y += err * rng.rand(N)
        return X, y

    X, y = make_data(40)
    X_test = np.linspace(-0.1, 1.1, 500)[:, np.newaxis]

    plt.scatter(X.ravel(), y, color='black')
    axis = plt.axis()
    for dgree in [1, 3, 5]:
        y_test = PolynomialRegression(dgree).fit(X, y).predict(X_test)
        plt.plot(X_test.ravel(), y_test, label='dgree={}'.format(dgree))
        plt.xlim(-0.1, 1.0)
        plt.ylim(-2, 12)
        plt.legend(loc='best')
    plt.show()


# 实例6 skl验证曲线 补充
def example_06():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

    # 创造一些数据
    def make_data(N, err=1.0, rseed=1):
        # 随机抽样数据
        rng = np.random.RandomState(rseed)
        X = rng.rand(N, 1) ** 2
        y = 10 - 1. / (X.ravel() + 0.1)
        if err > 0:
            y += err * rng.rand(N)
        return X, y

    X, y = make_data(40)
    X_test = np.linspace(-0.1, 1.1, 500)[:, np.newaxis]

    # 到底多项式的次数是多少，模型才能在偏差和方差间达到平衡
    # 引入skl的validation_curve 只需要提供模型 数据 参数名称 和 验证范围
    from sklearn.model_selection import validation_curve
    degree = np.arange(0, 21)
    train_score, val_score = validation_curve(PolynomialRegression(), X, y, param_name='polynomialfeatures__degree',
                                              param_range=degree, cv=7)
    plt.plot(degree, np.median(train_score, 1), color='blue', label='train')
    plt.plot(degree, np.median(val_score, 1), color='red', label='validation')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlabel('dgree')
    plt.ylabel('score')
    plt.show()


# 实例6 skl验证曲线 补充 得到学习曲线图
def example_06():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

    # 创造一些数据
    def make_data(N, err=1.0, rseed=1):
        # 随机抽样数据
        rng = np.random.RandomState(rseed)
        X = rng.rand(N, 1) ** 2
        y = 10 - 1. / (X.ravel() + 0.1)
        if err > 0:
            y += err * rng.rand(N)
        return X, y

    X, y = make_data(40)
    X_test = np.linspace(-0.1, 1.1, 500)[:, np.newaxis]

    # 到底多项式的次数是多少，模型才能在偏差和方差间达到平衡
    # 引入skl的validation_curve 只需要提供模型 数据 参数名称 和 验证范围
    from sklearn.model_selection import validation_curve
    degree = np.arange(0, 21)
    train_score, val_score = validation_curve(PolynomialRegression(), X, y, param_name='polynomialfeatures__degree',
                                              param_range=degree, cv=7)
    plt.plot(degree, np.median(train_score, 1), color='blue', label='train')
    plt.plot(degree, np.median(val_score, 1), color='red', label='validation')
    plt.legend(loc='best')
    plt.ylim(0, 1)
    plt.xlabel('dgree')
    plt.ylabel('score')
    plt.show()


# 实例7 skl验证曲线 补充 网格搜索，寻找最优参数
def example_07():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    def PolynomialRegression(degree=2, **kwargs):
        return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))

    # 创造一些数据
    def make_data(N, err=1.0, rseed=1):
        # 随机抽样数据
        rng = np.random.RandomState(rseed)
        X = rng.rand(N, 1) ** 2
        y = 10 - 1. / (X.ravel() + 0.1)
        if err > 0:
            y += err * rng.rand(N)
        return X, y

    X, y = make_data(40)
    X_test = np.linspace(-0.1, 1.1, 500)[:, np.newaxis]

    # 在实际工作中，模型有多个得分转折点(多个超参数),因此曲线会从2维变成N维
    from sklearn.model_selection import GridSearchCV
    param_grid = {'polynomialfeatures__degree': np.arange(21),
                  'linearregression__fit_intercept': [True, False],
                  'linearregression__normalize': [True, False]}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=7)
    grid.fit(X,y)
    print(grid.best_params_)
    model = grid.best_estimator_

    plt.scatter(X.ravel(), y)
    lim = plt.axis()
    y_test = model.fit(X,y).predict(X_test)
    plt.plot(X_test.ravel(), y_test)
    plt.axis(lim)

    plt.show()


if __name__ == '__main__':
    example_07()
