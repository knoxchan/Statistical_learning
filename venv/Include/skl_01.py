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
    grid.fit(X, y)
    print(grid.best_params_)
    model = grid.best_estimator_

    plt.scatter(X.ravel(), y)
    lim = plt.axis()
    y_test = model.fit(X, y).predict(X_test)
    plt.plot(X_test.ravel(), y_test)
    plt.axis(lim)

    plt.show()


# 专题 朴素贝叶斯分类
# 实例8 高斯朴素贝叶斯
def example_08():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu')

    # 计算模型边界
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X, y)

    rng = np.random.RandomState(0)
    Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
    ynew = model.predict(Xnew)

    lim = plt.axis()
    plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.5)
    plt.axis(lim)

    plt.show()

    # 可以使用predict_proba方法计算样本标签概率 【round保留小数位】
    yprob = model.predict_proba(Xnew)
    print(yprob[-8:].round(2))


# 多项式朴素贝叶斯：案例 文本分类
def example_09():
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(subset='train')
    test = fetch_20newsgroups(subset='test')
    # print(train.data[5])

    # 为了使文本数据可以用于机器学习，需要将字符串内容向量化
    # 可以创建一个管道，将TF-IDF向量话方法 和 多项式朴素贝叶斯分类器组合在一起
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(train.data, train.target)
    label = model.predict(test.data)

    # 可以使用 accuracy_score 查看分数
    from sklearn.metrics import accuracy_score
    print(accuracy_score(label, test.target))

    # 正确率为77% 我们可以使用混淆矩阵统计真实标签和预测标签的结果
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(test.target, label)
    plt.figure(figsize=(8, 8))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train.target_names,
                yticklabels=train.target_names)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

    plt.show()


# 专题 线性回归
# 简单线性回归
def example_10():
    rng = np.random.RandomState(1)
    x = 10 * rng.rand(50)
    y = 2 * x - 5 + rng.randn(50)
    plt.scatter(x, y)
    # 使用linearRegression模型来拟合数据
    from sklearn.linear_model import LinearRegression
    # fit_intercept 是否计算截距 默认False 如果false 回归线过原点
    model = LinearRegression(fit_intercept=True)
    model.fit(x[:, np.newaxis], y)
    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])
    plt.plot(xfit, yfit)

    plt.show()

    # 数据的斜率和截距都在模型的拟合参数中， coef_取[0]因为这是一次函数 只有一个w
    print('model slope:', model.coef_[0])
    print('model intercept:', model.intercept_)


# 多项式基函数
def example_11():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    # x = np.array([2, 3, 4])
    # # include_bias 是否包含0次幂项
    # poly = PolynomialFeatures(3, include_bias=False)
    # a = poly.fit_transform(x[:,None])
    from sklearn.pipeline import make_pipeline
    poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())

    rng = np.random.RandomState(1)
    x = 10 * rng.rand(50)
    y = np.sin(x) + 0.1 * rng.randn(50)
    poly_model.fit(x[:, np.newaxis], y)
    xfit = np.linspace(0, 10, 1000)
    yfit = poly_model.predict(xfit[:, np.newaxis])

    plt.scatter(x, y)
    plt.plot(xfit, yfit)
    plt.show()


# 专题 支持向量机

def plot_svc_decision_function(model, ax=None, plot_support=True):
    ''' 画二维SVC的决策函数'''
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建评估模型的网络
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    p = model.decision_function(xy).reshape(X.shape)

    # 画决策边界和边界
    ax.contour(X, Y, p, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    # 画支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=300, linewidth=1, facecolors='none',
                   edgecolor='g')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# 拟合支持向量机
def example_12():
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

    from sklearn.svm import SVC
    model = SVC(kernel='linear', C=1E10)
    model.fit(X, y)

    plot_svc_decision_function(model)

    plt.show()


# 超越线性边界 核函数SVM模型
def example_13():
    from sklearn.datasets import make_circles
    from sklearn.svm import SVC
    X, y = make_circles(100, factor=.1, noise=.1)

    r = np.exp(-(X ** 2)).sum(1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    # plot_svc_decision_function(clf,plot_support=False)
    # ax = plt.axes(projection='3d')
    # ax.scatter3D(X[:, 0], X[:, 1], r, c=y,s=50, cmap='autumn')

    clf = SVC(kernel='rbf', C=1E6)
    clf.fit(X, y)
    plot_svc_decision_function(clf)

    plt.show()


# SVM优划 软化边界
def example_14():
    from sklearn.datasets import make_blobs
    from sklearn.svm import SVC
    X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
    # plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)
    for axi, C in zip(ax, [10, 0.1]):
        model = SVC(kernel='linear', C=C).fit(X, y)
        axi.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
        plot_svc_decision_function(model, axi)
        axi.set_title('C = {0:.1f}'.format(C), size=14)
    plt.show()


# SVM案例 人脸识别
def example_15():
    from sklearn.datasets import fetch_lfw_people
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.decomposition import PCA
    faces = fetch_lfw_people(min_faces_per_person=60)

    # print(faces.target_names)
    # print(faces.images.shape)

    # fig, ax = plt.subplots(3, 5)
    # for i, axi in enumerate(ax.flat):
    #     axi.imshow(faces.images[i],cmap='bone')
    #     axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])

    # 通过PCA 选取150个基本元素然后将其交给支持向量机分类器
    pca = PCA(n_components=150, whiten=True, random_state=42)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)

    # 数据读取 与 数据分割
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

    # 通过网格搜索交叉检验 来寻找最优参数组合，通过不断调整C(svm 惩罚参数)，gamma(控制径向基函数核大小)
    from sklearn.model_selection import GridSearchCV
    param_grid = {'svc__C': [1, 5, 10, 50],
                  'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
    grid = GridSearchCV(model, param_grid)
    grid.fit(Xtrain, ytrain)
    print(grid.best_params_)
    # 使用最优参数模型进行数据预测
    model = grid.best_estimator_
    yfit = model.predict(Xtest)
    fig, ax = plt.subplots(4, 6)
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i], cmap='bone')
        axi.set(xticks=[], yticks=[])
        axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                       color='black' if yfit[i] == ytest[i] else 'red')
    fig.suptitle('predicted names; Incorrect Labels in Red', size=14)

    # 打印分类效果报告
    from sklearn.metrics import classification_report
    print(classification_report(ytest, yfit, target_names=faces.target_names))

    # 画出这个标签的混淆矩阵
    plt.figure(figsize=(10, 10))
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(ytest, yfit)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=faces.target_names,
                yticklabels=faces.target_names)
    plt.xlabel('ture label')
    plt.ylabel('predict label')

    plt.show()


# 决策树 和 随机森林
# 随机森林是建立在决策树基础上的集成学习器

# 辅助函数 对分类器的结果进行可视化
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()

    # 画出训练数据
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 用评估器拟合数据
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # 为结果生成彩色图
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap, clim=(y.min(), y.max()),
                           zorder=1)
    ax.set(xlim=xlim, ylim=ylim)


# 创建一颗决策树
def example_16():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=360, centers=4, random_state=0, cluster_std=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')

    # 导入决策树对数据进行拟合
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier().fit(X, y)
    visualize_classifier(tree, X, y)
    plt.show()


# 评估器集成算法 随机森林
# 使用baggingclassifier元评估器实现装袋分类器
def example_17():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import BaggingClassifier

    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=360, centers=4, random_state=0, cluster_std=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')

    tree = DecisionTreeClassifier()
    bag = BaggingClassifier(tree, n_estimators=500, max_samples=0.8, random_state=1)

    visualize_classifier(bag, X, y)

    plt.show()


# 使用随机森林评估器
def example_18():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=360, centers=4, random_state=0, cluster_std=1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='rainbow')

    model = RandomForestClassifier(n_estimators=100, random_state=0)
    visualize_classifier(model, X, y)
    plt.show()


# 随机森林回归
def example_19():
    rng = np.random.RandomState(42)
    x = 10 * rng.rand(200)

    def model(x, sigma=0.3):
        fast_oscillation = np.sin(5 * x)
        slow_oscillation = np.sin(0.5 * x)
        noise = sigma * rng.randn(len(x))

        return fast_oscillation + slow_oscillation

    y = model(x)

    # 误差棒图
    # plt.errorbar(x,y,0.3,fmt='o')

    from sklearn.ensemble import RandomForestRegressor
    forest = RandomForestRegressor(200)
    forest.fit(x[:, np.newaxis], y)

    xfit = np.linspace(0, 10, 1000)
    yfit = forest.predict(xfit[:, np.newaxis])
    yture = model(xfit, sigma=0)

    # 红色 预测曲线
    plt.plot(xfit, yfit, '-r')

    plt.plot(xfit, yture, '-k', alpha=0.5)

    plt.show()


# 案例 用随机森林识别手写数字
def example_20():
    from sklearn.datasets import load_digits
    digits = load_digits()
    # print(digits.keys())
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

    # 打印数字图片 每个数字是8 * 8 像素
    for i in range(64):
        ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
        ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')

        # 用target的值给图像作标注
        ax.text(0,7,str(digits.target[i]))

    # 用随机森林快读对数字进行分类
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)

    model = RandomForestClassifier(n_estimators=1000)
    model.fit(Xtrain,ytrain)
    ypred = model.predict(Xtest)

    # 查看分类结果报告
    from sklearn.metrics import classification_report
    print(classification_report(ytest,ypred))

    # 查看混淆矩阵
    from sklearn.metrics import confusion_matrix
    mat = confusion_matrix(ytest,ypred)
    plt.figure(figsize=(8,8))
    sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False)
    plt.xlabel('ture label')
    plt.ylabel('predict label')

    plt.show()


if __name__ == '__main__':
    example_20()
