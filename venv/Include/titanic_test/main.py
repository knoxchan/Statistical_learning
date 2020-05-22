''' import basic modules '''
import numpy as np  # for linear algebra
import pandas as pd  # for data manipulation【操控】
import matplotlib.pyplot as plt  # for 2D visualization
import seaborn as sns
from scipy import stats  # for statistics【统计学】

''' Plotly visualization '''
import plotly.graph_objs as go
from plotly.tools import make_subplots

# from plotly.offline import iplot, init_notebook_mode

# init_notebook_mode(connected=True)  # # Required to use plotly offline in jupyter notebook
# 需要再jupyter notebook 上使用plotly离线模式

''' machine learning models '''  # ensemble 【集成学习】
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier

''' Classification (evaluation) metrices '''  # metrices 【衡量指标】
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score

''' Ensembling '''  # Ensembling learning 【集成学习】
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.ensemble import BaggingClassifier
from mlens.ensemble import BlendEnsemble
from vecstack import stacking

''' Customize visualization '''
plt.style.use('bmh')  # 用bmh风格画图
sns.set_style({'axes_grid': False})  # removing gridlines 删除网格线

# 导入数据
''' Read and preview the train data from csv file'''
train = pd.read_csv('./train.csv')
print('preview of train data')
print(train.head(2))

"""Read and preview the test from csv file."""
test = pd.read_csv('./test.csv')
print('preview of test data')
print(test.head(2))

'''merge train and test data together. This eliminates【消除】the hassle【麻烦】of handling train and test data seperately for various analysis'''
merged = pd.concat([train, test], sort=False).reset_index(drop=True)
print('preview of merged data')
print(merged.head(5))

''' shape of combined data '''
print('shape of combined data')
# print(merged.shape)

''' Variables in the combined data '''
print('Name of the Variables in merged data ')
# print(merged.columns)

'''
PassengerId : unique identifying number assigned to each passager 乘客的唯一ID
Sruvived : 乘客是否存活标志位 1:存活 0:死亡
Pclass : 乘客舱位   1:一等舱     2:二等舱   3:三等舱
Name : 乘客名字  male:男     female:女
Sex : 乘客性别
Age : 乘客年龄
SibSp : 船上兄弟姐妹/配偶人数
Parch : 船上父母/孩子人数
Ticket : 乘客船票号码
Fare : 乘客买船票花费的钱
Cabin : 乘客占用的客舱类别
Embarked : 乘客出发的港口 C = Cherbourg, Q = Queenstown, S = Southampton

变量类型：
    分类变量: Survived Pclass Name Sex SibSp Parch Ticket Cabin Embarked
    数字变量: Fare Age PassengerId
'''

''' pandas data types for our different variables '''
# print('data types of our variables')
# print(merged.dtypes)

''' 
int data type variables : Pclass SibSp Parch PassengerId
float data type variables : Fare Age Survied(due to concatenation)
object data[number + string] type variables : Name Sex Ticket Cabin Embarked
'''

''' Univariate【单变量】 analysis '''


# create a function to plot a variable's absolute and relative frequency
def plotFrequency(variables):
    '''plot absolute and relative frequency of a variable'''

    # Calculates absolute frequency
    absFreq = variables.value_counts()

    # Calculates relative frequency 【normalize=True】计数占比
    relFreq = variables.value_counts(normalize=True).round(4) * 100

    # Creates a dataframe off absolute and relative frequency
    df = pd.DataFrame({
        "absoluteFrequency": absFreq,
        "relativeFrequency": relFreq
    })

    # 图片初始化 添加大标题
    fig = plt.figure()
    fig.suptitle(variables.name)

    # create two subplots of bar chart
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    # 画图
    ax1.bar(df.absoluteFrequency.index, df.absoluteFrequency, width=0.5, tick_label=df.absoluteFrequency.index,
            color='r')
    ax2.bar(df.relativeFrequency.index, df.relativeFrequency, width=0.5, tick_label=df.relativeFrequency.index,
            color='g')

    # 数字显示
    for a, b in zip(df.absoluteFrequency.index, df.absoluteFrequency):
        ax1.text(a, b, '%.2f' % b, ha='center', va='bottom')
    for a, b in zip(df.relativeFrequency.index, df.relativeFrequency):
        ax2.text(a, b, '%.2f' % b, ha='center', va='bottom')

    # 标题设置
    ax1.set_title('Abs Freq')
    ax2.set_title('Rel Freq(%)')

    return plt.show()


'''plot absolute and realtive frequebcy'''
# plotFrequency(merged.Survived)
# plotFrequency(merged.Sex)
# plotFrequency(merged.Pclass)
# plotFrequency(merged.Embarked)

''' Cabin '''
absFreqCabin = merged.Cabin.value_counts(dropna=False)
print(absFreqCabin.head())
print(absFreqCabin.count())

''' Name '''
print('Total categories in Name')
print(merged.Name.value_counts().count())
print(merged.Name.head(7))

# 有1307个名字在数据表里面，我们需要对名字进行预处理 才可以知道名字和存货情况有没有关系

''' Ticket '''
print('Total group in ticket')
print(merged.Ticket.value_counts().count())
print(merged.Ticket.head())

# 得知船票也有许多独特的类型，我们也需要对其进行数据预处理

''' SibSp 船上兄弟姐妹/配偶人数'''
# plotFrequency(merged.SibSp)

''' Parch 船上父母/孩子人数'''


# plotFrequency(merged.Parch)

def plotHistogram(variables):
    '''plot histogram and density plot of a variable'''

    # 图片初始化 添加大标题
    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(variables.name, y=0.02)

    # 创建子图
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    # bin = variables.max() - variables.min()

    (count1, bin1, patch1) = ax1.hist(variables, bins=12)
    (count2, bin2, patch2) = ax2.hist(variables, density=True, bins=12)

    # X轴标签设置
    ax1.set_xticks(bin1)
    ax2.set_xticks(bin2)

    x_coordinate = [(bin1[1] / 2) + (bin1[1] * i) for i in range(len(bin1))]
    # 数字显示
    for a, b in zip(x_coordinate, count1):
        ax1.text(a, b, '%.2f' % b, ha='center', va='bottom')

    # 标题设置
    ax1.set_title('Abs Freq')
    ax2.set_title('Rel Freq(%)')

    return plt.show()


def calculateSummaryStats(variable):
    # skewness 偏度  x > 1 or x < -1 高度偏斜 x > 0.5 or x < -0.5 中度偏斜 -0.5 < x < 0.5 基本对称 无偏斜
    stats = variable.describe()
    skewness = pd.Series(variable.skew(), index=["skewness"])
    statsDf = pd.DataFrame(pd.concat([skewness, stats], sort=False), columns=[variable.name])
    statsDf = statsDf.reset_index().rename(columns={"index": "summaryStats"})
    return print(statsDf.round(2))


''' Fare 票价'''
# plotHistogram(merged.Fare)
# calculateSummaryStats(merged.Fare)

''' Age '''
# plotHistogram(merged.Age)
calculateSummaryStats(merged.Age)

## Feature Engineering

''' process Cabin '''
# 使用仓位前面的代号替代仓位 并用X表示缺失值
merged["cabinProcessed"] = merged.Cabin.str.get(0)
merged["cabinProcessed"].fillna('X', inplace=True)
print("Cabin Categories after Processing:")
print(merged.cabinProcessed.value_counts())
# plotFrequency(merged.cabinProcessed)

''' process Name '''
print(merged.Name.head(10))
firstName = merged.Name.str.split(".").str.get(0).str.split(",").str.get(-1)
print(firstName.value_counts())
# 有几个名字标签出现的机率是比较低的，所以我们把它统合在一个标签里
# 将博士、牧师、上校、少校、上尉 统合为 军官officer
firstName.replace(to_replace=['Dr', 'Rev', 'Col', 'Major', 'Capt'], value='officer', inplace=True, regex=True)

# 将多纳，乔克希尔，伯爵夫人，先生，夫人，唐 统合为 有权贵族
firstName.replace(to_replace=['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value='Aristocrat', inplace=True,
                  regex=True)

# 将 mlle 和 ms 替换成miss 将mme 替换成 mrs
firstName.replace(to_replace={'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'}, inplace=True, regex=True)

# 用贵族替代贵族
firstName.replace({"the Aristocrat": "Aristocrat"}, inplace=True, regex=True)

"""Insert a column named 'nameProcessed'."""
merged["nameProcessed"] = firstName

print(merged['nameProcessed'].value_counts())

''' Process SibSp & Parch '''
''' SibSp 船上兄弟姐妹/配偶人数'''
''' Parch 船上父母/孩子人数'''

# 这两个变量共同表示一个家庭的大小，所以我们从这两个变量中创建出一个新的变量family_size
merged['familySize'] = merged.SibSp + merged.Parch + 1  # +1 加上自己本人
print(merged['familySize'].value_counts())
# 家庭大小在 1 -- 11 不等 我们将他分割成4部分
merged.familySize.replace(to_replace=[1], value='single', inplace=True)  # 单身家庭
merged.familySize.replace(to_replace=[2, 3], value='small', inplace=True)  # 小家庭
merged.familySize.replace(to_replace=[4, 5], value='medium', inplace=True)  # 中等家庭
merged.familySize.replace(to_replace=[6, 7, 8, 11], value='large', inplace=True)  # 大家庭

print(merged.familySize.value_counts())

''' process Ticket【船票号码】 '''
# print(merged.Ticket.head())
# Ticket也是一个由字母和数字组成的变量,我们将创建两个组，一个存放带有字母的船票，另一个是纯数字的船票。
otherwise = merged.Ticket.str.split(" ").str.get(0).str.get(0)  # This extracts the 1st character
# np.where 用法 np.where(condition, x, y) 满足条件(condition)，输出x，不满足输出y。
merged["ticketProcessed"] = np.where(merged.Ticket.str.isdigit(), "N", otherwise)

print(merged.ticketProcessed.value_counts())
# plotFrequency(merged.ticketProcessed)
# plt.plot()

''' Outliers Detection【异常值检测】'''


def removeOutliers(variable):
    try:
        ''' 使用4分位方法[IQR]计算与删除异常值'''

        # 计算1，3分位值
        q1, q3 = variable.quantile(0.25), variable.quantile(0.75)
        iqr = q3 - q1

        # calculate lower fence and upper fence for outliers
        lowerfence, upperfence = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        # Observations that are outliers
        outliers = variable[(variable < lowerfence) | (variable > upperfence)]

        # 删除outliers数据
        filtered = variable.drop(outliers.index, axis=0).reset_index(drop=True)

        return filtered
    except Exception as e:
        print(e)


'''2.Create another function to plot boxplot with and without outliers.'''


def plotBoxPlot(variable, filtedVarable):
    ''' 画出不带异常值的箱形图
    We will also use the output of removeOutliers function as the input to this function.
    variable = variable with outliers,
    filteredVariable = variable without outliers'''

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(variable.name, y=0.02)

    # 创建子图
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ax1.boxplot(variable, vert=False, labels=['natural'])
    ax2.boxplot(filtedVarable, vert=False, labels=['filted'])

    return fig.show()


'''Plot Age with and without outliers. '''
plotBoxPlot(merged.Age.dropna(), removeOutliers(merged.Age.dropna()))

''' plot Fare '''
plotBoxPlot(merged.Fare.dropna(), removeOutliers(merged.Fare.dropna()))

''' 处理缺失变量'''


# 1.计算缺失值数量
def calculateMissingValues(variable):
    return merged.isna().sum()[merged.isna().sum() > 0]


def plotScatterPlot(x, y, title, yaxis):
    fig = plt.figure()
    plt.scatter(x, y)
    fig.title = title
    fig.ylabel = yaxis

    return fig.show()


# 展示merged数据缺失值情况 age有263个缺失值 Fare有1个缺失值 Cabin有1014个缺失值 Embarked有2个缺失值
plotScatterPlot(calculateMissingValues(merged).index, calculateMissingValues(merged), 'x', 'y')

# 缺失值填充 Embarked & Fare
merged.Embarked.fillna(value='S', inplace=True)
merged.Fare.fillna(value=merged.Fare.median(),inplace=True)

# 缺失值填充 Age

# 第一步 先研究age和分布与什么特征高度相关
def plotAgeCorrelated():
    toSearch = merged.loc[:, ["Sex", "Pclass", "Embarked", "nameProcessed", "familySize", "Parch",
                                 "SibSp", "cabinProcessed", "ticketProcessed"]]

    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (25,25))
    for ax, column in zip(axes.flatten(), toSearch.columns):
        sns.boxplot(x = toSearch[column], y = merged.Age, ax = ax)
        ax.set_title(column, fontsize = 23)
        ax.tick_params(axis = "both", which = "major", labelsize = 20)
        ax.set_ylabel("Age", fontsize = 20)
        ax.set_xlabel("")
    fig.suptitle("Variables Associated with Age", fontsize = 30)
    fig.tight_layout(rect = [0, 0.03, 1, 0.95])
    fig.show()

plotAgeCorrelated()
# 结果表明 age性别上的年龄分布几乎是相同的 Embarked出发港口上的娘零分布也是相同的，所以性别和上船点不能很好的估算年龄
# 另一方面 Pclass的1，2，3三个亚群中年龄分布较明显，最后 nameProcess,familySize,SibSp,Parch、cabinProcessd在不同类别中
# 是不同的，所以他们也是预测年龄的好方法