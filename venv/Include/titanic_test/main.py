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
    ax1.bar(df.absoluteFrequency.index,df.absoluteFrequency,width=0.5,tick_label=df.absoluteFrequency.index,color='r')
    ax2.bar(df.relativeFrequency.index,df.relativeFrequency,width=0.5,tick_label=df.relativeFrequency.index,color='g')

    # 数字显示
    for a,b in zip(df.absoluteFrequency.index,df.absoluteFrequency):
        ax1.text(a,b,'%.2f'%b,ha='center',va='bottom')
    for a,b in zip(df.relativeFrequency.index,df.relativeFrequency):
        ax2.text(a,b,'%.2f'%b,ha='center',va='bottom')

    # 标题设置
    ax1.set_title('Abs Freq')
    ax2.set_title('Rel Freq(%)')

    return plt.show()

'''plot absolute and realtive frequebcy'''
plotFrequency(merged.Survived)
plotFrequency(merged.Sex)
plotFrequency(merged.Pclass)
plotFrequency(merged.Embarked)

''' Cabin '''
absFreqCabin = merged.Cabin.value_counts(dropna=False)
print(absFreqCabin.head())
print(absFreqCabin.count())

