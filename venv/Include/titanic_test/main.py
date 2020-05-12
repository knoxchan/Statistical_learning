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

'''merge train and test data together. This eliminates【消除】 the hassle【麻烦】 of handling train and test data seperately for various analysis'''
merged = pd.concat([train,test],sort=False).reset_index(drop=True)