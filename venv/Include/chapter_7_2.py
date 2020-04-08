# sklearn svm.svc 课后习题 1.2
# 导入基本库
import numpy as np
import pylab as pl
from sklearn import svm

# 每次程序运行时，产生的随机点都相同
np.random.seed(0)

# 产生40行随机坐标，且线性可区分, 以[2,2]为中心，随机产生上下40个线性可分的点，画出支持向量和所有的点
# x = np.r_[np.random.rand(20,2) - [2,2],np.random.rand(20,2) + [2,2]]
# y = [0]*20 + [1]*20
x=[[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]]
y=[1, 1, 1, -1, -1]
# 创建一个SVM分类器并进行预测
clf = svm.SVC(kernel='linear', C=10000)
clf.fit(x,y)

# 根据SVM分类器类参数，获取w_0,w_1，w3的值，并绘制出支持向量
# w_0*x + w_1 *y + w_3 = 0 --> y = -w0/w1*x - w_3/w_1
w = clf.coef_[0]
a = -w[0]/w[1]
b = -clf.intercept_[0]/w[1]
xx = np.linspace(-5, 5)
yy = a*xx + b

# 斜距式方程：y = kx + b，A(b[0],b[1])为一个支持向量点
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])

# 斜距式方程：y = kx  + b，B(b[0],b[1])为一个支持向量点
b = clf.support_vectors_[-1]
yy_up = a*xx + (b[1] - a*b[0])

#画出3条直线
pl.plot(xx,yy,'k-')
pl.plot(xx,yy_down,'k--')
pl.plot(xx,yy_up,'k--')

#画出支持向量点
pl.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1],
           s=150,facecolors = 'none', edgecolors='k')
pl.scatter([i[0] for i in x], [i[1] for i in x], c=y,cmap=pl.cm.Paired)

# 绘制平面图
pl.axis('tight')
pl.show()
