import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import numpy as np

# 假设一天中每隔2小时(range(2,26,2))的气温 分别是
def temperature_show():
    x = range(2, 26, 2)
    y1 = [15, 13, 14.5, 17, 20, 25, 25, 26, 24, 22, 18, 15]
    y2 = [18, 11, 10, 12, 29, 22, 25, 22, 28, 23, 11, 19]

    # 1 设置图片大小 dpi：图像每英寸长度内的像素点数
    plt.figure(figsize=(20,8),dpi=80)

    # 4 设置X\Y轴的刻度
    # X轴根据x的值设置刻度
    plt.xticks(x)

    # 绘图
    plt.plot(x,y1,c='r',)
    plt.plot(x,y2,c='g')

    # 图片保存
    # plt.savefig(path)
    # 展示图形
    plt.show()
'''
我们还可以做什么？
1.设置图片大小
2.保存到本地
3.添加描述信息 X轴Y轴表示什么，这个图表示什么
4.调整X\Y的刻度间距
5.线条的样式，例如颜色 透明度等
6.标出特殊的点 例如 最高点 最低点
7.给图片添加防伪水印
'''

def test():
    fig = plt.figure(figsize=(20,10))
    # fig, ax = plt.subplots(1,3)
    # ax[0].plot(np.random.randn(100).cumsum(),'k',label='one')
    # ax[1].plot(np.random.randn(100).cumsum(),'k--',label='two')
    # ax[2].plot(np.random.randn(100).cumsum(),'k-',label='three')
    plt.plot(np.random.randn(100).cumsum(),'k',label='one')
    plt.plot(np.random.randn(100).cumsum(),'k--',label='two')
    plt.plot(np.random.randn(100).cumsum(),'k.',label='three')

    plt.text(50,8,'hello world!',fontsize=10)
    fig.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    test()




