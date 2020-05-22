import numpy as np
import matplotlib.pyplot as plt
np.random.seed(100) #随机生成100个种子数

data = np.random.normal(size=1000,loc=0,scale =1)

plt.boxplot(data)
plt.show()