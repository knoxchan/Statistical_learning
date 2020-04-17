from pandas import DataFrame,Series
import pandas as pd
import numpy as np

df1 = DataFrame({'key':['b','b','a','c','a','b'],
                 'data1':range(6)})

df2 = DataFrame({'key':['a','b','a','b','d'],
                 'data2':range(5)})
# print(df1)
# print(df2)
# print(pd.merge(df1,df2))

left = DataFrame({'key1':['foo','foo','bar'],
                  'key2':['one','two','one'],
                  'lval':[1,2,3]})

right = DataFrame({'key1':['foo','foo','bar','bar'],
                  'key2':['one','one','one','two'],
                  'rval':[4,5,6,7]})
df3 = pd.merge(left,right,on=['key1','key2'],how='outer')

left1 = DataFrame({'key1':['a','b','a','a','b','c'],
                   'value':range(6)})
right1 = DataFrame({'group_val':[3.5,7]},index=['a','b'])

df4 = pd.merge(left1,right1,left_on='key1',right_index=True)

lefth = DataFrame({'key1':['cjp','cjp','cjp','hyy','hyy'],
                   'key2':[2000,2001,2002,2001,2002],
                   'data':range(5)})
print(lefth)
righth = DataFrame(np.arange(12).reshape((6,2)),
                   index=[['cjp','cjp','hyy','hyy','hyy','hyy'],
                          [2001,2000,2000,2000,2001,2002]],
                   columns=['event1','event2'])
print(righth)
df5 = pd.merge(lefth,righth,left_on=['key1','key2'],right_index=True)
print(df5)