from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

df1 = DataFrame({'key': ['b', 'b', 'a', 'c', 'a', 'b'],
                 'data1': range(6)})

df2 = DataFrame({'key': ['a', 'b', 'a', 'b', 'd'],
                 'data2': range(5)})
# print(df1)
# print(df2)
# print(pd.merge(df1,df2))

left = DataFrame({'key1': ['foo', 'foo', 'bar'],
                  'key2': ['one', 'two', 'one'],
                  'lval': [1, 2, 3]})

right = DataFrame({'key1': ['foo', 'foo', 'bar', 'bar'],
                   'key2': ['one', 'one', 'one', 'two'],
                   'rval': [4, 5, 6, 7]})
df3 = pd.merge(left, right, on=['key1', 'key2'], how='outer')

left1 = DataFrame({'key1': ['a', 'b', 'a', 'a', 'b', 'c'],
                   'value': range(6)})
right1 = DataFrame({'group_val': [3.5, 7]}, index=['a', 'b'])

df4 = pd.merge(left1, right1, left_on='key1', right_index=True)

lefth = DataFrame({'key1': ['cjp', 'cjp', 'cjp', 'hyy', 'hyy'],
                   'key2': [2000, 2001, 2002, 2001, 2002],
                   'data': np.arange(5.)})
righth = DataFrame(np.arange(12).reshape((6, 2)),
                   index=[['hyy', 'hyy', 'cjp', 'cjp', 'cjp', 'cjp'],
                          [2001, 2000, 2000, 2000, 2001, 2002]],
                   columns=['event1', 'event2'])
df5 = pd.merge(lefth, righth, left_on=['key1', 'key2'], right_index=True, how='outer')

df1 = DataFrame(np.random.randn(3, 4), columns=['a', 'b', 'c', 'd'])
df2 = DataFrame(np.random.randn(2, 3), columns=['b', 'd', 'a'])

a = Series([np.nan, 2.5, np.nan, 3.5, 4.5, np.nan], index=list('fedcba'))
b = Series(np.arange(len(a)), dtype=np.float64, index=list('fedcba'))
b[-1] = np.nan

data = DataFrame(np.arange(6).reshape((2, 3)),
                 index=pd.Index(['cjp', 'hyy'], name='name'),
                 columns=pd.Index(['one', 'two', 'three'], name='number'))
data = data.rename(index={'cjp': 'dddd'})
# print(data)

data = {'cjp': 'cjp@google.com', 'hyy': 'hyy@google.com', 'gy': 'gy@163.com', 'cxx': np.nan}
data = Series(data)
# print(data)
# print(data.str.contains('163'))
pattern = '([A-z0-9.%+-]+)@([a-z0-9.-]+)\\.([A-Z]{2,4})'
matches = data.str.findall(pattern, flags=re.IGNORECASE)
# print(matches.str[0])
#
# import json
#
# db = json.load(open('usda_food.json'))
# # print(db[0].keys())
# # db 有 ['id', 'description', 'tags', 'manufacturer', 'group', 'portions', 'nutrients']
#
# info_keys = ['description', 'group', 'id', 'manufacturer']
#
# info = DataFrame(db, columns=info_keys)
# # print(pd.value_counts(info.manufacturer)[:10])
#
# nutrients = []
#
# for rec in db:
#     fnuts = DataFrame(rec['nutrients'])
#     fnuts['id'] = rec['id']
#     nutrients.append(fnuts)
#
# nutrients = pd.concat(nutrients, ignore_index=True)
#
# # 查看nutrients 是否有重复项
# # print(nutrients.duplicated().sum())
# # 有重复项 14179
# # 使用 drop_duplicates()方法 去除重复项
# nutrients = nutrients.drop_duplicates()
#
# # 现在需要姜nutrients 和 info 进行df的合并
# # 因两个df对象中都有group 和 description 为了明确对象，我们需要对其进行重命名
# col_mapping = {'description': 'food',
#                'group': 'fgroup'}
# info = info.rename(columns=col_mapping, copy=False)
# col_mapping = {'description': 'nutrient',
#                'group': 'nutgroup'}
# nutrients = nutrients.rename(columns=col_mapping, copy=False)
#
# ndata = pd.merge(nutrients,info,on='id',how='outer')
#
# print(ndata.info())
#
# result = ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)

# print(result)
# print(result['Zinc, Zn'])
# result['Zinc, Zn'].sort_values().plot(kind='barh')


# pandas - matplotplib
# df = DataFrame(np.random.randn(10, 4).cumsum(0),
#                columns=['A', 'B', 'C', 'D'],
#                index=np.arange(0, 100, 10))
# df.plot()

tips = pd.read_csv('tips.csv')
# party_count = pd.crosstab(tips.sizes, tips.day)
# # party_counts = party_count.loc[:,2:5]
# party_pcts = party_count.div(party_count.sum(1).astype(float), axis=0)
# print(party_pcts)
# party_pcts.plot(kind='bar', stacked=True)
tips['tip_pct'] = tips['tip'] / tips['total_bill']

comp1 = np.random.normal(0, 1, size=200)  # N(0,1)
comp2 = np.random.normal(10, 2, size=200)  # N(10,4)
values = Series(np.concatenate([comp1, comp2]))
# values.hist(bins=200,alpha=0.3,color='k',density=False)
# values.plot(kind='kde',style='k--')

mocro = pd.read_csv('macrodata.csv')
data = mocro[['cpi', 'm1', 'tbilrate', 'unemp']]

# 计算对数差
trans_data = np.log(data).diff().dropna()
# print(trans_data[-5:])
#
# pd.plotting.scatter_matrix(trans_data,diagonal='kde',color='k',alpha=0.3)

# haiti 地震数据
data = pd.read_csv('haiti.csv')
# print(data.info())
# # data = data[['INCIDENT DATE','LATITUDE','LONGITUDE']]
# print(data.describe())
# print(data.CATEGORY[:10])
data = data[(data.LATITUDE < 20) & (data.LONGITUDE < -70) & data.CATEGORY.notnull()]

# 事件序列
from datetime import datetime

dates = [datetime(2011, 1, 2), datetime(2011, 1, 5), datetime(2011, 1, 7), datetime(2011, 1, 8), datetime(2011, 1, 10),
         datetime(2011, 1, 12)]
ts = Series(np.random.randn(6),index=dates)

ts = Series(np.random.randn(4),index=pd.date_range('1/1/2000',periods=4,freq='M'))


data = {'a':[11,22,33,44,55],'b':[66,77,88,99,100]}
df = DataFrame(data,index=['cjp','knox','amy','lilian','jeremy'])
# print(df)
# print(df.loc['cjp':'amy'])
# print(df.iloc[0:3])

data2 = {'a':'ssss','b':'adsSSSDFW'}
s = Series(data2)
print(s.str.capitalize())