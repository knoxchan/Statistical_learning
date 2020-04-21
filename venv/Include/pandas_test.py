from pandas import DataFrame, Series
import pandas as pd
import numpy as np
import re

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

import json

db = json.load(open('usda_food.json'))
# print(db[0].keys())
# db 有 ['id', 'description', 'tags', 'manufacturer', 'group', 'portions', 'nutrients']

info_keys = ['description', 'group', 'id', 'manufacturer']

info = DataFrame(db, columns=info_keys)
# print(pd.value_counts(info.manufacturer)[:10])

nutrients = []

for rec in db:
    fnuts = DataFrame(rec['nutrients'])
    fnuts['id'] = rec['id']
    nutrients.append(fnuts)

nutrients = pd.concat(nutrients, ignore_index=True)

# 查看nutrients 是否有重复项
# print(nutrients.duplicated().sum())
# 有重复项 14179
# 使用 drop_duplicates()方法 去除重复项
nutrients = nutrients.drop_duplicates()

# 现在需要姜nutrients 和 info 进行df的合并
# 因两个df对象中都有group 和 description 为了明确对象，我们需要对其进行重命名
col_mapping = {'description': 'food',
               'group': 'fgroup'}
info = info.rename(columns=col_mapping, copy=False)
col_mapping = {'description': 'nutrient',
               'group': 'nutgroup'}
nutrients = nutrients.rename(columns=col_mapping, copy=False)

ndata = pd.merge(nutrients,info,on='id',how='outer')

print(ndata.info())

result = ndata.groupby(['nutrient','fgroup'])['value'].quantile(0.5)

# print(result)
print(result['Zinc, Zn'])
