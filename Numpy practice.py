from array import array
from importlib.metadata import PathDistribution
import this
from tkinter.tix import COLUMN
from unittest import FunctionTestCase
import pandas as pd
import numpy as np
import os
import math

a = np.arange(5,6,2)
b = np.arange(3,10,1)

c = a+b
for rows in a:
    print(rows)
for (x,y) in (a,b):
    print(x,y)

i = np.linspace(2,6,50)
i.reshape(2,5)

df = pd.read_csv("xyz.csv", header = True, index= False)
df.shape
df.reindex('1,2,3')
df.groupby(by= 'stage')
df.to_csv('abc.csv')
df.strip('xy')

df.dropna(how= all, thresh=2)
df.merge(a, b, on= 'sku', suffixes= True, indicator = True, how= 'inner')
df.melt(value_name='', vars_name='')
df.stack(df, df)
df.pivot(column='', values='', agg_func =sum)
df.concat([df,df], axis=0, ignore_index=True)



# OS Modules
os.getcwd()
os.chdir()
os.rename()
os.mkdir('now')
os.makedirs('for/this')
os.environ.get('C://')
os.path.isfile()
os.path.exists()

# numpy functions

np.vstack(x)
np.hstack(x)
np.hsplit(x)
np.vsplit(x)
np.shape
np.arange(1000).reshape(2,500)
np.ravel('a')
np.size
np.itemsize()
np.sum()
np.loadtxt('xyz.txt', skiprows= 3, delimiter=0, dtype=int)
np.cumsum(a, axis=1)
np.cumsum(a, axis=0)
y = lambda y:x*x
#array
arr1 = np.array([1,2,3], [4,5,6], [7,8,9])
z= lambda z:x+y*x*z 

np.range([])
def my_func(x):
    return x*x
result= my_func(5)
print(result)

for (x,y) in np.nditer([a, b]):
    print(x,y)
for x in np.nditer(a, order= 'F', flags='external_loop'):
    print(x)
for x in np.nditer(a, op_flags=['readwrite']):
    print(x)
m = np.zeros((3,4))
m
o = np.ones((4,5))
o
o.dtype
np.std(o)
import numpy as np
np.linspace(5,6,9)

pip install matplotlib
import matplotlib.pyplot as plt

x1 = np.array([2, 5, 255])
x2 = np.array([4, 4, 4])
x1|x2
x1*x2
numpy.bitwise_or(x1, x2, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'bitwise_or'>
np.bitwise_or(np.array([2, 5, 255, 2147483647], dtype=np.int32),np.array([4, 4, 4, 2147483647], dtype=np.int32)), array([         6,          5,        255, 2147483647])

np.bitwise_or(np.array([2, 5, 255]), np.array([4, 4, 4]))
np.bitwise_and(np.array([2,4,255]), np.array([4,4,255]))

np.bitwise_and()
np.bitwise_and([True, True], [False, True])
np.bitwise_xor(np.array([2,5,255]), np.array([3,14,16]))
np.bitwise_repr(np.array())

pd.melt pd.stack pd.pivot pd.reshape, pd.range , df.describe, df.isull(), df.dropna, df.isna, df.area

from math import randians 

import matplotlib.pyplot()

from math import radians
import numpy as np     # installed with matplotlib
import matplotlib.pyplot as plt
def main():
    x = np.arange(0.01, radians(1800), radians(12))
    plt.plot(x, np.sin(x)/x, 'Goldenrod')
    plt.show()
main()
python -m pip install matplotlib

get-pip.py
python -m pip install matplotlib
get install python3-pip

import pandas as pd
import os
a = os.getcwd()
gh =os.rename('bquxjob_8f4cb5e_18336c174da.xlsx', 'New_file_for_dodo_products.csv')
gh.to_csv(  )
ab = os.rename('Catalogue_0086.xlsx', 'New_product_creation.csv')
ab.to_csv()
df= pd.read_csv('C:/Users/Admin/Downloads/HistoricalData_1663519979640.csv', index_col='Date', parse_dates=['Date'])
df
a= df[df['2017-01']
df['2020-01'].Volume.mean()
df.columns
df[df['2018-01']]['Volume'].mean()
df['Close/Last'].mean()
df.head()

import xlwings as xw

rng = pd.date_range(start='01/02/2022', end='01/20/2022', freq= 'B')
rng
df.set_index(rng, inplace=True)

a= np.arange(6).reshape(2,3)
for i in np.nditer(a, flags=['external_loop']):
    print(x, end=' ')   

for x in np.nditer(a, flags=['external_loop'], order='F'):
    print(x, end=' ')
it = np.nditer(a, flags=['multi_index'])

a= np.array([1,2,3])
a.reshape(3,1).mean()
a.std()
y = np.zeros(500)
pip install matplotlib
np.empty(y)

import matplotlib.pyplot as plt

a= np.arange(600)
np.nditer(z, op_flags= 'external_loop')
for x in np.nditer(a, flags=['external_loop'], order='F'):
    print(x, end=' ')

it = np.nditer(a, flags=['multi_index'])
while not it.finished:
    print("%d <%s>" % (it[0], it.multi_index), end=' ')
    is_not_finished = it.iternext()
rng= pd.date_range(start='1/1/2017', freq='B',periods= 80)
rng
np.random.randint(1,10, len(rng))

with np.nditer(a, flags=['multi_index'], op_flags=['writeonly']) as it:
    while not it.finished:
        it[0] = it.multi_index[1] - it.multi_index[0]
        is_not_finished = it.iternext()
rng = pd.date_range(start= 1/5/2021, periods=120, freq='B')
df.set_index(rng, inplace=True)
df= pd.read_csv('HistoricalData_1663519979640.csv', index_col=rng  )
df.as_freq('D', method='pad')

df.as.freq('D', method='pad')


n= {'x': [1,2,3,4] , 'y': ['a','b', 'c', 'd']}
n['x'][2]
n.iloc[3]

pip install matplotlib