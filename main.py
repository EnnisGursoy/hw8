import pandas as pd
import numpy as np

scores = pd.Series([88, 92, 79, 93, 85], index=['Alice', 'Bob', 'Charlie', 'David', 'Eva'])
print(scores.idxmax(), scores.max())
print(scores[scores > scores.mean()])

df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [20, 21, 19],
    'Passed': [True, True, False]
})
print(df.dtypes)
print(df.describe(include='all'))

csv = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': ['a', 'b', 'c', 'd', 'e'],
    'C': pd.date_range('2024-01-01', periods=5)
})
print(csv.head())
print(csv.tail())
csv.info()
print(csv.describe(include='all'))

dates = pd.date_range('2023-01-01', periods=10)
ts = pd.Series(np.arange(10), index=dates)
print(ts['2023-01-03':'2023-01-06'])

s1 = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
s2 = pd.Series([4, 5, 6], index=['b', 'c', 'd'])
print(s1 + s2)

df2 = pd.DataFrame({
    'n': ['Anna', 'Ben', 'Cathy'],
    's': [82, 91, 77],
    'g': ['B', 'A', 'C']
}, index=['r1', 'r2', 'r3'])
print(df2.loc['r1'])
print(df2.iloc[0])
print(df2[df2['s'] > 80])
print(df2['s'] > 80)

df3 = df2.copy()
df3.loc['r1', 's'] = 85
print(df3)

print(df2.loc['r1':'r2', ['n', 's']])
print(df2.iloc[0:2, 0:2])

idx = pd.Index(['x', 'y', 'z'])
print(idx.append(pd.Index(['w'])))

midx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)])
ser = pd.Series([10, 20, 30], index=midx)
print(ser)
print(ser['A'])

midx2 = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1), ('B', 2)], names=['L', 'N'])
df4 = pd.DataFrame(np.random.randn(4, 2), index=midx2, columns=['X', 'Y'])
print(df4)
print(df4.swaplevel())
print(df4.sort_index())

df5 = df4.reset_index()
df6 = df5.set_index(['L', 'N'], drop=True)
print(df5)
print(df6)

a = pd.DataFrame({'val': [1, 2]}, index=['x', 'y'])
b = pd.DataFrame({'val': [3, 4]}, index=['y', 'z'])
print(a + b)
