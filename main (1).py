import pandas as pd
import numpy as np

s = pd.Series(['a', 'b', 'c'], index=[0, 2, 4])
s_reindexed = s.reindex(range(6), fill_value='missing')
s_reindexed_method = s.reindex(range(6)).ffill()

df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']}, index=[0, 2, 4])
df_reindexed = df.reindex(range(6))
df_reindexed_shape = df_reindexed.shape
df_reindexed_nulls = df_reindexed.isnull().sum()

ts = pd.Series([1, 2, np.nan, 4], index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-05']))
ts_filled = ts.reindex(pd.date_range('2023-01-01', '2023-01-05')).interpolate()

def reindex_with_method(obj, new_idx, method='ffill'):
    return obj.reindex(new_idx).fillna(method=method)

sort_ex = pd.Series([3, 1, 2], index=['b', 'a', 'c'])
sort_indexed = sort_ex.sort_index()
sort_valued = sort_ex.sort_values()

u = pd.Series([1.0, 2.0, 3.0], index=['a', 'b', 'c'])
u_applied = np.exp(u)

u2 = pd.Series([4.0, 5.0, 6.0], index=['b', 'c', 'd'])
u_combined = u + u2

u_conditional = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
u_where = np.where(u_conditional > 15, 'High', 'Low')

def apply_clean_ufunc(s):
    return np.log(s.dropna())

arr_vs_series = (np.array([1, 2, 3]) + 1, pd.Series([1, 2, 3], index=['a', 'b', 'c']) + 1)

df1 = pd.DataFrame(np.arange(9).reshape((3, 3)), columns=list('ABC'))
broadcasted = df1 - df1.mean()
safe_add = df1.add(100, fill_value=0)
agg_results = df1.agg(['mean', 'sum', 'std'])
normalized_rows = df1.div(df1.sum(axis=1), axis=0)
normalized_cols = df1.div(df1.sum(axis=0), axis=1)

def zscore(df):
    return (df - df.mean()) / df.std()

zscore_df = zscore(df1)

df_nan = pd.DataFrame([[1, np.nan, 3], [4, 5, np.nan], [np.nan, 8, 9]])
nulls = df_nan.isnull()
not_nulls = df_nan.notnull()
dropped = df_nan.dropna(thresh=2)
filled = df_nan.fillna(0)
ffilled = df_nan.ffill()
bfilled = df_nan.bfill()

ts2 = pd.Series([1, np.nan, np.nan, 4], index=pd.date_range('2023-01-01', periods=4))
interpolated = ts2.interpolate()

def clean_pipeline(df):
    return df.fillna(method='ffill').fillna(0)

midx_df = pd.DataFrame(np.random.randn(6, 2),
                       index=pd.MultiIndex.from_product([['A', 'B'], [1, 2, 3]]),
                       columns=['X', 'Y'])
midx_stacked = midx_df.stack()
midx_unstacked = midx_stacked.unstack()
xs_val = midx_df.xs(2, level=1)
grouped_sum = midx_df.groupby(level=0).sum()
renamed = midx_df.rename_axis(['Group', 'Index'])
flattened = midx_df.reset_index()

s1 = pd.Series([1, 2], index=['a', 'b'])
s2 = pd.Series([3, 4], index=['c', 'd'])
concat_axis0 = pd.concat([s1, s2])
concat_axis1 = pd.concat([s1, s2], axis=1)
concat_keys = pd.concat([s1, s2], keys=['first', 'second'], names=['group'])

df7 = pd.DataFrame([[1, 2]], columns=['a', 'b'])
df8 = pd.DataFrame([[3, 4]], columns=['a', 'b'])
appended = pd.concat([df7, df8], ignore_index=True)

df9 = pd.DataFrame({'x': [1, 2]})
df10 = pd.DataFrame({'y': [3, 4]})
concat_mixed = pd.concat([df9, df10], axis=1)

def combine_and_clean(dfs):
    return pd.concat(dfs, ignore_index=True)

left = pd.DataFrame({'key': ['K0', 'K1'], 'A': ['A0', 'A1']})
right = pd.DataFrame({'key': ['K0', 'K1'], 'B': ['B0', 'B1']})
merged_one = pd.merge(left, right, on='key')

left_m = pd.DataFrame({'key': ['K0', 'K1', 'K2'], 'A': ['A0', 'A1', 'A2']})
right_m = pd.DataFrame({'key': ['K0', 'K1'], 'B': ['B0', 'B1']})
merged_many = pd.merge(left_m, right_m, on='key', how='left')
merged_disambig = pd.merge(left_m, right_m, left_on='key', right_on='key', suffixes=('_l', '_r'))
join_indexed = left.set_index('key').join(right.set_index('key'), how='outer')
multi_merge = pd.merge(left, right, on='key', indicator=True)

states = pd.DataFrame({
    'State': ['CA', 'TX', 'NY'],
    'Population': [39500000, 29000000, 19500000]
})
area = pd.DataFrame({
    'State': ['CA', 'TX', 'NY'],
    'Area': [163696, 268596, 54555]
})
abbr = pd.DataFrame({
    'State': ['CA', 'TX', 'NY'],
    'Abbreviation': ['CA', 'TX', 'NY']
})

merged_states = states.merge(area, on='State').merge(abbr, on='State')
merged_states['Density'] = merged_states['Population'] / merged_states['Area']
merged_states['SizeClass'] = pd.cut(merged_states['Population'],
                                    bins=[0, 20000000, 30000000, np.inf],
                                    labels=['Small', 'Medium', 'Large'])

print(merged_states)

