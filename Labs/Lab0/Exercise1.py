#Exercise 1:
import pandas as pd
import numpy as np

df = pd.read_csv('auto-mpg.csv')
number_rows = df.shape[0]
number_cols = df.shape[1]
print('Number of rows:', number_rows)
print('Number of columns:', number_cols)
df = df.replace('?', np.nan)
df = df.dropna()
number_rows = df.shape[0]
number_cols = df.shape[1]
print('\nNumber of rows after droping nan:', number_rows)
print('Number of columns after dropping nan:', number_cols)