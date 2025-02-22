#Exercise 2:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv('hsbdemo.csv')
df['gender'] = df['gender'].replace('male', 0)
df['gender'] = df['gender'].replace('female', 1)
df['ses'] = df['ses'].replace('low', 0)
df['ses'] = df['ses'].replace('middle', 1)
df['ses'] = df['ses'].replace('high', 2)
df['schtyp'] = df['schtyp'].replace('public', 0)
df['schtyp'] = df['schtyp'].replace('private', 1)
df['prog'] = df['prog'].replace('vocation', 0)
df['prog'] = df['prog'].replace('general', 1)
df['prog'] = df['prog'].replace('academic', 2)
df['honors'] = df['honors'].replace('not enrolled', 0)
df['honors'] = df['honors'].replace('enrolled', 1)
cols = [column for column in df.columns if column not in ['id', 'prog', 'cid']]
x = np.array(df[cols])
y = np.array(df['prog'])
x = StandardScaler().fit_transform(x)
pca = PCA(n_components = 10)
principalComponents = pca.fit_transform(x)
variance_ratio = pca.explained_variance_ratio_
print(f'Variance ratio:\n{variance_ratio}')
plt.plot(range(1,11), np.cumsum(variance_ratio))
plt.xlabel('Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.title('PC = 1-10')
plt.show()