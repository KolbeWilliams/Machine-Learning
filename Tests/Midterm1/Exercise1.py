#Exercise 1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('wdbc.data.csv')
x = np.array(df.iloc[:, 2:])
y = np.array(df.iloc[:, 1])
for i in range(len(y)):
    if y[i] == 'M':
        y[i] = 0
    else: 
        y[i] = 1

scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
pca = PCA(n_components=2).fit(x_scaled)
principalComponents = pca.transform(x_scaled)
variance_ratio = pca.explained_variance_ratio_
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
principalDf['Classes'] = y
print('Variance ratio: ', variance_ratio)

data_point = np.array([7.76, 24.54, 47.92, 181, 0.05263, 0.04362, 0, 0, 0.1587, 0.05884, 0.3857, 1.428, 2.548, 19.15, 0.007189, 0.00466, 
                       0, 0, 0.02676, 0.002783, 9.456, 30.37, 59.16, 268.6, 0.08996, 0.06444, 0, 0, 0.2871, 0.07039])
data_point_scaled = scaler.transform(data_point.reshape(1,-1))
data_point_pca = pca.transform(data_point_scaled)

plt.scatter(principalDf['PC1'], principalDf['PC2'], c = y)
plt.scatter(data_point_pca[0][0], data_point_pca[0][1], c = 'r', marker = '+', label = 'New Data Point')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA = 2, Variance:{variance_ratio}')
plt.legend()
plt.show()