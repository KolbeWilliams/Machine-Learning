#Exercise 1:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('wdbc.data.csv', header = None)
x = np.array(df.iloc[:, 2:])
y = np.array(df.iloc[:, 1])

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

Y = np.array(df.iloc[:, 1])
logReg = LogisticRegression().fit(principalComponents, Y)
intercept = logReg.intercept_
coefficents = logReg.coef_
x1 = np.array(principalComponents[:, 0])
x2 = -((intercept + coefficents[0][0] * x1) / coefficents[0][1])
model = intercept + coefficents[0][0] * x1 + coefficents[0][1] * x2
pred = logReg.predict(data_point_pca)
if pred[0] == 'B':
    prediction = 'Benign'
else:
    prediction = 'malignant'
print(f'The data point is predicted to be:', prediction)

label_list = ['M', 'B', 'New Data Point', 'Decison Boundry (Logistic Regression)']
Classes = ['M', 'B']
colors = ['r', 'g']
for classes, color in zip(Classes, colors):
    indicesToKeep = principalDf['Classes'] == classes
    plt.scatter(principalDf.loc[indicesToKeep, 'PC1'], principalDf.loc[indicesToKeep, 'PC2'], c = color)
plt.scatter(data_point_pca[0][0], data_point_pca[0][1], c = 'b', marker = '+', s = 200)
plt.plot(x1, x2, c = 'y')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'PCA = 2, Classifiction: {prediction}')
plt.legend(label_list)
plt.show()