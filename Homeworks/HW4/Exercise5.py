#Exercise 5:
import numpy as np
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('materialsOutliers.csv')
x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

outliers = [False for i in range(x.shape[0])]
for i in range(x.shape[1]):
    ransac = linear_model.RANSACRegressor(residual_threshold=15, stop_probability=1.00)
    ransac.fit(y.reshape(-1, 1), x[:, i].reshape(-1, 1))
    inlier_mask = ransac.inlier_mask_
    outlier_mask = np.logical_not(inlier_mask)
    for j in range(len(outlier_mask)):
        if outlier_mask[j] == True:
            outliers[j] = True

indicies = []
for i in range(len(outliers)):
    if outliers[i] == True:
        indicies.append(i)

df = df.drop(indicies)
x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

reg = linear_model.LinearRegression()
reg.fit(x,y)
for i in range(len(reg.coef_)):
    print(f'Coefficient {i + 1} is: {reg.coef_[i]:.4f}')
print(f'The y-intercept is: {reg.intercept_:.4f}')
