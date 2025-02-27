#Exercise 1:
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Student-Pass-Fail.csv')
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
logReg = linear_model.LogisticRegression().fit(x_scaled, y)
odds = np.exp(logReg.coef_)
print('Odds: ', odds)

data_points = np.array([[7, 28],
                        [10, 34],
                        [2, 39]])
data_points_scaled = scaler.transform(data_points)

predictions = []
probabilities = []
for point in data_points_scaled:
    pred = logReg.predict(point.reshape(1, -1))
    predictions.append(pred)
    probability = logReg.predict_proba(point.reshape(1, -1))
    probabilities.append(probability)

for i in range(len(data_points_scaled)):
    if predictions[i] == 0:
        print(f'Student {i+1} will fail with fail/pass probabilities of {probabilities[i][0]}')
    else:
        print(f'Student {i+1} will pass with fail/pass probabilities of {probabilities[i][0]}')