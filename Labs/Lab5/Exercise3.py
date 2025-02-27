#Exercise 3:
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('Bank-data.csv')
df = df.drop(df.columns[0], axis = 1)
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
logReg = linear_model.LogisticRegression().fit(x_scaled, y)
odds = np.exp(logReg.coef_)
print(f'Odds: {odds}')

data_points = np.array([[1.335, 0, 1, 0, 0, 109],
                        [1.25, 0, 0, 1, 0, 279]])

data_points_scaled = scaler.transform(data_points)
probabilities = logReg.predict_proba(data_points_scaled)
predictions = logReg.predict(data_points_scaled)
for i in range(len(data_points)):
    if predictions[i] == 'no':
        print(f'The client will not subscribe with no/yes probabilities of {probabilities[i]}')
    else:
        print(f'The client will subscribe with no/yes probabilities of {probabilities[i]}')