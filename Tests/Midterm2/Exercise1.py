#Exercise 1:
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def sigmoid(x, coef, intercept):
    model = intercept
    for i in range(len(coef)):
        model += coef[i] * x[i]
    prob = 1 / (1 + np.exp(-model))
    return 1 - prob, prob

df = pd.read_csv('universityAdmissions.csv')
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

for i in range(len(y)):
    if y[i] >= 0.75:
        y[i] = 1
    else:
        y[i] = 0

scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)

logReg = LogisticRegression().fit(x_scaled, y)

data_points = [[338, 119, 9.73, 1],
               [305, 100, 7, 0],
               [327, 112, 9.03, 0]]

data_points_scaled = scaler.transform(data_points)
odds = np.exp(logReg.coef_)
print('Odds: ', odds[0])

proba = logReg.predict_proba(data_points_scaled)
pred = logReg.predict(data_points_scaled)
print('\nProbabilities and predictions using built in function:')
for i in range(len(data_points)):
    print(f'Probabilites for data point {i + 1}: {proba[i]}')
    print(f'Predicted class for data point {i + 1}: {pred[i]}\n')

print('\nProbabilities and predictions without using built in function:')
for i in range(len(data_points)):
    probability_no, probability_yes = sigmoid(data_points_scaled[i], logReg.coef_[0], logReg.intercept_[0])
    pred = 0 if probability_yes < 0.5 else 1
    print(f'Probabilites for data point {i + 1}: [{probability_no}, {probability_yes}]')
    print(f'Predicted class for data point {i + 1}: {pred}\n')