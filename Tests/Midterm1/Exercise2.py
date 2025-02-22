#Exercise 2:
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('universityAdmissions.csv')
x = np.array(df.iloc[:, :3])
y = np.array(df.iloc[:, 4])

data_points = [[338, 119, 9.73, 1],
               [305, 100, 7, 0],
               [327, 112, 9.03, 0]]

reg = LinearRegression().fit(x, y)
coefficients = reg.coef_
intercept = reg.intercept_
print(f'The coefficents are: {coefficients}')
print(f'The y-intercept is: {intercept}')

predictions = []
for point in data_points:
    pred = intercept
    for i in range(len(coefficients)):
        pred += coefficients[i] * point[i]
    predictions.append(pred)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        print(f'Data point {i + 1} is accepted with an Admitted score of {predictions[i]}')
    else:
        print(f'Data point {i + 1} is rejected with an Admitted score of {predictions[i]}')