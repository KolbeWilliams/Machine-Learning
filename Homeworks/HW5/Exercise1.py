#Exercise 1:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

df = pd.read_csv('vehicles.csv').drop('make', axis = 1)
x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

x_scaled = StandardScaler().fit_transform(x)
reg = LinearRegression().fit(x_scaled, y)
print(f'The weighted coefficients are: \n{reg.coef_}')
coefficients = np.abs(reg.coef_)

weight_coefficients = []
for i in range(5):
    most_weight = np.argmax(coefficients)
    weight_coefficients.append(df.columns[most_weight + 1])
    coefficients[most_weight] = -np.inf

print(f'\nThe 5 coefficients with the most weight are: {weight_coefficients}')