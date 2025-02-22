#Exercise 3:
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

california_housing = fetch_california_housing(as_frame=True)

columns = california_housing.frame.columns

x = np.array(california_housing.frame.iloc[::10, :-1])
y = np.array(california_housing.frame.iloc[::10, -1])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
reg = LinearRegression()
reg.fit(x_scaled, y)

print('Coefficients:')
for i in range(len(reg.coef_)):
    print(f'{reg.coef_[i]:.4f}', end = ' ')
print(f'\ny-intercept: {reg.intercept_:.4f}')

coefficients = []
for i in range(len(reg.coef_)):
    coefficients.append(np.abs(reg.coef_[i]))

max_index = np.argmax(coefficients)
feature = columns[max_index]

print('The coefficient that carries the most weight is:', feature)
print(f'The feature {feature} has a coefficent calue of: {reg.coef_[max_index]}')

