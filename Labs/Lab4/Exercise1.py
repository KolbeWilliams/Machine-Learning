#Exercise 1:
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

california_housing = fetch_california_housing(as_frame=True)

x = np.array(california_housing.frame.iloc[::10, :-1])
y = np.array(california_housing.frame.iloc[::10, -1])

reg = LinearRegression()
reg.fit(x, y)

print('Coefficients:')
for i in range(len(reg.coef_)):
    print(f'{reg.coef_[i]:.4f}', end = ' ')
print(f'\ny-intercept: {reg.intercept_:.4f}')

data_point = np.array([[8.3153, 41.0, 6.894423, 1.053714, 323.0, 2.533576, 37.88, -122.23]])
pred = reg.predict(data_point)
print(f'Predicted meadian house value: {pred[0]}')