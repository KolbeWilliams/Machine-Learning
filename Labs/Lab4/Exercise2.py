#Exercise 2:
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

california_housing = fetch_california_housing(as_frame=True)

x = np.array(california_housing.frame.iloc[::10, :2])
x1 = np.array(california_housing.frame.iloc[::10, 0])
x2 = np.array(california_housing.frame.iloc[::10, 1])
y = np.array(california_housing.frame.iloc[::10, -1])

reg = LinearRegression()
reg.fit(x, y)

print('Coefficients:')
for i in range(len(reg.coef_)):
    print(f'{reg.coef_[i]:.4f}', end = ' ')
print(f'\ny-intercept: {reg.intercept_:.4f}')

X1, X2 = np.meshgrid(x1, x2)
Z = reg.intercept_ + reg.coef_[0]*X1 + reg.coef_[1]*X2

#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'blue')
#3D scattet plot (data points)
ax.scatter3D(x1, x2, y, c=y, cmap='Greens')
ax.set_title('3D Graph')
ax.set_xlabel('Medinc')
ax.set_ylabel('HouseAge')
ax.set_zlabel('Median House Value')
plt.show()

