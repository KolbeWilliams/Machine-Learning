#Exercise 4:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Read dataset
df = pd.read_csv('materials.csv')
x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

#Calculate means
means = []
for i in range(x.shape[1]):
    mean = 0
    count = 0
    for j in range(x.shape[0]):
        mean += x[j, i]
        count += 1
    mean /= count
    means.append(mean)
y_mean = 0
for i in y:
    y_mean += i
y_mean /= len(y)

#Calculate correlations
correlations = []
for i in range(x.shape[1]):
    numerator, denominator1, denominator2 = 0, 0, 0
    for j in range(x.shape[0]):
        numerator += (x[j, i] - means[i]) * (y[j] - y_mean)
        denominator1 += (x[j, i] - means[i])**2
        denominator2 += (y[j] - y_mean)**2
    r = numerator / ((denominator1 * denominator2)**0.5)
    correlations.append(r)

#Find highest correlation attributes
correlations_weight = []
for i in range(len(correlations)):
    correlations_weight.append(np.abs(correlations[i]))
feature1 = np.argmax(correlations_weight)
correlations_weight[feature1] = -np.inf
feature2 = np.argmax(correlations_weight)

x1 = x[:, feature2].reshape(-1,1)
x2 = x[:, feature1].reshape(-1,1)
x = np.hstack((x1, x2))

#Perform multiple linear regression
x_matrix = x
ones = np.ones((len(x), 1))
x_matrix = np.hstack((ones, x_matrix))
y_matrix = y
x_transposed = x_matrix.T
x_multiplied = x_transposed @ x_matrix
y_multiplied = x_transposed @ y_matrix
A = np.linalg.inv(x_multiplied)
A = A @ y_multiplied
intercept = A[0]
coefficients = A[1:]

X1, X2 = np.meshgrid(x1, x2)
Z = intercept + coefficients[0]*X1 + coefficients[1]*X2

#3D plot
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_wireframe(X1, X2, Z, color = 'blue')

#3D scattet plot (data points)
ax.scatter3D(x1.ravel(), x2.ravel(), y, c = y, cmap = 'Greens')
ax.set_title('3D Graph')
ax.set_xlabel(f'{df.columns[feature2 + 1]}')
ax.set_ylabel(f'{df.columns[feature1 + 1]}')
ax.set_zlabel('Strength')
plt.show()