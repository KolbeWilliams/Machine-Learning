#Exercise 3:
import pandas as pd
import numpy as np

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
for i in range(len(correlations)):
    print(f'The correlation coefficient between {df.columns[0]} and {df.columns[i + 1]} is: {correlations[i]:.4f}')

#Perform multiple linear regression
data_points = np.array([[32.1, 37.5, 128.95],[36.9, 35.37, 130.03]])
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

#Make predictions
predictions = []
for i in data_points:
    pred = intercept
    for j in range(len(coefficients)):
        pred += coefficients[j] * i[j]
    predictions.append(pred)
print(f'\nThe strength prediction for the first data point is: {predictions[0]:.4f}')
print(f'The strength prediction for the second data point is: {predictions[1]:.4f}')