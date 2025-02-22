#Exercise 2:
from turtle import st
import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.array([[1, 5], [3, 2], [8, 4], [7, 14]])
x1 = data[:,0]
x2 = data[:,1]

#Without Using Built-in Functions
print('Without using built-in function:')
x1_mean = 0
x2_mean = 0
for i in range(len(x1)):
    x1_mean += x1[i]
    x2_mean += x2[i]
x1_mean /= len(x1)
x2_mean /= len(x2)
print(f'Mean of X1 (before scaling): {x1_mean}')
print(f'Mean of X2 (before scaling): {x2_mean}')

std_dev_x1 = 0
std_dev_x2 = 0
for i in range(len(x1)):
    std_dev_x1 += (x1[i] - x1_mean)**2
    std_dev_x2 += (x2[i] - x2_mean)**2
std_dev_x1 = (std_dev_x1 / (len(x1)))**0.5
std_dev_x2 = (std_dev_x2 / (len(x2)))**0.5
print(f'Standard deviation of X1(before scaling): {std_dev_x1}')
print(f'Standard deviation of X2(before scaling): {std_dev_x2}')

x1_standardized = (x1 - x1_mean) / std_dev_x1
x2_standardized = (x2 - x2_mean) / std_dev_x2
print('X1 standardized: ', x1_standardized)
print('X2 standardized: ', x2_standardized)
standardized_data = np.array([])
for i in range(len(x1_standardized)):
    standardized_data = np.append(standardized_data, [x1_standardized[i],x2_standardized[i]])
standardized_data = standardized_data.reshape(4,2)
print('Standardized data:\n', standardized_data)

x1_reversed = x1_standardized * std_dev_x1 + x1_mean
x2_reversed = x2_standardized * std_dev_x2 + x2_mean
data_reversed = np.array([])
for i in range(len(x1_reversed)):
    data_reversed = np.append(data_reversed, [x1_reversed[i],x2_reversed[i]])
data_reversed = data_reversed.reshape(4,2)
print('Reversed data:\n', data_reversed)

#Using Built-in Functions
print('\nUsing built-in functions:')
scaler = StandardScaler()
z_scaledData = scaler.fit_transform(data)
print(f'Data (after scaling):\n{z_scaledData}')
reverseScale = scaler.inverse_transform(z_scaledData)
print(f'Revert scaled data:\n{reverseScale}')