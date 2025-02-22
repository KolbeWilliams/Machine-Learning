#Exercise 2:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

df = pd.read_csv('iris.data.csv', names=names)
x_sepal = np.array(df.iloc[:, 0])
y_sepal = np.array(df.iloc[:, 1])
x_petal = np.array(df.iloc[:, 2])
y_petal = np.array(df.iloc[:, 3])
label = df['class'].copy()

fig, ax = plt.subplots(nrows = 1, ncols = 2)
for i in range(len(label)):
    if label[i] == 'Iris-setosa':
        label[i] = 1
    if label[i] == 'Iris-versicolor':
        label[i] = 2
    if label[i] == 'Iris-virginica':
        label[i] = 3
    # if label[i] == 'Iris-setosa':
    #     ax[0].scatter(x_sepal[i], y_sepal[i], c = 'purple')
    #     ax[1].scatter(x_petal[i], y_petal[i], c = 'purple')
    # if label[i] == 'Iris-versicolor':
    #     ax[0].scatter(x_sepal[i], y_sepal[i], c = 'b')
    #     ax[1].scatter(x_petal[i], y_petal[i], c = 'b')
    # if label[i] == 'Iris-virginica':
    #     ax[0].scatter(x_sepal[i], y_sepal[i], c = 'y')
    #     ax[1].scatter(x_petal[i], y_petal[i], c = 'y')
ax[0].scatter(x_sepal, y_sepal, c = label)
ax[1].scatter(x_petal, y_petal, c = label)
ax[0].set_xlabel('Sepla Length')
ax[0].set_ylabel('Sepal Width')
ax[0].set_title('Sepal Features')
ax[1].set_xlabel('Petal Length')
ax[1].set_ylabel('Petal Width')
ax[1].set_title('Petal Features')
plt.show()

