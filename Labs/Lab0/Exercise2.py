#Exercise 2:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('auto-mpg.csv')
number_rows = df.shape[0]
number_cols = df.shape[1]
print('Number of rows:', number_rows)
print('Number of columns:', number_cols)
df = df.replace('?', np.nan)
df = df.dropna()
number_rows = df.shape[0]
number_cols = df.shape[1]
print('\nNumber of rows after droping nan:', number_rows)
print('Number of columns after dropping nan:', number_cols)

x = np.array(df['weight'])
y = np.array(df['mpg'])
w1 = ((np.mean(x * y)) - (np.mean(x) * np.mean(y))) / ((np.mean(x ** 2)) - (np.mean(x) ** 2))
w0 = np.mean(y) - (w1 * np.mean(x))
model = w1 * x + w0
test_points = np.array([i for i in range(1500, 5001, 500)])
predictions = np.array([i for i in w1 * test_points + w0])
sst = np.sum((y - np.mean(y)) ** 2)
sse = np.sum((model - y) ** 2)
r = (1 - (sse / sst)) ** (1/2)
if w1 < 0:
    r *= -1

plt.scatter(x, y, label = 'training data points', c = 'b')
plt.scatter(test_points, predictions, label = 'test data points', c = 'y')
plt.plot(x, model, label = 'OLS', c = 'r')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.title(f'Linear Regression with OLS; slope: {w1:.2f}, y-intercept: {w0:.2f}, r: {r:.2f}')
plt.text(3900, 27.5, 'Linear Regression line')
plt.grid(True)
plt.legend()
plt.show()