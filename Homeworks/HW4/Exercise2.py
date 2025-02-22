#Exercise 2:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

df = pd.read_csv('avgHigh_jan_1895-2018.csv').drop(['Anomaly'], axis = 1)
x = np.array(df['Date']).reshape(-1,1)
y = np.array(df['Value'])

test_size = float(input('Enter test size: '))
num_rows = round(len(x)*test_size)
test_start = len(x) - num_rows
x_train = x[:test_start]
x_test = x[test_start:]
y_train = y[:test_start]
y_test = y[test_start:]

reg = LinearRegression().fit(x_train, y_train)
slope, intercept, r, p, std_error = stats.linregress(x_train.ravel(), y_train.ravel())
model = slope*x_train + intercept

print(f'\nSlope: {slope}, y-intercept: {intercept}, correlation: {r}, p-value: {p}, standard eroor: {std_error}\n')
pred = reg.predict(x_test)
for i in range(len(pred)):
    print(f'Actual: {y_test[i]}, Predicted: {pred[i]}')
rmse = ((np.sum((pred - y_test)**2)) / len(pred))**0.5
print(f'\nRoot Mean Square Error: {rmse}')

plt.scatter(x_train, y_train, c = 'b', label = 'Train')
plt.scatter(x_test, y_test, c = 'g', label = 'Test')
plt.plot(x_train, model, c = 'r', label = 'Model')
plt.title(f'Slope: {slope:.2f}, Inercept: {intercept:.2f}, Test size: {test_size:.2f} ({num_rows}/{len(x)}), RMSE: {rmse:.2f}')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()