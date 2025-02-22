#Exercise 1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('avgHigh_jan_1895-2018.csv')
x = np.array(df.drop(['Value', 'Anomaly'], axis = 1))
y = np.array(df['Value'])

reg = LinearRegression().fit(x, y)
data_points = np.array([[201901, 202301, 202401]])
pred = []
for i in data_points:
    pred.append(reg.predict(i.reshape(-1,1)))

model = reg.coef_*x + reg.intercept_

plt.scatter(x,y, c = 'b', label = 'Data Points')
plt.scatter(data_points, pred, c = 'g', label = 'Predicted')
plt.plot(x, model, c = 'r', label = 'Model')
plt.title(f'January Average High Temperatures. Slope: {reg.coef_[0]:.2f}, Intercept: {reg.intercept_:.2f}')
plt.xlabel('Year')
plt.ylabel('Temperature')
plt.legend()
plt.show()