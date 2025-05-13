#Exercise 2:
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('usedcars.csv')
df['model'] = df['model'].map({'SEL': 0, 'SE': 1, 'SES': 2})
df['color'] = df['color'].map({'Black': 0, 'Blue': 1, 'Gold': 2, 'Gray': 3, 'Green': 4, 'Red': 5, 'Silver': 6, 'White': 7, 'Yellow': 8})
df['transmission'] = df['transmission'].map({'AUTO': 0, 'MANUAL': 1})

y = np.array(df['price'])
x = np.array(df.drop('price', axis = 1))

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.10, random_state = 0)

estimators = np.arange(100, 1001, 100)
models = []
predictions = []
error = []
for i in estimators:
    rf = RandomForestRegressor(n_estimators = i).fit(x_train, y_train)
    models.append(rf)
    pred = rf.predict(x_test)
    predictions.append(pred)
    error.append(root_mean_squared_error(y_test, pred))

for i in range(len(models)):
    print(f'Root Mean Square Error ({estimators[i]} estimators): {error[i]:.2f}')

best_estimators = estimators[np.argmin(error)]
print(f'\nMinimum RMSE: {error[np.argmin(error)]} at index {np.argmin(error)}')

rf = RandomForestRegressor(n_estimators = best_estimators, random_state = 0).fit(x_train, y_train)
pred = rf.predict(x_test)

print()
for i in range(len(pred)):
    print(f'Actual: ${y_test[i]} Predicted: ${pred[i]:.2f}')

data_point = np.array([2017, 'SE', 113067, 'Blue', 'AUTO'])
for i in range(len(data_point)):
    if data_point[i] in ['SEL', 'Black', 'AUTO']:
        data_point[i] = 0
    elif data_point[i] in ['SE', 'Blue', 'MANUAL']:
        data_point[i] = 1
    elif data_point[i] in ['SES', 'Gold']:
        data_point[i] = 2
    elif data_point[i] in ['Gray']:
        data_point[i] = 3 
    elif data_point[i] in ['Green']: 
        data_point[i] = 4
    elif data_point[i] in ['Red']:
       data_point[i] = 5 
    elif data_point[i] in ['Silver']: 
       data_point[i] = 6
    elif data_point[i] in ['White']:
       data_point[i] = 7
    elif data_point[i] in ['Yellow']:
       data_point[i] = 8

data_point = data_point.reshape(1, -1)
data_point_scaled = scaler.transform(data_point)
feature_importances = rf.feature_importances_
print(f'\nPredicted Value of single data point: ${rf.predict(data_point_scaled)}')
print(f'Importanant features: {feature_importances}')
print(f'Importanant Features: ', end = '')
features = []
for col in df.columns:
    if col != 'price':
        print(f'{col} ', end = '')
        features.append(col)
print()