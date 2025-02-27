#Exercise 2:
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('Student-Pass-Fail.csv')
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
logReg = linear_model.LogisticRegression().fit(x_scaled, y)
odds = np.exp(logReg.coef_)
print('Odds: ', odds)

data_points = np.array([[7, 28],
                        [10, 34],
                        [2, 39]])
data_points_scaled = scaler.transform(data_points)

predictions = []
probabilities = []
for point in data_points_scaled:
    pred = logReg.predict(point.reshape(1, -1))
    predictions.append(pred)
    probability = logReg.predict_proba(point.reshape(1, -1))
    probabilities.append(probability)

for i in range(len(data_points_scaled)):
    if predictions[i] == 0:
        print(f'Student {i+1} will fail with fail/pass probabilities of {probabilities[i][0]}')
    else:
        print(f'Student {i+1} will pass with fail/pass probabilities of {probabilities[i][0]}')

test_size = float(input('\nEnter the test size: '))
test_length = round(len(y) * test_size)
x_train = np.array(df.iloc[test_length:, :-1])
x_test = np.array(df.iloc[:test_length, :-1])
y_train = np.array(df.iloc[test_length:, -1])
y_test = np.array(df.iloc[:test_length, -1])

x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
logReg = linear_model.LogisticRegression().fit(x_train_scaled, y_train)
predictions = logReg.predict(x_test_scaled)

classes = np.unique(y_test)
num_classes = len(classes)
correct, incorrect = 0, 0
cm = np.zeros((num_classes, num_classes), dtype = int)
for i in range(len(predictions)):
    if predictions[i] == y_test[i]:
        for x in range(len(classes)):
            if y_test[i] == classes[x]:
                index = x
        cm[index, index] += 1
        correct += 1
    else:
        for x in range(len(classes)):
            if y_test[i] == classes[x]:
                actual_index = x
            if predictions[i] == classes[x]:
                predicted_index = x
        cm[actual_index, predicted_index] += 1
        incorrect += 1
accuracy = correct / len(y_test)
print('\nWithout using built-in functions:')
print(f'Accuracy: {accuracy}')
print('Confuion Matrix:\n', cm)

print('\nUsing built-in functions:')
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: ', accuracy)
cm = confusion_matrix(y_test, predictions)
print('Confuion Matrix:\n', cm)
