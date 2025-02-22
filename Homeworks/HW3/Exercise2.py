#Exercise 2:
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('iris.data.csv', names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'classes'])
x = np.array(df.iloc[:, :-1])
y = np.array(df['classes'])

start = 0
test_range = int(len(x) / 5)
end = test_range
accuracies = []
for i in range(5):
    x_train = np.append(x[0:start, :], x[end:, :], axis = 0)
    x_test = x[start:end, :]
    y_train = np.append(y[0:start], y[end:], axis = 0)
    y_test = y[start:end]

    knn = KNeighborsClassifier(n_neighbors = 9)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, pred)
    accuracies.append(accuracy)
    print(f'The accuracy for test dataset {i + 1} is {accuracy:.2f}')
    start = end
    end += test_range
print(f'The average accuracy of the test datasets is: {np.mean(accuracies):.2f}')