#Exercise 1:
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

df = pd.read_csv('speedLimits.csv')
x = np.array(df.iloc[:, 0])
y = np.array(df.iloc[:, 1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 0)

kernel = ['linear', 'poly', 'rbf', 'sigmoid']
accuracies = []
for k in kernel:
    clf = SVC(kernel = k).fit(x_train.reshape(-1, 1), y_train)
    pred = clf.predict(x_test.reshape(-1, 1))
    accuracy = accuracy_score(y_test, pred)
    print(f'Accuracy score for {k}: {accuracy}')
    accuracies.append(accuracy)

best_accuracy_index = np.argmax(accuracies)
best_kernel = kernel[best_accuracy_index]
print('The optimal kernel is:', best_kernel)

for i in range(len(y)):
    if y[i] == 'NT':
        plt.scatter(x[i], y[i], c = 'g')
    else:
        plt.scatter(x[i], y[i], c = 'r')
plt.title('Speed vs Ticket')
plt.xlabel('Speed')
plt.ylabel('Ticket?')
plt.show()