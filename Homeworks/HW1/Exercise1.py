#Exercise 1:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

names = ['class', 'Alcohol','Malic Acid','Ash','Acadlinity','Magnisium','Total Phenols','Flavanoids', 
         'NonFlavanoid Phenols', 'Proanthocyanins', 'Color Intensity', 'Hue', 'OD280/OD315', 'Proline']

df = pd.read_csv('wine.data.csv', names=names)
X = np.array(df.iloc[:, 1:])
y = np.array(df['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

scores = np.array([])
num = [i for i in range (1,11)]
for i in num:
    knn = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
    pred = knn.predict(X_test)
    scores = np.append(scores, accuracy_score(y_test, pred))
    print(f'Model accuracy score using K = {i}: ', scores[i - 1])

plt.plot(num, scores)
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('Accuracy for K = 1-10')
plt.grid()
plt.show()
