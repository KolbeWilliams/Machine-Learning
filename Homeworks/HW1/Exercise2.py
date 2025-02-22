#Exercise 2:
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion',
        'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']

df = pd.read_csv('breast-cancer-wisconsin.data.csv', names = names)
df = df.replace('?', np.nan)
df = df.dropna()
print('Shape: ', df.shape)

X = np.array(df.iloc[:, 1:11])
Y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

knn = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train)
pred = knn.predict(X_test)
print('Model accuracy score: ', accuracy_score(y_test, pred))

print('Index\tPredicted\tActual')
for i in range(len(pred)):
    if pred[i] != y_test[i]:
        print(i, '\t', pred[i], '\t\t', y_test[i])

conf_matrix = confusion_matrix(y_test, pred)
print(f'\nConfusion Matrix: \n{conf_matrix}')

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

