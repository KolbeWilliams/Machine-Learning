#Exercise 2:
import numpy as np
import pandas as pd
from math import nan
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC

df = pd.read_csv('breast-cancer-wisconsin-data.csv', header = None)
df = df.replace('?', nan)
df = df.dropna()

x = np.array(df.iloc[:, 1:-1])
y = np.array(df.iloc[:, -1])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

pca = PCA(n_components = 2)
principalComponents = pca.fit_transform(x_scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf['Classes'] = y

x_train, x_test, y_train, y_test = train_test_split(principalComponents, y, random_state = 42)

clf = SVC(kernel = 'linear').fit(x_train, y_train)
pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n {cm}')

w = clf.coef_[0]
a = -w[0] / w[1]
xx = principalDf['principal component 1']
yy = a * xx - (clf.intercept_[0] / w[1])

classes = df.iloc[:, -1].unique()
liClasses = list(classes)
colors = ['blue', 'yellow']
for Classes, color in zip(classes,colors):
    indicesToKeep = y == Classes
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1'], principalDf.loc[indicesToKeep, 'principal component 2'], c = color)
    plt.legend(liClasses, loc = 'lower left')
plt.plot(xx, yy, c = 'g')
plt.title('SVC with PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.ylim(min(principalDf['principal component 2']), max(principalDf['principal component 2']))
plt.show()