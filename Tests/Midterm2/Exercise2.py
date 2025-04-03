#Exercise 2:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC

df = pd.read_csv('Skin_NonSkin.csv')
x = np.array(df.iloc[::100, :3])
y = np.array(df.iloc[::100, -1])

pca = PCA(n_components = 2).fit(x)
principal_components = pca.transform(x)
principalDf = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])
principalDf['Classes'] = y

x_train, x_test, y_train, y_test = train_test_split(principal_components, y, test_size = 0.2, random_state = 0)
clf = SVC(kernel = 'linear').fit(x_train, y_train)
pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)

print(f'Accuracy of the model: {accuracy}')
print('\nClassification Report:\n', classification_report(y_test, pred))
print(f'\nConfusion Matrix:\n{cm}\n')

data_points = np.array([[73, 80, 122],
               [251, 250, 245]])

data_points_pca = pca.transform(data_points)

predictions = clf.predict(data_points_pca)
for i in range(len(predictions)):
    print(f'Data point {i + 1} is predicted to be {predictions[i]}')

w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-250, 250)
yy = a *xx - (clf.intercept_[0] / w[1])
b = clf.support_vectors_[0]
b2 = clf.support_vectors_[1]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

classes = principalDf['Classes'].unique()
legend = ['1', '2']
colors = ['r', 'g']

for Classes, color in zip(classes,colors):
    indicesToKeep = y == Classes 
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1'] ,principalDf.loc[indicesToKeep, 'principal component 2'], c = color)
plt.plot(xx, yy, c = 'black')
plt.plot(xx, yy_down, 'b--')
plt.plot(xx, yy_up, 'b--')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Support Vector Machine')
plt.xlim(min(principalDf['principal component 1']), max(principalDf['principal component 1']))
plt.ylim(min(principalDf['principal component 2']), max(principalDf['principal component 2']))
plt.legend(legend)
plt.show()