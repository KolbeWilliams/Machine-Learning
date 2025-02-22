#Exercise 1:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns

#Convert to numerical data in dataframe
df = pd.read_csv('hsbdemo.csv')
df['gender'] = df['gender'].replace('male', 0)
df['gender'] = df['gender'].replace('female', 1)
df['ses'] = df['ses'].replace('low', 0)
df['ses'] = df['ses'].replace('middle', 1)
df['ses'] = df['ses'].replace('high', 2)
df['schtyp'] = df['schtyp'].replace('public', 0)
df['schtyp'] = df['schtyp'].replace('private', 1)
#df['prog'] = df['prog'].replace('vocation', 0)
#df['prog'] = df['prog'].replace('general', 1)
#df['prog'] = df['prog'].replace('academic', 2)
df['honors'] = df['honors'].replace('not enrolled', 0)
df['honors'] = df['honors'].replace('enrolled', 1)
print(df)

#Seperate attributes from labels
cols = [column for column in df.columns if column not in ['id', 'prog', 'cid']]
x = np.array(df[cols])
y = np.array(df['prog'])

#Train data and perform KNN classification
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 3)
knn = KNeighborsClassifier(n_neighbors = 5).fit(x_train, y_train)
pred = knn.predict(x_test)
#Print metrics
print('accuracy score: ', accuracy_score(y_test, pred))
conf_matrix = confusion_matrix(y_test, pred)
print('\nMisclassified Points:')
print('Predicted:\t Actual:')
for i in range(len(pred)):
    if pred[i] != y_test[i]:
        print(pred[i],'\t\t',y_test[i])

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Blues', 
            xticklabels = knn.classes_, yticklabels = knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
