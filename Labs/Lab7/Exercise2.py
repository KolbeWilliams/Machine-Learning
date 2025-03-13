#Exercise 2:
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

df = pd.read_csv('golf.csv')
df_encoded = pd.read_csv('golf.csv')

le_arr = []
for i in range(len(df_encoded.columns)):
    le = preprocessing.LabelEncoder()
    df_encoded[df_encoded.columns[i]] = le.fit_transform(df_encoded[df_encoded.columns[i]])
    le_arr.append(le)

x = np.array(df_encoded.iloc[:, :-1], dtype=int)
y = np.array(df_encoded.iloc[:, -1], dtype=int)

data_points = np.array([['Rainy', 'Hot', 'High', 'TRUE'], 
                        ['Sunny', 'Mild', 'Normal', 'FALSE'], 
                        ['Sunny', 'Cool', 'High', 'FALSE']])

for i in range(len(data_points)):
    for j in range(len(data_points[0])):
        data_points[i][j] = le_arr[j].transform([data_points[i][j]])[0]

data_points = np.array(data_points, dtype=int)
nb = GaussianNB().fit(x, y)
pred = nb.predict(data_points)
for i in range(len(pred)):
    if pred[i] == 1:
        print(f'Data point {i + 1} prediction: yes')
    else:
        print(f'Data point {i + 1} prediction: no')
