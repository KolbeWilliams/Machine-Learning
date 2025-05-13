import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('golf.csv')
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

le_x = LabelEncoder()
le_y = LabelEncoder()
x_encoded = np.array([le_x.fit_transform(col) for col in x.T]).T
y_encoded = le_y.fit_transform(y)

class_counts = np.bincount(y_encoded)
prior_yes = class_counts[1] / len(y_encoded)
prior_no = class_counts[0] / len(y_encoded)

likelyhood_yes = {}
likelyhood_no = {}

for feature_idx in range(x_encoded.shape[1]):
    likelyhood_yes[feature_idx] = {}
    likelyhood_no[feature_idx] = {}

    values = np.unique(x_encoded[:, feature_idx])
    for val in values:
        count_val_given_yes = np.sum((x_encoded[:, feature_idx] == val) & (y_encoded == 1))
        count_val_given_no = np.sum((x_encoded[:, feature_idx] == val) & (y_encoded == 0))
        likelyhood_yes[feature_idx][val] = (count_val_given_yes) / (class_counts[1])
        likelyhood_no[feature_idx][val] = (count_val_given_no) / (class_counts[0])

def predict(data_point):
    prob_no = prior_no
    prob_yes = prior_yes
    for i, value in enumerate(data_point):
        prob_no *= likelyhood_no[i][value]
        prob_yes *= likelyhood_yes[i][value]
    total = prob_no + prob_yes
    prob_no /= total
    prob_yes /= total
    return round(prob_no, 2), round(prob_yes, 2)

probabilities = []
predictions = []
for i in range(len(y)):
    pred_no, pred_yes = predict(x_encoded[i])
    probabilities.append([pred_no, pred_yes])
    predictions.append('NO' if pred_no > pred_yes else 'YES')

for i in range(len(predictions)):
    print(f'Data point {i + 1} is predicted as: {predictions[i]} with no/yes probabilities of {probabilities[i]}')