import numpy as np
import pandas as pd

def entropy(y):
    if len(y) == 0:
        return 0
    probs = np.bincount(y) / len(y)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def info_gain(x, y, root_entropy, feature_index):
    values = np.unique(x[:, feature_index])
    weighted_entropy = 0
    for val in values:
        subset_y = y[x[:, feature_index] == val]
        weighted_entropy += len(subset_y) / len(y) * entropy(subset_y)
    return root_entropy - weighted_entropy

def build_tree(x, y, features, feature_indicies):
    if len(np.unique(y)) == 1:
        return y[0]
    if len(features) == 0:
        return np.argmax(np.bincount(y))

    root_entropy = entropy(y)
    info_gains = [info_gain(x, y, root_entropy, i) for i in feature_indicies]
    best_feature_index = np.argmax(info_gains)
    best_feature = features[best_feature_index]

    tree = {best_feature: {}}

    unique_values = np.unique(x[:, best_feature_index])
    for val in unique_values:
        subset_x = x[x[:, best_feature_index] == val]
        subset_y = y[x[:, best_feature_index] == val]
        remaining_features = [f for i, f in enumerate(features) if i != best_feature_index]
        remaining_indicies = [i for i in range(len(remaining_features))]
        subset_x = np.delete(subset_x, best_feature_index, axis = 1)
        tree[best_feature][val] = build_tree(subset_x, subset_y, remaining_features, remaining_indicies)
    return tree

def predict(tree, data_point, features):
    if not isinstance(tree, dict):
        return tree

    feature = list(tree.keys())[0]
    index = features.index(feature)
    value = data_point[index]
    return predict(tree[feature][value], data_point, features)

df = pd.read_csv('balloons_extended.csv')
df['Color'] = df['Color'].map({'YELLOW': 0, 'PURPLE': 1})
df['size'] = df['size'].map({'SMALL': 0, 'LARGE': 1})
df['act'] = df['act'].map({'STRETCH': 0, 'DIP': 1})
df['age'] = df['age'].map({'ADULT': 0, 'CHILD': 1})
df['inflated'] = df['inflated'].map({'T': 0, 'F': 1})

x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

data_point = ['YELLOW', 'SMALL', 'DIP', 'ADULT']
for i in range(len(data_point)):
    if data_point[i] in ['YELLOW', 'SMALL', 'STRETCH', 'ADULT']:
        data_point[i] = 0
    if data_point[i] in ['PURPLE', 'LARGE', 'DIP', 'CHILD']:
        data_point[i] = 1

features = list(df.columns[:-1])
feature_indicies = list(range(len(features)))

tree = build_tree(x, y, features, feature_indicies)
pred = predict(tree, data_point, features)
prediction = 'T' if pred == 0 else 'F'
print(f'The data point is predicted to be: {prediction}')