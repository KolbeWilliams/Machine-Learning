#Exercise 1:
import numpy as np
import pandas as pd

df = pd.read_csv('balloons_2features.csv')
df['Act'] = df['Act'].map({'Stretch': 0, 'Dip': 1})
df['Age'] = df['Age'].map({'Child': 0, 'Adult': 1})
df['Inflated'] = df['Inflated'].map({'F': 0, 'T': 1})
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

data_point = ['Stretch', 'Adult']
if data_point[0] == 'Stretch':
    data_point[0] = 0
else:
    data_point[0] = 1
if data_point[1] == 'Child':
    data_point[1] = 0
else:
    data_point[1] = 1

def info_gain(x, y, attribute, root_entropy):
    unique_values = np.unique(x[:, attribute])
    value_counts = []
    entropies = []
    for value in unique_values:
        subset_y = y[x[:, attribute] == value]
        p_class_1 = np.sum(subset_y == 1) / len(subset_y)
        p_class_0 = 1 - p_class_1
        entropies.append(-p_class_1 * np.log2(p_class_1) - p_class_0 * np.log2(p_class_0))
        value_counts.append(np.sum(x[:, attribute] == value))
    p1 = value_counts[0] / sum(value_counts)
    p2 = 1 - p1
    entropy = p1 * entropies[0] + p2 * entropies[1]
    return root_entropy - entropy

#Find root entropy
true_values = [i for i in range(len(y)) if y[i] == 1]
num_true = len([num for num in y if num == 1])
p1 = num_true / len(y)
p2 = 1 - p1
root_entropy = -(p1 * np.log2(p1)) - (p2 * np.log2(p2))

#Find information gain of each class and select split class
info_gains = [info_gain(x, y, i, root_entropy) for i in range(x.shape[1])]
split_class = np.argmax(info_gains)

#Select second split class
if split_class == 0:
    second_split_class = 1
else:
    second_split_class = 0

#Map all paths to T or F
count_split_class0_second_split_class0 = 0
count_split_class0_second_split_class1 = 0
count_split_class1_second_split_class0 = 0
count_split_class1_second_split_class1 = 0
count_split_class0_second_split_class0_t = 0
count_split_class0_second_split_class1_t = 0
count_split_class1_second_split_class0_t = 0
count_split_class1_second_split_class1_t = 0
for i in range(x.shape[0]):
    if x[i, split_class] == 0 and x[i, second_split_class] == 0:
        count_split_class0_second_split_class0 += 1
        if y[i] == 1:
            count_split_class0_second_split_class0_t += 1
    elif x[i, split_class] == 0 and x[i, second_split_class] == 1:
        count_split_class0_second_split_class1 += 1
        if y[i] == 1:
            count_split_class0_second_split_class1_t += 1
    elif x[i, split_class] == 1 and x[i, second_split_class] == 0:
        count_split_class1_second_split_class0 += 1
        if y[i] == 1:
            count_split_class0_second_split_class1_t += 1
    elif x[i, split_class] == 1 and x[i, second_split_class] == 1:
        count_split_class1_second_split_class1 += 1
        if y[i] == 1:
            count_split_class1_second_split_class1_t += 1
count_t_arr = [count_split_class0_second_split_class0_t, count_split_class0_second_split_class1_t, count_split_class0_second_split_class1_t, count_split_class1_second_split_class1_t]
count_arr = [count_split_class0_second_split_class0, count_split_class0_second_split_class1, count_split_class0_second_split_class1, count_split_class1_second_split_class1]
result = []
for count_t, count in zip(count_t_arr, count_arr):
    if count_t >= (count - count_t):
        result.append('T')
    else:
        result.append('F')

paths = [[0, 0], [0, 1], [1, 0], [1, 1]]
for i in range(len(paths)):
    if [data_point[split_class], data_point[second_split_class]] == paths[i]:
        index = i
pred = result[index]
print('The data point is predicted to be:', pred)
