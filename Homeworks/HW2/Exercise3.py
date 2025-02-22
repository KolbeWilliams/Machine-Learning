#Exercise 3:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.metrics import accuracy_score
# The handwritten digits dataset contains 1797 images where each image is 8x8
# Thus, we have 64 features (8x8)
# X: features (64)
# y: label (0-9)
# Load the digits dataset
digits = load_digits()
X, y = digits.data, digits.target
print(f'Shape X: {X.shape}')
print(f'Shape y: {y.shape}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
X_train_features = X_train
X_test_features = X_test
X_train_labels = y_train
X_test_labels = y_test
nearest_distances_and_labels = []
pred = []
for i in range(X_test_features.shape[0]):
    distances = []
    labels = []
    for j in range(X_train_features.shape[0]):
        #find distance between each feature for each training point:
        distance = (np.sum((X_train_features[j, :] - X_test_features[i, :])**2))**0.5
        distances.append(distance)
        labels.append(X_train_labels[j]) #record label of each training point
    nearest_3_distances = []
    nearest_3_labels = []
    for k in range(3):
        min_distance = min(distances)
        min_index = distances.index(min_distance)
        nearest_3_distances.append(min_distance) #find closest 3 points
        nearest_3_labels.append(labels[min_index]) #find labels of closest 3 points
        distances.remove(min_distance)
        labels.pop(min_index)
    nearest_distances_and_labels.append((nearest_3_distances, nearest_3_labels))
    most_common_label = mode(nearest_3_labels)[0][0] #returns tuple of ([most common values], [frequency])
    pred.append(most_common_label) #predicts label for testing point
print('Model accuracy score: ', accuracy_score(y_test, pred))
# Visualize some samples
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, idx in zip(axes, range(5)):
    ax.imshow(X_test[idx].reshape(8,8), cmap='gray')
    ax.set_title(f'Actual: {y_test[idx]}\nPredicted: {pred[idx]}')
    ax.axis('off')
plt.show()


