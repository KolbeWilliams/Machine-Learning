#Exercise 1:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

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
knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
pred = knn.predict(X_test)
print('Model accuracy score: ', accuracy_score(y_test, pred))
conf_matrix = confusion_matrix(y_test, pred)

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
xticklabels=knn.classes_, yticklabels=knn.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Visualize some samples
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, idx in zip(axes, range(5)):
    ax.imshow(X_test[idx].reshape(8,8), cmap='gray')
    ax.set_title(f'Actual: {y_test[idx]}, Predicted: {pred[idx]}')
    ax.axis('off')
plt.show()

