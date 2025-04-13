#Exercise 2:
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import sklearn.tree as tree
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('balloons_extended.csv')
df['Color'] = df['Color'].map({'YELLOW': 0, 'PURPLE': 1})
df['size'] = df['size'].map({'SMALL': 0, 'LARGE': 1})
df['act'] = df['act'].map({'STRETCH': 0, 'DIP': 1})
df['age'] = df['age'].map({'ADULT': 0, 'CHILD': 1})
df['inflated'] = df['inflated'].map({'T': 1, 'F': 0})
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

clf = RandomForestClassifier(n_estimators = 100).fit(x_train, y_train)
important_features = clf.feature_importances_
most_important = df.columns[np.argmax(important_features)]
#text_representation = tree.export_text(clf)
pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
report = classification_report(y_test, pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')
print(f'Confusion Matrix:\n{cm}')
print(f'\nFeature Importances: {important_features}')
print(f'The most important feature is: {most_important}')
#print(f'\nDecision Tree:\n{text_representation}')

sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues',
xticklabels = ['F', 'T'], yticklabels = ['F', 'T'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

fig, axes = plt.subplots(nrows = 2, ncols = 5, figsize = (30, 10))
axes = axes.flatten()
for i in range(10):
    ax = axes[i]
    plot_tree(clf.estimators_[i], feature_names = df.columns[:-1], class_names = ['F', 'T'], filled = True, ax = axes[i])
    ax.set_title(f'Decision Tree {i + 1} Visualization:')
plt.tight_layout()
plt.show()