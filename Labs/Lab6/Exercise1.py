#Exercise 1:
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')
x_train = np.array(train_df.iloc[:, 1:])
y_train = np.array(train_df.iloc[:, 0])
x_test = np.array(test_df.iloc[:, 1:])
y_test = np.array(test_df.iloc[:, 0])

logReg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial').fit(x_train, y_train)
pred = logReg.predict(x_test)
accuracyScore = accuracy_score(y_test, pred)
classificationReport = classification_report(y_test, pred)
cm = confusion_matrix(y_test, pred)
print('Accuracy Score: ', accuracyScore)
print('Classification Report: ', classificationReport)
print('Confusion Matrix:\n', cm)

disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = logReg.classes_)
disp.plot()
plt.show()