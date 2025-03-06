#Exercise 2:
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2

train_df = pd.read_csv('fashion-mnist_train.csv')
test_df = pd.read_csv('fashion-mnist_test.csv')
x_train = np.array(train_df.iloc[:, 1:])
y_train = np.array(train_df.iloc[:, 0])
x_test = np.array(test_df.iloc[:, 1:])
y_test = np.array(test_df.iloc[:, 0])

logReg = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial').fit(x_train, y_train)

'''
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
'''

img1 = cv2.imread('trousers.bmp', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('bag.jpg', cv2.IMREAD_GRAYSCALE)

img1_reshaped = img1.reshape(1, 28 * 28)
pred_trousers = logReg.predict(img1_reshaped)
img2_reshaped = img2.reshape(1, 28 * 28)
pred_bag = logReg.predict(img2_reshaped)

label_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
print('\nThe trousers image is predicted to be: ', label_dict[pred_trousers[0]])
print('The bag image is predicted to be: ', label_dict[pred_bag[0]])
