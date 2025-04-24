#The dataset used to detect credit card fraud in this program has the following features:
#   distance_from_home - the distance from home where the transaction happened.
#   distance_from_last_transaction - the distance from last transaction happened.
#   ratio_to_median_purchase_price - Ratio of purchased price transaction to median purchase price.
#   repeat_retailer - Is the transaction happened from same retailer.
#   used_chip - Is the transaction through chip (credit card).
#   used_pin_number - Is the transaction happened by using PIN number.
#   online_order - Is the transaction an online order.
#   fraud - Is the transaction fraudulent. (Target)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

#Read CSV file and obtain attributes and target columns (x and y)
df = pd.read_csv('card_transdata.csv')
x = np.array(df.iloc[:, :-1])
y = np.array(df.iloc[:, -1])

#Scale attributes
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

#Split data and cross-validation
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 15)
kf = KFold(n_splits = 5, shuffle = True, random_state = 15)
classes = ['Normal', 'Fraudulent']

classifier_list = []

#Perform KNN
knn = KNeighborsClassifier(n_neighbors = 5)
classifier_list.append(knn)
knn.fit(x_train, y_train)
pred_knn = knn.predict(x_test)
y_pred = cross_val_predict(knn, x_scaled, y, cv = 5)
accuracy_knn = accuracy_score(y_test, pred_knn)
f1_knn = f1_score(y_test, pred_knn)
scores_knn = cross_val_score(knn, x_scaled, y, cv = kf, scoring = 'accuracy')
conf_matrix_knn = confusion_matrix(y_test, pred_knn)
print(f'Classification report for KNN:\n{classification_report(y_test, pred_knn, target_names = classes)}')
print(f'Confusion matrix for KNN:\n{conf_matrix_knn}\n')
print(f'Cross-validation scores: {scores_knn}\n\n')

#Perform Logistic Regression
logReg = LogisticRegression()
classifier_list.append(logReg)
logReg.fit(x_train, y_train)
pred_logreg = logReg.predict(x_test)
accuracy_logreg = accuracy_score(y_test, pred_logreg)
f1_logreg = f1_score(y_test, pred_logreg)
scores_logReg = cross_val_score(logReg, x_scaled, y, cv = kf, scoring = 'accuracy')
conf_matrix_logreg = confusion_matrix(y_test, pred_logreg)
print(f'Classification report for Logistic Regression:\n{classification_report(y_test, pred_logreg, target_names = classes)}')
print(f'Confusion matrix for Logistic Regression:\n{conf_matrix_logreg}\n')
print(f'Cross-validation scores: {scores_logReg}')
odds = np.exp(logReg.coef_)
odds_dict = {}
for attribute, odds_ in zip(df.columns, odds[0]):
    odds_dict[attribute] = round(float(odds_), 2)
print(f'Odds for each attribute:\n{odds_dict}\n')

#Perform Naive Bayes
nb = GaussianNB()
classifier_list.append(nb)
nb.fit(x_train, y_train)
pred_nb = nb.predict(x_test)
accuracy_nb = accuracy_score(y_test, pred_nb)
f1_nb = f1_score(y_test, pred_nb)
scores_nb = cross_val_score(nb, x_scaled, y, cv = kf, scoring = 'accuracy')
conf_matrix_nb = confusion_matrix(y_test, pred_nb)
print(f'Classification report for Naive Bayes:\n{classification_report(y_test, pred_nb, target_names = classes)}')
print(f'Confusion matrix for Naive Bayes:\n{conf_matrix_nb}\n')
print(f'Cross-validation scores: {scores_nb}\n\n')

#Perform SVM
svm = SVC(kernel = 'rbf')
classifier_list.append(svm)
svm.fit(x_train, y_train)
pred_svm = svm.predict(x_test)
accuracy_svm = accuracy_score(y_test, pred_svm)
f1_svm = f1_score(y_test, pred_svm)
scores_svm = cross_val_score(svm, x_scaled, y, cv = kf, scoring = 'accuracy')
conf_matrix_svm = confusion_matrix(y_test, pred_svm)
print(f'Classification report for SVM:\n{classification_report(y_test, pred_svm, target_names = classes)}')
print(f'Confusion matrix for SVM:\n{conf_matrix_svm}\n')
print(f'Cross-validation scores: {scores_svm}\n\n')

#Perform Random Forrest
rf = RandomForestClassifier(random_state = 15)
classifier_list.append(rf)
rf.fit(x_train, y_train)
pred_rf = rf.predict(x_test)
accuracy_rf = accuracy_score(y_test, pred_rf)
f1_rf = f1_score(y_test, pred_rf)
scores_rf = cross_val_score(rf, x_scaled, y, cv = kf, scoring = 'accuracy')
conf_matrix_rf = confusion_matrix(y_test, pred_rf)
feature_importance = rf.feature_importances_
print(f'Classification report for Random Forest:\n{classification_report(y_test, pred_rf, target_names = classes)}')
print(f'Confusion matrix for Random Forest:\n{conf_matrix_rf}\n')
print(f'Cross-validation scores: {scores_rf}\n\n')
featue_importance_dict = {}
for attribute, importance in zip(df.columns, feature_importance):
    featue_importance_dict[attribute] = round(float(importance), 3)
print(f'Feature Importance for each attribute:\n{featue_importance_dict}\n')

#Create a model with pad predictions for visual comparison
bad_predictions = np.arange(len(y_test)) % 2

#Find the best model
model_list = ['KNN', 'Logistic Regression', 'Naive Bayes', 'SVM', 'Random Forest', 'Bad Model']
accuracy_list = [accuracy_knn * 100, accuracy_logreg * 100, accuracy_nb * 100, accuracy_svm * 100, accuracy_rf * 100]
f1_list = [f1_knn, f1_logreg, f1_nb, f1_svm, f1_rf]
cross_val_scores_list = [scores_knn, scores_logReg, scores_nb, scores_svm, scores_rf]
conf_matrix_list = [conf_matrix_knn, conf_matrix_logreg, conf_matrix_nb, conf_matrix_svm, conf_matrix_rf]
pred_list = [pred_knn, pred_logreg, pred_nb, pred_svm, pred_rf, bad_predictions]

best_model_index = np.argmax(f1_list)
best_model = model_list[best_model_index]
best_classifier = classifier_list[best_model_index]
best_model_accuracy = accuracy_list[best_model_index]
conf_matrix_best_model = conf_matrix_list[best_model_index]
cross_val_scores_best_model = cross_val_scores_list[best_model_index]

print(f'\nThe model with the highest accuracy is: {best_model} with an accuracy score of: {best_model_accuracy:.2f}% and an F1-Score of {f1_list[best_model_index]:.2f}')
print(f'{best_model} has cross validation scores of {cross_val_scores_best_model}')

#Find least and most influential attributes using odds and run the best model with and without those attributes
most_influential_index = np.argmax(abs(feature_importance))
least_influential_index = np.argmin(abs(feature_importance))
print(f'\nThe most influential attributes is: {df.columns[most_influential_index]}')
print(f'The least influential attribute is: {df.columns[least_influential_index]}')

#Without Most Influential
df2 = df.copy()
df2 = df2.drop(df.columns[most_influential_index], axis = 1)
x_without_most = np.array(df2.iloc[:, :-1])
x_train_without_most, x_test_without_most, y_train_without_most, y_test_without_most = train_test_split(x_without_most, y, random_state = 15)

clf = best_classifier.fit(x_train_without_most, y_train_without_most)
pred = clf.predict(x_test_without_most)
accuracy_without_most = accuracy_score(y_test_without_most, pred)

#Without Least Influential
df3 = df.copy()
df3 = df3.drop(df.columns[least_influential_index], axis = 1)
x_without_least = np.array(df2.iloc[:, :-1])
x_train_without_least, x_test_without_least, y_train_without_least, y_test_without_least = train_test_split(x_without_least, y, random_state = 15)

clf = best_classifier.fit(x_train_without_least, y_train_without_least)
pred = clf.predict(x_test_without_least)
accuracy_without_least = accuracy_score(y_test_without_least, pred)

#Compare
print(f'\n{best_model} accuracy before removing most influential or least influential attribute: {best_model_accuracy}')
print(f'{best_model} accuracy after removing most influential attribute: {accuracy_without_most}')
print(f'{best_model} accuracy after removing least influential attribute: {accuracy_without_least}')

#Accuracy Table
print('\nAccuracy Table:')
print('---------------------------------------------------------------------------------------------------------------')
print('|   Model    |       KNN       | Logistic Regression |   Naive Bayes   |        SVM       |   Random Forest   |')
print('---------------------------------------------------------------------------------------------------------------')
print(f'|  Accuracy  |      {accuracy_list[0]:.2f}%     |        {accuracy_list[1]:.2f}%       |      {accuracy_list[2]:.2f}%     |       {accuracy_list[3]:.2f}%     |       {accuracy_list[4]:.2f}%     |')
print('---------------------------------------------------------------------------------------------------------------')
print(f'|  F1-Score  |      {f1_list[0]:.2f}       |        {f1_list[1]:.2f}         |      {f1_list[2]:.2f}       |       {f1_list[3]:.2f}       |       {f1_list[4]:.2f}        |')
print('---------------------------------------------------------------------------------------------------------------')

#Perform PCA for explained variance of each attribute
pca = PCA(n_components = 7).fit(x_scaled)
principle_components = pca.transform(x_scaled)
variance_ratio = pca.explained_variance_ratio_
total_variance = np.cumsum(variance_ratio)
print(f'\nVariance Ratio: {variance_ratio}')

plt.plot(range(1, x.shape[1] + 1), total_variance, c = 'orange')
plt.title(f'Explained Variance over {x.shape[1]} Attributes')
plt.xlabel('Number of Attributes')
plt.ylabel('Cumulative Variance Ratio')
plt.grid()
plt.show()

#Plot CAP Curve for each model
fig, ax = plt.subplots(nrows = 2, ncols = 3, figsize = (10, 6))
counter = 0
for i in range(2):
    for j in range(3):
        total = len(y_test)
        one_count = np.sum(y_test)
        ax[i][j].plot([0, total], [0, one_count], c = 'b', linestyle = '--', label = 'Random Model')

        sorted_indices = np.argsort(pred_list[counter])[::-1]
        sorted_y_true = y_test[sorted_indices]
        cumulative_positives = np.cumsum(sorted_y_true)
        x_cap = np.arange(0, total + 1)
        y_cap = np.append([0], cumulative_positives)
        ax[i][j].plot(x_cap, y_cap, c = 'r', label = model_list[counter], linewidth = 2)

        ax[i][j].plot([0, one_count, total], [0, one_count, one_count], c = 'grey', linewidth = 2, label = 'Perfect Model')

        ax[i][j].set_title(model_list[counter])
        ax[i][j].set_xlabel('Total Observations')
        ax[i][j].set_ylabel('Number of Positive Predictions')
        ax[i][j].legend()
        counter += 1
plt.tight_layout()
plt.show()

#Plot confusion matrix of the best model
sns.heatmap(conf_matrix_best_model, annot = True, fmt = 'd', cmap = 'Blues', xticklabels = classes, yticklabels = classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for Best Model: {best_model}')
plt.show()

'''
pca = PCA(n_components = 2).fit(x_scaled)
principle_components = pca.transform(x_scaled)
principalDf = pd.DataFrame(data = principle_components, columns = ['principal component 1', 'principal component 2'])
principalDf['Classes'] = y
classes = df.iloc[:, -1].unique()
liClasses = list(classes)
colors = ['g', 'r']
for classes, color in zip(classes,colors):
    indicesToKeep = y == classes
    plt.scatter(principalDf.loc[indicesToKeep, 'principal component 1'] ,principalDf.loc[indicesToKeep, 'principal component 2'], c = color)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA = 2')
plt.grid()
plt.legend(['Normal', 'Fraud'])
plt.show()
'''