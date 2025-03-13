#Exercise 3:
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing

df = pd.read_csv('golf.csv')
le_x = preprocessing.LabelEncoder()
le_y = preprocessing.LabelEncoder()

x = np.array(df.iloc[:, 0])
y = np.array(df.iloc[:, -1])
y_encoded = le_y.fit_transform(y)
x_encoded = le_x.fit_transform(x).reshape(-1, 1)

nb = GaussianNB()
nb.fit(x_encoded, y_encoded)

pred_var = ['Rainy']
pred_var_encoded = le_x.transform(pred_var).reshape(-1, 1)
pred_built_in = nb.predict_proba(pred_var_encoded)
print('Using built-in function:\n', pred_built_in[0])

count_yes, count_var_given_yes, count_var = 0, 0, 0
for i in range(len(y)):
    if y[i] == 'Yes':
        count_yes += 1
    if y[i] == 'Yes' and x[i] == pred_var[0]:
        count_var_given_yes += 1
    if x[i] == pred_var[0]:
        count_var += 1
prior_prob_yes = count_yes / len(y)
prior_prob_no = 1 - prior_prob_yes

prob_var_given_yes = count_var_given_yes / count_yes
prob_var_given_no = (count_var - count_var_given_yes) / (len(y) - count_yes)

posterior_var_yes = prob_var_given_yes * prior_prob_yes
posterior_var_no = prob_var_given_no * prior_prob_no

total = posterior_var_yes + posterior_var_no
posterior_yes = posterior_var_yes / total
posterior_no = posterior_var_no / total
probabilities = [posterior_yes, posterior_no]
print('Without using built-in function:\n', probabilities)