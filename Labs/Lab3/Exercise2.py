#Exercise 2:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns

df = pd.read_csv('recipes_muffins_cupcakes_scones.csv')
df['Type'] = df['Type'].replace({'Muffin': 0, 'Cupcake': 1, 'Scone': 2})
x = df.iloc[:, 1:]
y = df['Type']

scaler = StandardScaler()
x = scaler.fit_transform(x)
pca = PCA(n_components = 8)
principal_Components = pca.fit_transform(x)
variance_ratio = pca.explained_variance_ratio_
cummulative_sum = np.cumsum(variance_ratio)
print('\nVariance Ratio: ', variance_ratio)

pc1 = principal_Components[:, 0]
pc2 = principal_Components[:, 1]
print('\nPC1: ', pc1)
print('\nPC2: ', pc2)

#Cummulative Variance
plt.plot(range(1,9), cummulative_sum)
plt.title('PC = 1-8')
plt.xlabel('Principal Components')
plt.ylabel('Cummulative Variance Ratio')
plt.show()

#Exercise 2
data_point = np.array([38, 18, 23, 20, 9, 3, 1, 0]).reshape(1, -1)
data_point = scaler.transform(data_point)
data_point_PCA = pca.transform(data_point)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
pred = knn.predict(data_point_PCA)

if pred == 0:
    prediction = 'Muffin'
elif pred == 1:
    prediction = 'Cupcake' 
else:
    prediction = 'Scone'
print(f'\nThe data point is predicted to be a {prediction}')

#Scatter Plot
plt.scatter(pc1, pc2, c = y)
plt.scatter(data_point_PCA[0, 0], data_point_PCA[0, 1], c = 'r')
plt.title(f'PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

#Histogram
fig,axes =plt.subplots(2,4, figsize=(12, 9))
ax=axes.ravel()
features = df.iloc[:, 1:9]
muffin=df[df['Type']==0]
cupcake=df[df['Type']==1]
scone=df[df['Type']==2]
for i, feature in enumerate(features):
  _,bins=np.histogram(df[feature],bins=25)
  ax[i].hist(muffin[feature],bins=bins,color='r',alpha=.5, label = 'Muffin')
  ax[i].hist(cupcake[feature],bins=bins,color='g',alpha=0.3, label = 'Cupcake')
  ax[i].hist(scone[feature],bins=bins,color='b',alpha=0.3, label = 'Scone')
  ax[i].set_title(feature,fontsize=9)
  ax[i].axes.get_xaxis().set_visible(False)
  ax[i].set_yticks(())
ax[0].legend(['Muffin','Cupcake', 'Scone'],loc='best',fontsize=8)
plt.tight_layout()
plt.show()

#Variation Heatmap
names = df.columns[1:]
plt.matshow(pca.components_[:2, :],cmap='viridis')
plt.yticks([0,1],['PC1','PC2'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(names)),names,rotation=65,ha='left')
plt.show()

pc1_min = np.min(pca.components_[0,:])
pc1_max = np.max(pca.components_[0,:])
pc2_min = np.min(pca.components_[1,:])
pc2_max = np.max(pca.components_[1,:])
low_index = 0
high_index = 0
for i in range(len(pca.components_[0,:])):
    if pca.components_[0][i] == pc1_min:
        low_index = i
    if pca.components_[0][i] == pc1_max:
        high_index = i
print('\nThe feature with the lowest variation in PC1 is: ', names[low_index])
print('The feature with the highest variation in PC1 is: ', names[high_index])
for i in range(len(pca.components_[1,:])):
    if pca.components_[1][i] == pc2_min:
        low_index = i
    if pca.components_[1][i] == pc2_max:
        high_index = i
print('The feature with the lowest variation in PC2 is: ', names[low_index])
print('The feature with the highest variation in PC2 is: ', names[high_index])

#Correlation Heatmap
df = df.drop(['Type'], axis = 1)
s=sns.heatmap(df.corr(),cmap='coolwarm') 
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=7)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=7)
plt.show()