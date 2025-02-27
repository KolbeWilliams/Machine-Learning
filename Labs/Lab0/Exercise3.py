#Exercise 3:
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

df = pd.read_csv('housing.csv')
x = np.array(df['Longitude'])
y = np.array(df['Latitude'])
data = []
for i in range(len(x)):
    data.append([x[i], y[i]])
n_clusters = 6
kmeans = KMeans(n_clusters = n_clusters).fit(data)
centroidsK = kmeans.cluster_centers_
labelsK = kmeans.labels_

plt.scatter(x, y, c = labelsK)
for centroid in centroidsK:
    plt.scatter(centroid[0], centroid[1], color = 'r')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'K-means = {n_clusters}')
plt.show()
