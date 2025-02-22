#Exercise 4:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#define function to calculate covariance
def Covariance(x, x2):
    x_mean, y_mean = 0, 0
    for i in range(len(x)):
        x_mean += x[i]
        y_mean += x2[i]
    x_mean /= len(x)
    y_mean /= len(x2)
    covar = 0
    for i in range(len(x)):
        covar += (x[i] - x_mean) * (x2[i] - y_mean)
    return covar / (len(x) - 1)

#Extract attributes and labels
df = pd.read_csv('iris.data.csv')
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
#Standardize data
for column in x.columns:
    x_mean = 0
    for i in range(len(x)):
        x_mean += x[column][i]
    x_mean /= x[column].shape[0]
    std_dev_x = 0
    for i in range(len(x)):
        std_dev_x += (x[column][i] - x_mean)**2
    std_dev_x = (std_dev_x / (len(x) - 1))**0.5
    x[column] = (x[column] - x_mean) / std_dev_x
#Perform PCA:
#Create covariance matrix with standardized data
covarMatrix = np.ones((len(x.columns),len(x.columns)))
for i in range(covarMatrix.shape[0]):
    for j in range(covarMatrix.shape[1]):
        covarMatrix[i][j] = Covariance(x[x.columns[i]],x[x.columns[j]])
#Get eigenvalues and eigenvectors (I had to research the numpy library to find this function)
eigenvalues, eigenvectors = np.linalg.eig(covarMatrix)
#Calculate variance ratio
total_variance = 0
for i in eigenvalues:
    total_variance += i
variance_ratio = []
for i in eigenvalues:
    variance_ratio.append(i / total_variance)
print('Variance Ratio: ', variance_ratio)
#Sort eigenvectors by eigenvalues in descending order
num_components = int(input('How many principle components:'))
indicies = []
for i in range(num_components):
    biggest = -np.inf #use -infinity
    for j in range(len(eigenvalues)):
        if eigenvalues[j] > biggest:
            biggest = eigenvalues[j]
            index = j
    indicies.append(index)
    eigenvalues[index] = -np.inf #can't be selected again
sorted_vectors = []
for i in range(len(indicies)):
    sorted_vectors.append(eigenvectors[:, indicies[i]])
#Calculate principal components
top_eigenvectors = np.array(sorted_vectors).T #each eigenvector is now an array
principal_components = np.array(x @ top_eigenvectors)
Columns = []
for i in range(num_components):
    Columns.append(f'PC{i+1}')
principalDf = pd.DataFrame(data = principal_components, columns = Columns)
principalDf['Classes'] = y
print(principalDf)
#Plot cumulative variance for 1-4 principle components
plt.plot(range(1,5), np.cumsum(variance_ratio))
plt.title(f'PC = 1-{len(x.columns)}')
plt.xlabel('Principle Components')
plt.ylabel('Cumulative Variance Ratio')
plt.show()
