#Exercise 1:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

#Define dataframe, x, and y
data = {'Square Feet:': [100, 150, 185, 235, 310, 370, 420, 430, 440, 530, 600,
                        634, 718, 750, 850, 903, 978, 1010, 1050, 1990], 
        'Price ($):': [12300, 18150, 20100, 23500, 31005, 359000, 44359, 52000, 
                       53853, 61328, 68000, 72300, 77000, 89379, 93200, 97150, 
                       102750, 115358, 119330, 323989]}
df = pd.DataFrame(data)
x = np.array(df['Square Feet:']).reshape(-1,1)
y = np.array(df['Price ($):']).reshape(-1,1)

#Perform linear regression
lr = linear_model.LinearRegression().fit(x,y)
line_y = lr.predict(x)
slope = lr.coef_
intercept = lr.intercept_

#Perform RANSAC
ransac = linear_model.RANSACRegressor().fit(x,y)
line_y_ransac = ransac.predict(x)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
slope_RANSAC = ransac.estimator_.coef_
intercept_RANSAC = ransac.estimator_.intercept_

#Print line parameters
print(f'Before RANSAC: Slope: {slope[0][0]:.2f}, y-intercept: {intercept[0]:.2f}')
print(f'After RANSAC: Slope: {slope_RANSAC[0][0]:.2f}, y-intercept: {intercept_RANSAC[0]:.2f}')

#Plot regression line before and after RANSAC, inliers, and outliers
plt.plot(x, line_y, c = 'b', linewidth = 2, label = 'Before RANSAC')
plt.plot(x, line_y_ransac, c = 'orange', linewidth = 2, label = 'After RANSAC')
plt.scatter(x[inlier_mask], y[inlier_mask], c = 'g', marker = '.', label = 'Inliers')
plt.scatter(x[outlier_mask], y[outlier_mask], c = 'r', marker = '.', label = 'Outliers')
plt.title('Linear Regression Before and After RANSAC')
plt.xlabel('Square Feet')
plt.ylabel('Price')
plt.legend(loc = 'lower right')
plt.show()

