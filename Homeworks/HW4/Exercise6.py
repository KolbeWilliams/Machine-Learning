import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('materialsOutliers.csv')
x = np.array(df.iloc[:, 1:])
y = np.array(df.iloc[:, 0])

outliers = [False for i in range(x.shape[0])]

iterations = 1000
residual_threshold = 15

for feature_idx in range(x.shape[1]):
    best_inliers = None
    best_model = None
    best_inlier_count = 0

    for j in range(iterations):
        idx = np.random.choice(len(x), size=2, replace=False)
        X_subset = x[idx, feature_idx].reshape(-1, 1)
        y_subset = y[idx]
        model = LinearRegression()
        model.fit(X_subset, y_subset)

        predictions = model.predict(x[:, feature_idx].reshape(-1, 1))
        residuals = np.abs(predictions - y)

        inlier_mask = residuals <= residual_threshold
        inlier_count = np.sum(inlier_mask)
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inlier_mask
            best_model = model

    outliers = np.logical_or(outliers, np.logical_not(best_inliers))

indices_to_drop = [i for i in range(len(outliers)) if outliers[i]]
df_cleaned = df.drop(indices_to_drop)
x_cleaned = np.array(df_cleaned.iloc[:, 1:])
y_cleaned = np.array(df_cleaned.iloc[:, 0])
final_model = LinearRegression()
final_model.fit(x_cleaned, y_cleaned)

for i in range(len(final_model.coef_)):
    print(f'Coefficient {i + 1} is: {final_model.coef_[i]:.4f}')
print(f'The y-intercept is: {final_model.intercept_:.4f}')
