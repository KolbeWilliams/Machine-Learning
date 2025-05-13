import numpy as np
import pandas as pd
from sklearn import preprocessing

# Read the CSV file
df = pd.read_csv('golf.csv')

# Preprocessing: Label encode both the features and target
le_x = preprocessing.LabelEncoder()
le_y = preprocessing.LabelEncoder()

# Assuming the first columns are the features and the last column is the target
# You can update the column indices depending on your CSV file
x = np.array(df.iloc[:, :-1])  # All columns except the last one (features)
y = np.array(df.iloc[:, -1])  # The last column (target)

# Encode the features and target
x_encoded = np.array([le_x.fit_transform(col) for col in x.T]).T  # Apply label encoding to each feature column
y_encoded = le_y.fit_transform(y)

# Step 1: Calculate prior probabilities
class_counts = np.bincount(y_encoded)
prior_prob_yes = class_counts[1] / len(y_encoded)  # P(Yes)
prior_prob_no = class_counts[0] / len(y_encoded)  # P(No)

# Step 2: Calculate likelihoods (P(Feature | Class))
# For each feature, we calculate the probability of each possible feature value for each class
likelihood_yes = {}
likelihood_no = {}

# Loop over each feature
for feature_idx in range(x_encoded.shape[1]):
    # Calculate the likelihood for each feature value for class "Yes"
    #likelihood_yes[feature_idx] = {}
    #likelihood_no[feature_idx] = {}
    
    unique_feature_values = np.unique(x_encoded[:, feature_idx])  # Get unique values for the feature
    
    for value in unique_feature_values:
        # Count occurrences of value given class "Yes"
        count_feature_given_yes = np.sum((x_encoded[:, feature_idx] == value) & (y_encoded == 1))
        count_feature_given_no = np.sum((x_encoded[:, feature_idx] == value) & (y_encoded == 0))
        
        # Count occurrences of class "Yes" and "No"
        count_yes = class_counts[1]
        count_no = class_counts[0]
        
        # Compute likelihoods
        likelihood_yes[feature_idx][value] = (count_feature_given_yes + 1) / (count_yes + len(unique_feature_values))  # Laplace smoothing
        likelihood_no[feature_idx][value] = (count_feature_given_no + 1) / (count_no + len(unique_feature_values))  # Laplace smoothing

# Step 3: Make predictions for each data point
def predict(features):
    # Calculate the posterior probabilities for "Yes" and "No"
    posterior_yes = prior_prob_yes
    posterior_no = prior_prob_no

    # Multiply by the likelihoods for each feature
    for feature_idx, feature_value in enumerate(features):
        posterior_yes *= likelihood_yes[feature_idx].get(feature_value, 1 / (len(y_encoded) + len(np.unique(x_encoded[:, feature_idx]))))
        posterior_no *= likelihood_no[feature_idx].get(feature_value, 1 / (len(y_encoded) + len(np.unique(x_encoded[:, feature_idx]))))

    # Normalize the probabilities (to sum to 1)
    total = posterior_yes + posterior_no
    posterior_yes /= total
    posterior_no /= total
    
    return posterior_yes, posterior_no

# Step 4: Predict for all data points
predictions = []
probabilities = []
for i in range(x_encoded.shape[0]):
    posterior_yes, posterior_no = predict(x_encoded[i])
    predictions.append(le_y.inverse_transform([1 if posterior_yes > posterior_no else 0])[0])
    probabilities.append([posterior_yes, posterior_no])

# Print the predictions and probabilities for each data point
for i in range(len(predictions)):
    print(f"Data Point {i+1}: Features: {x[i]}, Predicted Class: {predictions[i]}, Probabilities (Y/N): {probabilities[i][0]:.2f}, {probabilities[i][1]:.2f}")