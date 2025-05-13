#Exercise 1:
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = data.data
y = data.target
print(f'Shape of x: {X.shape}')
print(f'Shape of y: {y.shape}\n')

x_scaled = StandardScaler().fit_transform(X)

y_2d = y.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_2d)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state = 42, test_size = 0.30)

model = MLPClassifier(
    hidden_layer_sizes = (160, 160, 160),
    activation = 'relu',
    solver = 'adam',
    max_iter = 1000,
    batch_size = 10,
    alpha = 0.0001,
    random_state = 42,
    verbose = True,
    early_stopping = False)

model.fit(x_train, y_train)
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
cr = classification_report(y_test, pred)
cm = confusion_matrix(y_test, pred)

print(f'\nAccuracy Score: {accuracy}')
print(f'Classification Report:\n{cr}')
print(f'\nConfusionMatrix:\n{cm}')