import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Load diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature for visualization purposes
feature_index = 2  # You can change this to use different features
diabetes_X = diabetes.data[:, feature_index].reshape(-1, 1)  # Select and reshape the feature
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

# Create and fit the model
model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

# Predict
diabetes_y_predicted = model.predict(diabetes_X_test)

# Print metrics
print("Mean squared error is: ", mean_squared_error(diabetes_y_test, diabetes_y_predicted))
print("Weights: ", model.coef_)
print("Intercept: ", model.intercept_)

# Plot
plt.scatter(diabetes_X_test, diabetes_y_test, color='black', label='Test data')
plt.plot(diabetes_X_test, diabetes_y_predicted, color='blue', linewidth=3, label='Regression line')
plt.xlabel('Feature value')
plt.ylabel('Target value')
plt.title('Linear Regression on Diabetes Dataset')
plt.legend()
plt.show()
