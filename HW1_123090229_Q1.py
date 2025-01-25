import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Load the training data
train_data_path = r"C:\Users\31290\Desktop\linear_regression_train.txt"
test_data_path = r"C:\Users\31290\Desktop\linear_regression_test.txt"

train_data = np.loadtxt(train_data_path)
test_data = np.loadtxt(test_data_path)

# Separate features (X) and labels (y) in the training data
X_train = train_data[:, :-1]
y_train = train_data[:, -1]
X_test = test_data

# Using sklearn for linear regression
model = LinearRegression()

# Fit the model
model.fit(X_train, y_train)

# Get the bias and coefficients
w0 = model.intercept_
w_coefficients = model.coef_

# Predict on the test set
y_pred = model.predict(X_test)

# Output the results from sklearn
print("Using sklearn:")
print(f"Bias (w0): {w0}")
print(f"Coefficients (w1, w2, ..., wd): {w_coefficients}")
print(f"Predicted values (ŷ): {y_pred}")

# My implementation using Normal Equation
# Add a column of ones to X_train for the bias term
X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

# Calculate the coefficients using the Normal Equation
# w = (X^T X)^(-1) X^T y
w = np.linalg.inv(X_train_bias.T @ X_train_bias) @ (X_train_bias.T @ y_train)

# Extract bias and coefficients
w0_manual = w[0]
w_coefficients_manual = w[1:]

# Prepare the test set similarly
X_test_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Predict using the custom model
y_pred_manual = X_test_bias @ w

# Output the results from custom implementation
print("My Implementation:")
print(f"Bias (w0): {w0_manual}")
print(f"Coefficients (w1, w2, ..., wd): {w_coefficients_manual}")
print(f"Predicted values (ŷ): {y_pred_manual}")
