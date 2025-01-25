import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Load dataset and split
file_i = "C:/Users/31290/Desktop/code HW/DDA3020HW1/Classification iris.xlsx"
df = pd.read_excel(file_i)

X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df['class']
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Split dataset by class
train_indices, test_indices = [], []
for label in classes:
    class_indices = df[df['class'] == label].index.tolist()
    train_size = int(len(class_indices) * 0.7)
    train_indices.extend(class_indices[:train_size])
    test_indices.extend(class_indices[train_size:])

X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

print("Q2.1.1 Split training set and test set:")
print(f"Training set: {[i + 1 for i in train_indices]}")
print(f"Test set: {[i + 1 for i in test_indices]}")

# 2. SVD decomposition
X_train_array = X_train.values
mean_vector = np.mean(X_train_array, axis=0)
X_centered = X_train_array - mean_vector
N = X_train_array.shape[0]
cov_matrix = (X_centered.T @ X_centered) / N
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)


idx = eigenvalues.argsort()[::-1]   
eigenvalues = eigenvalues[idx]        
eigenvectors = eigenvectors[:, idx]  

print("\nQ2.1.2 SVD decomposition:")
print(f"Mean vector: {mean_vector}")
print(f"Covariance matrix:\n{cov_matrix}")
print(f"Eigenvalues (vector): {eigenvalues}")
print(f"Eigenvectors (matrix):\n{eigenvectors}")

W = eigenvectors[:, :1]
X_train_centered = X_train_array - mean_vector
X_train_mapped = X_train_centered @ W
X_train_reconstruct = X_train_mapped @ W.T + mean_vector
    
    # Project and reconstruct test set
X_test_array = X_test.values
X_test_centered = X_test_array - mean_vector
X_test_mapped = X_test_centered @ W
X_test_reconstruct = X_test_mapped @ W.T + mean_vector
    

# 3-5. Project and reconstruct for different dimensions
for d in range(1, 4):
    # Projection matrix
    W = eigenvectors[:, :d]
    
    # Project and reconstruct training set
    X_train_centered = X_train_array - mean_vector
    X_train_mapped = X_train_centered @ W
    X_train_reconstruct = X_train_mapped @ W.T + mean_vector
    
    # Project and reconstruct test set
    X_test_array = X_test.values
    X_test_centered = X_test_array - mean_vector
    X_test_mapped = X_test_centered @ W
    X_test_reconstruct = X_test_mapped @ W.T + mean_vector
    

    # Calculate metrics
    train_variance = np.sum(np.var(X_train_reconstruct - mean_vector, axis=0))
    test_variance = np.sum(np.var(X_test_reconstruct - mean_vector, axis=0))
    train_loss = np.mean(np.sum((X_train_array - X_train_reconstruct) ** 2, axis=1))
    test_loss = np.mean(np.sum((X_test_array - X_test_reconstruct) ** 2, axis=1))
    
    print(f"\nQ2.1.{d+2} Project onto {d}-dimensional subspace and reconstruct:")
    print(f"Project matrix W:\n{W}")
    print(f"shape of X_train_mapped: {X_train_mapped.shape}, shape of X_train_reconstruct: {X_train_reconstruct.shape}")
    print(f"variance_train: {train_variance}")
    print(f"reconstruction_loss_train: {train_loss}")
    print(f"shape of X_test_mapped: {X_test_mapped.shape}, shape of X_test_reconstruct: {X_test_reconstruct.shape}")
    print(f"variance_test: {test_variance}")
    print(f"reconstruction_loss_test: {test_loss}")
    
    # Store results for plotting
    if d == 1:
        dimensions = []
        train_variances = []
        test_variances = []
        train_losses = []
        test_losses = []
    
    dimensions.append(d)
    train_variances.append(train_variance)
    test_variances.append(test_variance)
    train_losses.append(train_loss)
    test_losses.append(test_loss)

# 6. Plotting
print("\nQ2.1.6 Plotting:")
# Variance plot
plt.figure(figsize=(8, 6))
plt.plot(dimensions, train_variances, 'b-', label='Training')
plt.plot(dimensions, test_variances, 'r--', label='Testing')
plt.xlabel('Dimension')
plt.ylabel('Variance')
plt.title('Dimension vs Variance')
plt.legend()
plt.grid(True)
plt.show()

# Reconstruction loss plot
plt.figure(figsize=(8, 6))
plt.plot(dimensions, train_losses, 'b-', label='Training')
plt.plot(dimensions, test_losses, 'r--', label='Testing')
plt.xlabel('Dimension')
plt.ylabel('Reconstruction Loss')
plt.title('Dimension vs Reconstruction Loss')
plt.legend()
plt.grid(True)
plt.show()