import pandas as pd
import numpy as np
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier

# Load dataset
file_i="C:/Users/31290/Desktop/Classification iris.xlsx"
df = pd.read_excel(file_i)

X = df[['sepal length', 'sepal width', 'petal length', 'petal width']]
y = df['class']
instance_ids = df['instance_id']

# Define classes
classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

# Manually split the dataset into training and testing sets by class
train_indices, test_indices = [], []
for label in classes:
    # Get data indices for each class
    class_indices = df[df['class'] == label].index.tolist()
    train_size = int(len(class_indices) * 0.7)
    # First 70% as training set, last 30% as test set
    train_indices.extend(class_indices[:train_size])
    test_indices.extend(class_indices[train_size:])

# Split dataset based on these indices
X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

print("Q2.2.1 Split training set and test set:")
print(f"Training set: {[i + 1 for i in train_indices]}")
print(f"Test set: {[i + 1 for i in test_indices]}")


# Q2.2.2 Calculation using Standard SVM Model
svm_model = OneVsRestClassifier(SVC(kernel='linear', C=1e5, decision_function_shape='ovr'))
svm_model.fit(X_train, y_train)
y_train_pred = svm_model.predict(X_train)
total_training_error = 1 - accuracy_score(y_train, y_train_pred)
y_test_pred = svm_model.predict(X_test)
total_testing_error = 1 - accuracy_score(y_test, y_test_pred)


def train_svm(X_train, y_train, x_test, y_test, class_label):
    y_train_binary = (y_train == class_label).astype(int)
    y_test_binary = (y_test == class_label).astype(int)

    train_svm = SVC(kernel='linear', C=1e5)
    train_svm.fit(X_train, y_train_binary)

    y_train_pred = train_svm.predict(X_train)
    y_test_pred = train_svm.predict(x_test)

    train_error = 1 - accuracy_score(y_train_binary, y_train_pred)
    test_error = 1 - accuracy_score(y_test_binary, y_test_pred)

    w = train_svm.coef_[0]
    b = train_svm.intercept_[0]

    return train_error, test_error, w, b, train_svm.support_

def svm_analysis(X_train, y_train, X_test, y_test, classes):
    
    linearly_separable_classes = []
    print(f"total training error: {total_training_error}, total testing error: {total_testing_error}")
    print()

    # for each class, train SVM and calculate training and testing errors
    for class_label in classes:
        train_error, test_error, w, b, support_vector_indices = train_svm(X_train, y_train, X_test, y_test, class_label)
        
        if train_error == 0 and test_error == 0:
            linearly_separable_classes.append(class_label)

        print(f"class {class_label}:")
        print(f"training error: {train_error}, testing error: {test_error}")
        print(f"w: {w}, b: {b}")
        print(f"support vector indices: {support_vector_indices}")
        print()

    print(f"Linearly separable classes: {', '.join(linearly_separable_classes)}")
    print()


print("Q2.2.2 Calculation using Standard SVM Model:")
svm_analysis(X_train, y_train, X_test, y_test, classes)


#SVM with different C
def calculate_slack(y, X, w, b):
    f_X = np.dot(X, w) + b
    slack = np.maximum(0, 1 - y * f_X)
    return slack

def train_svm_with_slack(X_train, y_train, x_test, y_test, class_label, C):
    y_train_binary = (y_train == class_label).astype(int)
    y_train_binary[y_train_binary == 0] = -1  

    y_test_binary = (y_test == class_label).astype(int)
    y_test_binary[y_test_binary == 0] = -1

    train_svm = SVC(kernel='linear', C=C)
    train_svm.fit(X_train, y_train_binary)

    y_train_pred = train_svm.predict(X_train)
    y_test_pred = train_svm.predict(x_test)

    train_error = 1 - accuracy_score(y_train_binary, y_train_pred)
    test_error = 1 - accuracy_score(y_test_binary, y_test_pred)

    w = train_svm.coef_[0]
    b = train_svm.intercept_[0]

    slack=calculate_slack(y_train_binary, X_train, w, b)

    return train_error, test_error, w, b, train_svm.support_, slack



def svm_analysis_with_slack(X_train, y_train, X_test, y_test, classes, C_values):
    for C in C_values:
        print(f"Q2.2.3 Calculation using SVM Model with C={C}:")
        
        print(f"-----------------------------------\nC={C}\n")
        C_values = [0.25*t for t in range(1, 5)]
        svm_model = OneVsRestClassifier(SVC(kernel='linear', C=C, decision_function_shape='ovr'))
        svm_model.fit(X_train, y_train)
        y_train_pred = svm_model.predict(X_train)
        total_training_error = 1 - accuracy_score(y_train, y_train_pred)
        y_test_pred = svm_model.predict(X_test)
        total_testing_error = 1 - accuracy_score(y_test, y_test_pred)
        print(f"total training error: {total_training_error}, total testing error: {total_testing_error}")  
        print()

        for class_label in classes:
            train_error, test_error, w, b, support_vector_indices,slack = train_svm_with_slack(X_train, y_train, X_test, y_test, class_label, C)
            
            
            print(f"class {class_label}:")
            print(f"training error: {train_error}, testing error: {test_error}")
            print(f"w: {w}, b: {b}")
            print(f"support vector indices: {support_vector_indices}")
            print(f"slack: {slack.tolist()}")
            print()

        
    
print("Q2.2.3 Calculation using SVM Model with Slack Variables(C=0.25*t,where t =1,...,4):")
C_values = [0.25*t for t in range(1, 5)]

svm_analysis_with_slack(X_train, y_train, X_test, y_test, classes, C_values)



#SVM with different kernel
print("Q2.2.4 Calculation using SVM Model with Different Kernels:")
kernel_types = [
    ('poly', 2),  # 2nd-order Polynomial Kernel
    ('poly', 3),  # 3rd-order Polynomial Kernel
    ('rbf', 0.5),   # Radial Basis Function Kernel with σ = 1,gamma=1/2*σ^2
    ('sigmoid', 0.5) # Sigmoidal Kernel with σ = 1,gamma=1/2*σ^2
]


def train_svm_with_kernel(X_train, y_train, x_test, y_test, class_label, kernel,degree=None, gamma='scale'):
    
    y_train_binary = (y_train == class_label).astype(int)
    y_train_binary[y_train_binary == 0] = -1  

    y_test_binary = (y_test == class_label).astype(int)
    y_test_binary[y_test_binary == 0] = -1
    if degree:
        train_svm = SVC(kernel=kernel, degree=degree, C=1e5)
    else:
        train_svm = SVC(kernel=kernel, gamma=gamma, C=1e5)
    train_svm.fit(X_train, y_train_binary)

    y_train_pred = train_svm.predict(X_train)
    y_test_pred = train_svm.predict(x_test)

    train_error = 1 - accuracy_score(y_train_binary, y_train_pred)
    test_error = 1 - accuracy_score(y_test_binary, y_test_pred)
    
    b=train_svm.intercept_[0]
    support_vector_indices = train_svm.support_
    return train_error, test_error, b, support_vector_indices

def svm_analysis_with_kernel(X_train, y_train, X_test, y_test, classes, kernel_type,degree=None, gamma='scale'):
    
    for class_label in classes:
        train_error, test_error, b, support_vector_indices = train_svm_with_kernel(X_train, y_train, X_test, y_test, class_label, kernel_type,degree, gamma)
        
        print(f"class {class_label}:")
        print(f"training error: {train_error}, testing error: {test_error}")
        print(f"b: {b}")
        print(f"support vector indices: {support_vector_indices}")
        print()
    
    
for kernel_type, parameter in kernel_types:
    if kernel_type == 'poly':
        svm_model = OneVsRestClassifier(SVC(kernel=kernel_type, degree=parameter, C=1e5, decision_function_shape='ovr'))
    else:
        svm_model = OneVsRestClassifier(SVC(kernel=kernel_type, gamma=parameter, C=1e5, decision_function_shape='ovr'))
    svm_model.fit(X_train, y_train)
    y_train_pred = svm_model.predict(X_train)
    total_training_error = 1 - accuracy_score(y_train, y_train_pred)
    y_test_pred = svm_model.predict(X_test)
    total_testing_error = 1 - accuracy_score(y_test, y_test_pred)
    print(f"------------------------------------------\nQ2.2.4 Calculation using SVM Model with {kernel_type} kernel ({'degree=' + str(parameter) if kernel_type == 'poly' else 'gamma=' + str(parameter)}):")
    print(f"Total training error: {total_training_error}, Total testing error: {total_testing_error}")
    if kernel_type == 'poly':
        svm_analysis_with_kernel(X_train, y_train, X_test, y_test, classes, kernel_type='poly', degree=parameter) 
    else:
        svm_analysis_with_kernel(X_train, y_train, X_test, y_test, classes, kernel_type=kernel_type, gamma=parameter) 




