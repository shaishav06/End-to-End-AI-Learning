# Supervised Learning

Supervised Learning is a type of machine learning where the model is trained on labeled data, meaning the dataset contains both input features and corresponding correct outputs. The algorithm learns from this data to make predictions on new, unseen data.

## 1. Regression
Regression is used to predict continuous values based on input features.

### 1.1 Linear Regression
Linear Regression models the relationship between independent variables (X) and a continuous dependent variable (Y) using a linear equation:

**Equation:**
\[
Y = b_0 + b_1X + \epsilon
\]
Where:
- \(Y\) is the predicted value.
- \(b_0\) is the intercept.
- \(b_1\) is the coefficient (slope of the line).
- \(X\) is the input feature.
- \(\epsilon\) is the error term.

**Example:** Predicting house prices based on square footage.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)  # Square footage
Y = np.array([200000, 250000, 300000, 350000, 400000])  # House prices

# Model training
model = LinearRegression()
model.fit(X, Y)

# Prediction
predicted_price = model.predict([[1800]])
print(f"Predicted Price: ${predicted_price[0]:,.2f}")
```

### 1.2 Logistic Regression
Logistic Regression is used for binary classification problems. It predicts the probability of an event occurring using the sigmoid function:

**Sigmoid Function:**
\[
P(Y=1) = \frac{1}{1 + e^{- (b_0 + b_1X)}}
\]
Where the output is between 0 and 1.

**Example:** Predicting whether a student will pass an exam based on study hours.
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Sample data
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # Study hours
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Pass (1) or Fail (0)

# Model training
model = LogisticRegression()
model.fit(X, Y)

# Prediction
prediction = model.predict([[4.5]])
print("Predicted Outcome:", "Pass" if prediction[0] == 1 else "Fail")
```

## 2. Classification
Classification models categorize input data into discrete classes.

### 2.1 Support Vector Machines (SVM)
SVM finds the best hyperplane that separates different classes by maximizing the margin between them.

**Example:** Classifying spam vs. non-spam emails.
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Prediction
predictions = svm_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
```

### 2.2 Decision Trees
Decision Trees split data based on feature values to classify new instances.

**Example:** Classifying whether a person will buy a product.
```python
from sklearn.tree import DecisionTreeClassifier

# Sample data
X = [[25, 50000], [35, 60000], [45, 80000], [20, 20000], [50, 100000]]  # Age, Salary
y = [1, 1, 1, 0, 1]  # Buy (1) or Not (0)

# Train model
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# Prediction
print("Prediction:", dt_model.predict([[30, 70000]]))
```

### 2.3 Random Forest
Random Forest is an ensemble method that builds multiple decision trees and takes the majority vote for classification.

**Example:** Predicting if a patient has a disease based on symptoms.
```python
from sklearn.ensemble import RandomForestClassifier

# Sample data
X = [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]]  # Symptoms
y = [1, 0, 1, 1, 0]  # Disease (1) or Not (0)

# Train model
rf_model = RandomForestClassifier(n_estimators=10)
rf_model.fit(X, y)

# Prediction
print("Prediction:", rf_model.predict([[1, 0, 0]]))
```

## 3. Neural Networks
Neural Networks are inspired by the human brain and consist of layers of interconnected neurons.

### 3.1 Simple Neural Network
A basic neural network consists of:
- **Input Layer**: Takes input features.
- **Hidden Layers**: Apply transformations using activation functions.
- **Output Layer**: Produces final predictions.

**Example:** Handwritten digit classification using TensorFlow.
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import mnist

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train.reshape(-1, 28*28) / 255.0, X_test.reshape(-1, 28*28) / 255.0

# Model creation
model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile & Train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

## Conclusion
Supervised Learning is one of the most widely used AI techniques, applied in fields ranging from healthcare to finance. With regression, classification models, and neural networks, AI can make accurate predictions and automate decision-making processes.

---
### [Back to Main README](../README.md)