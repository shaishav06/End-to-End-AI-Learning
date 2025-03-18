# 🤖 Classical Machine Learning Models

Classical Machine Learning (ML) models are widely used for predictive modeling. These models can be categorized into regression and classification techniques. This guide covers **Linear & Logistic Regression, Decision Trees, Random Forest, and XGBoost**. 🚀

![Classical ML Models](../images/Classical%20Machine%20Learning%20Models%20.png)

---

## 📏 1. Linear & Logistic Regression

### 📈 1.1 Linear Regression
Linear Regression models the relationship between independent variables (X) and a continuous dependent variable (Y) using a linear equation:

**Equation:**
\[
Y = b_0 + b_1X + \epsilon
\]
where:
- 🔹 \(Y\) is the predicted output,
- 🔹 \(b_0\) is the intercept,
- 🔹 \(b_1\) is the coefficient (slope),
- 🔹 \(X\) is the input feature,
- 🔹 \(\epsilon\) is the error term.

**Example: Predicting House Prices** 🏡
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1000, 1500, 2000, 2500, 3000]).reshape(-1, 1)
Y = np.array([200000, 250000, 300000, 350000, 400000])

model = LinearRegression()
model.fit(X, Y)
predicted_price = model.predict([[1800]])
print(f"🏠 Predicted Price: ${predicted_price[0]:,.2f}")
```

### 🏆 1.2 Logistic Regression
Logistic Regression is used for binary classification problems and predicts the probability of an event occurring using the sigmoid function:

**Example: Predicting if a student passes an exam based on study hours** 📚
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])  # Pass (1) or Fail (0)

model = LogisticRegression()
model.fit(X, Y)
prediction = model.predict([[4.5]])
print("📊 Predicted Outcome:", "Pass" if prediction[0] == 1 else "Fail")
```

---

## 🌳 2. Decision Trees
Decision Trees split data based on feature values to classify new instances. They follow a hierarchical structure where each node represents a feature, and each branch represents a decision rule.

**Example: Classifying whether a customer will buy a product** 🛍️
```python
from sklearn.tree import DecisionTreeClassifier

X = [[25, 50000], [35, 60000], [45, 80000], [20, 20000], [50, 100000]]
y = [1, 1, 1, 0, 1]  # Buy (1) or Not (0)

dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)
print("🛒 Prediction:", dt_model.predict([[30, 70000]]))
```

---

## 🌲 3. Random Forest
Random Forest is an ensemble method that builds multiple decision trees and takes the majority vote for classification.

**Example: Predicting if a patient has a disease based on symptoms** 🏥
```python
from sklearn.ensemble import RandomForestClassifier

X = [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [0, 0, 0]]  # Symptoms
y = [1, 0, 1, 1, 0]  # Disease (1) or Not (0)

rf_model = RandomForestClassifier(n_estimators=10)
rf_model.fit(X, y)
print("🩺 Prediction:", rf_model.predict([[1, 0, 0]]))
```

---

## ⚡ 4. XGBoost
XGBoost (Extreme Gradient Boosting) is a powerful boosting algorithm that improves model performance by combining multiple weak models iteratively.

### ✅ Advantages of XGBoost:
- Handles missing data automatically.
- Faster training time using parallelization.
- L1 & L2 regularization to prevent overfitting.

**Example: Predicting Customer Churn** 💼
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

X = np.array([[34, 40000], [45, 60000], [23, 20000], [40, 50000], [50, 100000]])
y = np.array([0, 1, 0, 0, 1])  # Churn (1) or Not (0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(objective="binary:logistic", n_estimators=10)
xgb_model.fit(X_train, y_train)
y_pred = xgb_model.predict(X_test)
print("📊 Accuracy:", accuracy_score(y_test, y_pred))
```

---

## 🎯 Conclusion
Classical ML models, including **Regression, Decision Trees, and Ensemble Methods** like **Random Forest and XGBoost**, are powerful tools for structured data analysis. Choosing the right model depends on the **problem type (regression vs classification), data characteristics, and performance requirements**. 🚀

📖 **[Back to Main README](../README.md)**
