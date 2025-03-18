# ⚡ AI Model Training & Optimization

Model training is a critical step in machine learning, where the model learns patterns from data to make predictions. This process involves selecting the right **hyperparameters**, optimizing **loss functions**, and leveraging **pretrained models** for **transfer learning**. 🚀

![modeltraining](../images/AI%20Model%20Training.png)

---

## 🔧 1. Hyperparameter Tuning
Hyperparameters are settings that control the learning process. Unlike model parameters, hyperparameters are set **before** training.

### 🔹 Common Hyperparameters:
- 🔥 **Learning Rate (α)**: Determines how much the model updates weights.
- 📦 **Batch Size**: Number of samples processed before updating weights.
- 🏗 **Number of Layers & Neurons**: Controls the depth and complexity of the model.
- 🛑 **Dropout Rate**: Prevents overfitting by randomly deactivating neurons.

### 🎯 Methods for Hyperparameter Tuning:
1️⃣ **Grid Search**: Tries all possible combinations.  
2️⃣ **Random Search**: Samples random values for each hyperparameter.  
3️⃣ **Bayesian Optimization**: Uses probability models to find optimal values.  

**Example: Using Grid Search for Tuning Hyperparameters** 🔍
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

X, y = [[0, 1], [1, 1], [2, 2], [3, 3]], [0, 1, 1, 0]

model = RandomForestClassifier()
param_grid = {'n_estimators': [10, 50, 100], 'max_depth': [None, 5, 10]}

grid_search = GridSearchCV(model, param_grid, cv=3)
grid_search.fit(X, y)
print("🎯 Best Parameters:", grid_search.best_params_)
```

---

## 📉 2. Loss Functions & Optimizers
Loss functions measure **how well a model is performing**, while optimizers adjust the model’s weights to **minimize the loss**.

### ⚖️ 2.1 Common Loss Functions
- 🔹 **Regression**: Mean Squared Error (MSE), Mean Absolute Error (MAE)
- 🔹 **Classification**: Cross-Entropy Loss (for multi-class), Hinge Loss (for SVM)

**Example: Using Cross-Entropy Loss in PyTorch** 🎯
```python
import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
output = torch.tensor([[2.0, 1.0, 0.1]])  # Logits
labels = torch.tensor([0])  # True class
loss = criterion(output, labels)
print("💡 Loss:", loss.item())
```

### 🔄 2.2 Optimizers
Optimizers adjust model parameters to **minimize loss**. Popular choices include:
- 🎯 **SGD (Stochastic Gradient Descent)**: Updates weights using small batches.
- 🚀 **Adam (Adaptive Moment Estimation)**: Combines momentum and adaptive learning rates.

**Example: Training a Model with Adam Optimizer** ⚙️
```python
import torch.optim as optim

model = nn.Linear(2, 1)  # Simple Linear Model
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Dummy training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## 🔁 3. Transfer Learning & Pretrained Models
Transfer learning leverages **pretrained models** on large datasets to improve performance on new tasks with limited data. 🏆

### 🛠 Steps for Transfer Learning:
1️⃣ Load a pretrained model (e.g., **ResNet, VGG, BERT**).  
2️⃣ **Freeze** initial layers to retain learned features.  
3️⃣ **Fine-tune** the last layers for the specific task.  

**Example: Using a Pretrained ResNet Model for Image Classification** 🖼️
```python
import torch
import torchvision.models as models

# Load pretrained model
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer
model.fc = torch.nn.Linear(model.fc.in_features, 2)  # For binary classification
print(model)
```

---

## 🏆 Conclusion
Proper **hyperparameter tuning**, selecting the right **loss function** and **optimizer**, and utilizing **transfer learning** can significantly enhance **model performance** and reduce **training time**. ⚡

📖 **[Back to Main README](../README.md)**
