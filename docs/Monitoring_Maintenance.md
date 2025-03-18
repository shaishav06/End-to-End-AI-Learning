# Monitoring & Maintenance

Once a machine learning model is deployed, continuous monitoring and maintenance are essential to ensure it remains accurate and reliable over time. This guide covers **model drift & retraining, logging & performance metrics, and MLOps for CI/CD pipelines**.

---

## 1. Model Drift & Retraining
### 1.1 What is Model Drift?
Model drift occurs when the model's performance degrades over time due to changes in data distribution.
- **Concept Drift**: The relationship between input features and the target variable changes.
- **Data Drift**: The input data distribution shifts without affecting the relationship.

### 1.2 Detecting Model Drift
**Example: Checking Data Drift with SciPy KS-Test**
```python
from scipy.stats import ks_2samp
import numpy as np

# Historical data
distribution_old = np.random.normal(50, 10, 1000)
# New incoming data
distribution_new = np.random.normal(55, 12, 1000)

# Perform KS test
stat, p_value = ks_2samp(distribution_old, distribution_new)
if p_value < 0.05:
    print("Significant data drift detected!")
else:
    print("No significant data drift detected.")
```

### 1.3 Retraining the Model
Once drift is detected, retraining the model with updated data is necessary.
```python
# Load new training data
X_new, y_new = load_new_data()

# Retrain model
model.fit(X_new, y_new)

# Save updated model
import pickle
with open("model_updated.pkl", "wb") as file:
    pickle.dump(model, file)
```

---

## 2. Logging & Performance Metrics
Logging helps track model predictions, errors, and system performance.

### 2.1 Implementing Logging with Python
```python
import logging

logging.basicConfig(filename="model.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

def predict(input_data):
    try:
        prediction = model.predict([input_data])
        logging.info(f"Input: {input_data}, Prediction: {prediction}")
        return prediction
    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return None
```

### 2.2 Monitoring Model Performance with Prometheus & Grafana
1. **Install Prometheus & Grafana**: 
   ```bash
   sudo apt install prometheus grafana
   ```
2. **Expose Model Metrics in FastAPI**:
   ```python
   from prometheus_client import Counter, generate_latest
   from fastapi import FastAPI
   
   app = FastAPI()
   prediction_counter = Counter("model_predictions", "Number of model predictions")
   
   @app.get("/metrics")
   def get_metrics():
       return generate_latest()
   ```
3. **Visualize in Grafana**: Connect Prometheus to Grafana and create dashboards for monitoring.

---

## 3. MLOps for CI/CD Pipelines
MLOps automates model training, validation, deployment, and monitoring.

### 3.1 Automating Model Retraining with GitHub Actions
**Example: CI/CD Workflow for Model Training**
```yaml
name: Retrain Model
on:
  schedule:
    - cron: "0 0 * * 1"  # Runs every Monday
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.8"
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Train model
        run: python train.py
      - name: Save model
        run: mv model.pkl models/
      - name: Commit new model
        run: |
          git add models/model.pkl
          git commit -m "Updated model"
          git push
```

### 3.2 Deploying New Model Versions with Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-deployment
spec:
  replicas: 2
  template:
    spec:
      containers:
      - name: model-container
        image: myregistry/model:v2
        ports:
        - containerPort: 5000
```
**Deploy Updated Model:**
```bash
kubectl apply -f deployment.yaml
```

---

## Conclusion
- **Model Drift & Retraining**: Detect drift and retrain models regularly.
- **Logging & Performance Metrics**: Track predictions and monitor system performance.
- **MLOps for CI/CD Pipelines**: Automate retraining and deployment with CI/CD.

Continuous monitoring ensures that AI models remain reliable and effective over time.

---
### [Back to Main README](../README.md)
