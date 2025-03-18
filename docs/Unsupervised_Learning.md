# 🤖 Unsupervised Learning

Unsupervised Learning is a type of machine learning where the model learns patterns and structures from unlabeled data. Unlike supervised learning, there are no predefined labels, and the algorithm finds hidden structures in the data on its own.

---

## 🔍 1. Clustering
Clustering is a technique used to group similar data points into clusters based on similarity.

### 📌 1.1 K-Means Clustering
K-Means is a centroid-based clustering algorithm that partitions data into K clusters.

**🔢 Steps:**
1️⃣ Select K (number of clusters).  
2️⃣ Randomly initialize K cluster centroids.  
3️⃣ Assign each data point to the nearest centroid.  
4️⃣ Update centroids based on assigned points.  
5️⃣ Repeat until centroids stabilize.  

**Example:** Customer segmentation for marketing.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample data
X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='X', c='red')
plt.show()
```

### 🏠 1.2 DBSCAN (Density-Based Clustering)
DBSCAN groups points based on density rather than centroids, making it effective for irregular shapes and noise handling.

**Example:** Anomaly detection in network traffic.
```python
from sklearn.cluster import DBSCAN
import numpy as np

# Sample data
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])

# Apply DBSCAN
dbscan = DBSCAN(eps=2, min_samples=2)
labels = dbscan.fit_predict(X)
print("🚀 Cluster Labels:", labels)
```

### 🌲 1.3 Hierarchical Clustering
Hierarchical clustering builds a hierarchy of clusters using either:
- **Agglomerative (Bottom-Up):** Merges small clusters into larger ones.
- **Divisive (Top-Down):** Splits large clusters into smaller ones.

**Example:** Document classification based on text similarity.
```python
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import numpy as np

# Sample data
X = np.array([[5, 3], [10, 15], [24, 10], [30, 30], [85, 70], [71, 80]])

# Dendrogram
plt.figure(figsize=(8, 5))
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()
```

---

## 🎭 2. Dimensionality Reduction
Dimensionality Reduction techniques reduce the number of features while preserving meaningful information.

### 📉 2.1 Principal Component Analysis (PCA)
PCA transforms data into a lower-dimensional space while retaining the most variance.

**Example:** Reducing image dataset dimensions.
```python
from sklearn.decomposition import PCA
import numpy as np

# Sample data
X = np.array([[2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0]])

# Apply PCA
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(X)
print("📊 Reduced Data:", X_reduced)
```

### 🎨 2.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE is a nonlinear dimensionality reduction technique that maps high-dimensional data into 2D or 3D for visualization.

**Example:** Visualizing handwritten digits (MNIST dataset).
```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# Load dataset
digits = load_digits()
X, y = digits.data, digits.target

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

# Plot results
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap='jet')
plt.colorbar()
plt.show()
```

---

## 🏆 Conclusion
Unsupervised learning techniques like clustering and dimensionality reduction are powerful tools for finding patterns, segmenting data, and reducing complexity without labeled data.

📖 **[Back to Main README](../README.md)**
