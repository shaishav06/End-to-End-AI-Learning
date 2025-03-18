# Deep Learning

Deep Learning is a subset of Machine Learning that uses neural networks with multiple layers to extract patterns and representations from data. It is particularly powerful in areas like image processing, natural language processing, and time-series forecasting.

---

## 1. Convolutional Neural Networks (CNN) for Image Processing
CNNs are specialized neural networks designed for processing structured grid data, such as images. They use convolutional layers to detect features like edges, textures, and shapes.

### CNN Architecture:
- **Convolutional Layers**: Extract spatial features.
- **Pooling Layers**: Reduce dimensionality and computation.
- **Fully Connected Layers**: Map features to final predictions.

**Example: CNN for Image Classification (MNIST Handwritten Digits)**
```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load and preprocess data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
```

---

## 2. Recurrent Neural Networks (RNN) & LSTM for Time-Series & NLP
RNNs are designed for sequential data, making them useful for tasks like time-series forecasting and natural language processing. However, standard RNNs suffer from vanishing gradients, which LSTMs solve.

### LSTM Architecture:
- **Forget Gate**: Decides what information to discard.
- **Input Gate**: Decides what new information to store.
- **Output Gate**: Decides what information to pass forward.

**Example: LSTM for Stock Price Prediction**
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Sample data (random time-series)
X_train = np.random.rand(100, 10, 1)  # (samples, timesteps, features)
y_train = np.random.rand(100, 1)

# Build LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(10, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

# Compile and train model
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10)
```

---

## 3. Transformer Models (BERT, GPT)
Transformers are state-of-the-art architectures for NLP tasks. They use **self-attention mechanisms** to capture long-range dependencies in text.

### 3.1 BERT (Bidirectional Encoder Representations from Transformers)
BERT is designed for **bidirectional language understanding**. It is used for tasks like text classification and question answering.

**Example: Using BERT for Sentiment Analysis**
```python
from transformers import pipeline

# Load pre-trained BERT model
classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)
```

### 3.2 GPT (Generative Pre-trained Transformer)
GPT is a generative model used for text generation and completion.

**Example: Using GPT for Text Generation**
```python
from transformers import pipeline

# Load GPT model
generator = pipeline("text-generation", model="gpt2")
result = generator("Once upon a time, there was a curious AI", max_length=50)
print(result)
```

---

## Conclusion
Deep Learning enables powerful AI applications across domains. CNNs excel in image processing, LSTMs handle sequential data, and Transformers revolutionize NLP tasks.

---
### [Back to Main README](../README.md)
