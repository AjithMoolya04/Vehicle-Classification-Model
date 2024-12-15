# Vehicle-Classification-Model

## Vehicle Classification Using CNN

### Overview

This project involves building a Convolutional Neural Network (CNN) to classify vehicle types based on images. Using a dataset of various vehicle categories, the model aims to achieve high accuracy in distinguishing between different classes, aiding in automated vehicle recognition systems.

### Dataset

- **Source**: [Specify Source]
- **Structure**: Training and testing datasets containing vehicle images in the following categories:
  - Cars
  - Trucks
  - Motorcycles
  - Buses

### Model Architecture

- **Base Model**: Custom CNN with convolutional and pooling layers.
- **Output Layer**: Softmax activation for multi-class classification.

```python
# Import necessary libraries
from tensorflow.keras import models, layers

# Define the model architecture
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Use 'num_classes' for flexibility
    ])
    return model
```

# Compile the model
```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
