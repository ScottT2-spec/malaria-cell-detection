"""
Malaria Cell Detection — CNN Model
Built by Scott Antwi

Detects malaria-infected red blood cells from microscope images
using a Convolutional Neural Network. Trained on the NIH dataset.

Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
Accuracy: 95.43% on unseen test data
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


# CONFIGURATION

IMG_SIZE = 64
EPOCHS = 10
BATCH_SIZE = 32
TEST_SIZE = 0.2
VAL_SPLIT = 0.15
RANDOM_STATE = 42


# DATA LOADING

# Find the dataset path (works on Kaggle)
DATA_PATH = None
for root, dirs, files in os.walk('/kaggle/input/'):
    if 'Parasitized' in dirs:
        DATA_PATH = root + '/'
        break

# Fallback for local use
if DATA_PATH is None:
    DATA_PATH = './cell_images/'

print(f'Data path: {DATA_PATH}')

def load_images(folder, label):
    """Load images from a folder, resize, and assign label."""
    images, labels = [], []
    for filename in os.listdir(folder):
        try:
            img = Image.open(os.path.join(folder, filename))
            img = img.resize((IMG_SIZE, IMG_SIZE))
            images.append(np.array(img))
            labels.append(label)
        except:
            pass  # Skip corrupted images
    return images, labels

print('Loading infected cells...')
X_infected, y_infected = load_images(DATA_PATH + 'Parasitized/', 1)
print(f'  Loaded {len(X_infected)} infected images')

print('Loading healthy cells...')
X_healthy, y_healthy = load_images(DATA_PATH + 'Uninfected/', 0)
print(f'  Loaded {len(X_healthy)} healthy images')

# Combine and normalize
X = np.array(X_infected + X_healthy, dtype='float32') / 255.0
y = np.array(y_infected + y_healthy)

# Free memory
del X_infected, X_healthy, y_infected, y_healthy

print(f'\nTotal images: {len(X)}')
print(f'Image shape: {X[0].shape}')
print(f'Infected: {sum(y)} | Healthy: {len(y) - sum(y)}')


# TRAIN/TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y  # Keep class balance in both sets
)

print(f'\nTrain: {X_train.shape[0]} images')
print(f'Test:  {X_test.shape[0]} images')


# MODEL ARCHITECTURE

model = models.Sequential([
    # Block 1: Detect basic patterns (edges, spots)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),

    # Block 2: Combine into shapes
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Block 3: Recognize complex features (parasite shapes)
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Classification head
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Prevent overfitting
    layers.Dense(1, activation='sigmoid')  # Binary output: 0=healthy, 1=infected
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# TRAINING
print('\nTraining...')
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=VAL_SPLIT,
    verbose=1
)

# EVALUATION
loss, accuracy = model.evaluate(X_test, y_test)
print(f'\nTest Accuracy: {round(accuracy * 100, 2)}%')
print(f'Test Loss: {round(loss, 4)}')

# VISUALIZATION

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['accuracy'], label='Train')
ax1.plot(history.history['val_accuracy'], label='Validation')
ax1.set_title('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['loss'], label='Train')
ax2.plot(history.history['val_loss'], label='Validation')
ax2.set_title('Loss')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig('training_history.png', dpi=150)
plt.show()

# Show sample predictions
predictions = model.predict(X_test[:10])
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i in range(10):
    row, col = i // 5, i % 5
    axes[row][col].imshow(X_test[i])
    pred = 'INFECTED' if predictions[i] > 0.5 else 'HEALTHY'
    actual = 'INFECTED' if y_test[i] == 1 else 'HEALTHY'
    color = 'green' if pred == actual else 'red'
    axes[row][col].set_title(pred, color=color)
    axes[row][col].axis('off')

plt.suptitle('Predictions (Green=Correct, Red=Wrong)')
plt.tight_layout()
plt.savefig('predictions.png', dpi=150)
plt.show()
