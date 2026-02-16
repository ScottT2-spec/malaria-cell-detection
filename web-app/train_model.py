"""
Malaria Cell Detection — Model Training
Uses MobileNetV2 transfer learning for fast, accurate training
Downloads NIH malaria cell dataset automatically
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import zipfile
import urllib.request

# ─── Download Dataset ───
DATA_DIR = '/tmp/malaria_data'
ZIP_PATH = '/tmp/cell_images.zip'

if not os.path.exists(os.path.join(DATA_DIR, 'cell_images')):
    print("Downloading malaria cell dataset...")
    url = "https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip"
    urllib.request.urlretrieve(url, ZIP_PATH)
    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as z:
        z.extractall(DATA_DIR)
    print("Done!")
else:
    print("Dataset already exists")

DATASET_PATH = os.path.join(DATA_DIR, 'cell_images')
IMG_SIZE = 128
BATCH_SIZE = 32

# ─── Data Generators ───
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"Classes: {train_gen.class_indices}")
print(f"Training samples: {train_gen.samples}")
print(f"Validation samples: {val_gen.samples}")

# ─── Build Model (MobileNetV2 Transfer Learning) ───
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # Freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── Train ───
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(patience=3, factor=0.5, monitor='val_loss')
]

print("\n=== Phase 1: Train head only ===")
history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks
)

# Fine-tune last 30 layers
print("\n=== Phase 2: Fine-tune ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen,
    callbacks=callbacks
)

# ─── Evaluate ───
val_loss, val_acc = model.evaluate(val_gen)
print(f"\nFinal Validation Accuracy: {val_acc:.4f}")

# ─── Save ───
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'malaria_model.keras')
model.save(MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# Also save class indices
import json
with open(os.path.join(os.path.dirname(__file__), 'class_indices.json'), 'w') as f:
    json.dump(train_gen.class_indices, f)
print(f"Classes: {train_gen.class_indices}")
