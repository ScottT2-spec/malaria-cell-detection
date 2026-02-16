# Malaria Cell Detection — Train + Export to ONNX
# Run this on Kaggle with GPU enabled
# Dataset: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

# ─── Dataset path on Kaggle ───
DATASET_PATH = '/kaggle/input/cell-images-for-detecting-malaria/cell_images/cell_images'
# If that doesn't work, try:
if not os.path.exists(DATASET_PATH):
    DATASET_PATH = '/kaggle/input/cell-images-for-detecting-malaria/cell_images'
if not os.path.exists(DATASET_PATH):
    # List what's available
    import glob
    print("Available paths:")
    for p in glob.glob('/kaggle/input/**/*', recursive=True)[:20]:
        print(p)

IMG_SIZE = 128
BATCH_SIZE = 32

# ─── Data ───
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
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

print(f"Classes: {train_gen.class_indices}")
print(f"Train: {train_gen.samples}, Val: {val_gen.samples}")

# ─── Model ───
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
out = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=out)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ─── Phase 1: Train head ───
print("\n=== Phase 1: Training head ===")
model.fit(train_gen, epochs=8, validation_data=val_gen,
          callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

# ─── Phase 2: Fine-tune ───
print("\n=== Phase 2: Fine-tuning ===")
base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_gen, epochs=8, validation_data=val_gen,
          callbacks=[EarlyStopping(patience=3, restore_best_weights=True),
                     ReduceLROnPlateau(patience=2, factor=0.5)])

# ─── Evaluate ───
loss, acc = model.evaluate(val_gen)
print(f"\nValidation Accuracy: {acc:.4f}")

# ─── Save Keras model ───
model.save('malaria_model.keras')
print("Saved malaria_model.keras")

# ─── Export to ONNX ───
print("\nExporting to ONNX...")
import subprocess
subprocess.run(['pip', 'install', 'tf2onnx', '-q'])

import tf2onnx
import onnxruntime

spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input"),)
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path="malaria_model.onnx")
print("Saved malaria_model.onnx")

# Verify ONNX model
sess = onnxruntime.InferenceSession("malaria_model.onnx")
dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)
result = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(f"ONNX test output: {result[0][0]}")
print(f"\nClasses: {train_gen.class_indices}")
print("0 = Parasitized, 1 = Uninfected (output is sigmoid probability of class 1)")

print("\n✅ DONE! Download malaria_model.onnx from the Output tab")
print("Upload it to your server in the web-app folder")
