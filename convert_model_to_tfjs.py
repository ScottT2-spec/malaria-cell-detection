"""
Run this in Google Colab AFTER training your malaria model.
It converts the Keras model to TensorFlow.js format and downloads it.

Steps:
1. Train your model (run malaria_detection.py first)
2. Run this script in the same Colab session
3. Upload the output files to your GitHub repo under docs/model/
"""

# Step 1: Install tensorflowjs converter
!pip install tensorflowjs

import tensorflowjs as tfjs

# Step 2: Save the trained model (if not already saved)
# If your model variable is called 'model':
model.save('malaria_model.h5')
print("✅ Keras model saved")

# Step 3: Convert to TensorFlow.js format
tfjs.converters.save_keras_model(model, 'tfjs_model')
print("✅ Model converted to TF.js format")

# Step 4: Zip and download
import shutil
shutil.make_archive('tfjs_model', 'zip', 'tfjs_model')

from google.colab import files
files.download('tfjs_model.zip')
print("📦 Download started! Unzip and upload contents to docs/model/ in your GitHub repo")
