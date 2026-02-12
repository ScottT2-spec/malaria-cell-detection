# 🦟 Malaria Cell Detection — CNN with 95.43% Accuracy

A convolutional neural network that detects malaria-infected red blood cells from microscope images. Trained on the NIH Malaria Cell Images dataset (27,558 images).

![Python](https://img.shields.io/badge/Python-3.12-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange) ![Accuracy](https://img.shields.io/badge/Test%20Accuracy-95.43%25-green)

## Why I Built This

Malaria kills over 600,000 people every year, mostly in Sub-Saharan Africa. Diagnosis requires a trained technician to manually examine blood smear slides under a microscope — it's slow and error-prone, especially in rural clinics with limited staff.

I wanted to see if a CNN could automate this screening process. The answer: yes, with 95.43% accuracy on unseen test data.

## The Data

- **Source:** [NIH Malaria Cell Images Dataset](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) (National Institutes of Health)
- **Total images:** 27,558
- **Classes:** Parasitized (infected) vs Uninfected — perfectly balanced (13,779 each)
- **Image type:** Microscope photos of individual red blood cells

### What the AI sees:

| Infected (Parasitized) | Healthy (Uninfected) |
|---|---|
| Dark purple spots visible inside the cell — these are the Plasmodium parasite | Clean, uniform pink/red color with no dark spots |

## How It Works

### Preprocessing
1. Resized all images to 64×64 pixels
2. Normalized pixel values to [0, 1] range (divide by 255)
3. Stratified train/test split: 80% training (22,046), 20% test (5,512)

### Model Architecture

```
Conv2D(32, 3×3, relu)     → Detects basic patterns (edges, spots)
MaxPooling2D(2×2)          → Reduces size, keeps important features
Conv2D(64, 3×3, relu)     → Combines patterns into shapes
MaxPooling2D(2×2)
Conv2D(128, 3×3, relu)    → Recognizes complex features (parasite shapes)
MaxPooling2D(2×2)
Flatten()                  → Converts 2D features to 1D
Dense(128, relu)           → Weighs all features together
Dropout(0.5)               → Prevents overfitting (randomly disables 50% of neurons)
Dense(1, sigmoid)          → Final prediction: 0 = healthy, 1 = infected
```

**Total parameters:** 3,304,769

### Training
- **Optimizer:** Adam
- **Loss function:** Binary cross-entropy
- **Epochs:** 10
- **Batch size:** 32
- **Validation split:** 15% of training data

## Results

| Metric | Score |
|---|---|
| Training Accuracy | 97.12% |
| Validation Accuracy | 95.74% |
| **Test Accuracy** | **95.43%** |

The model generalizes well — the small gap between training and test accuracy shows it learned actual patterns, not just memorization.

## What I Learned

- **CNNs detect spatial patterns** — convolutional filters slide across images finding edges, textures, and shapes. Regular neural networks can't do this because they flatten the image and lose all spatial information.
- **Normalization matters** — dividing by 255 keeps values small so gradients don't explode during backpropagation.
- **Stratified splitting** — ensures both train and test sets maintain the same class distribution, preventing biased evaluation.
- **Dropout prevents overfitting** — randomly disabling neurons forces the network to learn robust features instead of memorizing specific training examples.
- **MaxPooling gives translation invariance** — a parasite spot detected in the corner or center still gets recognized.

## Run It Yourself

**On Kaggle (recommended):**
- [View the notebook](https://www.kaggle.com/scottantwi) — click "Copy & Edit" to run it

**Locally:**
```bash
pip install tensorflow numpy pillow scikit-learn matplotlib
python malaria_detection.py
```

## Future Improvements

- Data augmentation (rotation, flip, zoom) to improve generalization
- Transfer learning with a pretrained model (ResNet, EfficientNet)
- Grad-CAM visualization to show exactly WHERE the model is looking
- Deploy as a web app for real clinic use

## Dataset Citation

Rajaraman et al., "Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection in thin blood smear images." PeerJ, 2018.

---

Built by [Scott Antwi](https://github.com/ScottT2-spec) · 17 y/o · Ghana 🇬🇭
