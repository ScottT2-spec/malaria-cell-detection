# Malaria Cell Detection

CNN that classifies red blood cells as **Parasitized** (malaria-infected) or **Uninfected** using microscope images from the NIH dataset.

**Accuracy:** 95.43% on held-out test data

## What It Does

Takes a 64x64 microscope image of a blood cell and predicts whether the Plasmodium parasite is present. Trained on ~27,500 images (balanced classes) from the National Institutes of Health.

## Model

3-block CNN built with TensorFlow/Keras:

- **Block 1:** 32 filters (3x3) → ReLU → MaxPool — picks up edges and color spots
- **Block 2:** 64 filters (3x3) → ReLU → MaxPool — combines into shapes
- **Block 3:** 128 filters (3x3) → ReLU → MaxPool — recognizes parasite-like structures

Then a dense layer (128 units) with 50% dropout, and a sigmoid output for binary classification.

Trained for 10 epochs with Adam optimizer and binary cross-entropy loss. 80/20 train-test split, stratified to keep class balance.

## Results

- **Test accuracy:** 95.43%
- Generates `training_history.png` (accuracy/loss curves) and `predictions.png` (sample predictions with color-coded correctness)

## Usage

Run on [Kaggle](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) with GPU enabled:

```bash
python malaria_detection.py
```

It auto-detects the Kaggle dataset path. For local use, put the `cell_images/` folder (with `Parasitized/` and `Uninfected/` subfolders) in the same directory.

## Dataset

[NIH Malaria Cell Images](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
- 27,558 images total
- 13,779 Parasitized + 13,779 Uninfected
- Thin blood smear slides, stained and photographed under microscope

## What I Learned

- How CNNs extract features hierarchically (edges → shapes → complex patterns)
- Why stratified splits matter for balanced evaluation
- The effect of dropout on reducing overfitting in small-ish datasets
- Image preprocessing and normalization for neural networks

## Author

Scott Antwi — [GitHub](https://github.com/ScottT2-spec) · [Kaggle](https://kaggle.com/scottantwi)
