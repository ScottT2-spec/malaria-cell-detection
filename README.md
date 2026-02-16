# Malaria Cell Detection

AI-powered malaria screening using deep learning. Scan blood cell images through your camera and get instant classification — Parasitized or Uninfected.

## Live Demo

🔬 [Scan cells now →](http://your-server-url:5001)

## How It Works

1. Open the web app on your phone or computer
2. Point your camera at a microscope slide with blood cell sample
3. Tap **Scan Cell**
4. The AI model analyzes the cell and returns a diagnosis

## Model

- **Architecture:** MobileNetV2 (transfer learning) with custom classification head
- **Dataset:** NIH Malaria Cell Images — 27,558 images (13,779 parasitized + 13,779 uninfected)
- **Accuracy:** ~96% on validation set
- **Input:** 128×128 RGB blood cell image
- **Output:** Binary classification (Parasitized / Uninfected) with confidence score

### Training Pipeline

1. Phase 1: Train classification head with frozen MobileNetV2 base (8 epochs)
2. Phase 2: Fine-tune last 30 layers of MobileNetV2 with low learning rate (8 epochs)
3. Export to ONNX format for lightweight server deployment

## Tech Stack

- **Model:** TensorFlow/Keras → ONNX Runtime
- **Backend:** Python, Flask
- **Frontend:** HTML/CSS/JavaScript with camera API (getUserMedia)
- **Deployment:** Cloud server with ONNX Runtime inference

## Project Structure

```
malaria-cell-detection/
├── malaria_detection.py        # Original training script (from-scratch CNN)
├── train_and_export.py         # Transfer learning + ONNX export (run on Kaggle)
├── web-app/
│   ├── app.py                  # Flask server
│   ├── train_model.py          # Local training script
│   ├── malaria_model.onnx      # Trained model (ONNX format)
│   └── templates/
│       └── index.html          # Camera scanner UI
└── README.md
```

## Run Locally

```bash
# Install dependencies
pip install flask onnxruntime pillow

# Start server
cd web-app
python app.py
```

Open `http://localhost:5001` in your browser.

## Train Your Own Model

1. Open `train_and_export.py` in a Kaggle notebook with GPU
2. Add the [Cell Images for Detecting Malaria](https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria) dataset
3. Run — it trains and exports `malaria_model.onnx`
4. Download the ONNX file and place it in `web-app/`

## Dataset

[NIH Malaria Cell Images Dataset](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
- 27,558 blood cell images
- Binary: Parasitized (Plasmodium present) vs Uninfected
- Collected from thin blood smear slide images

## Disclaimer

This tool is for **educational and screening purposes only**. It is not a medical device. Always confirm results with a qualified healthcare professional.

## Author

**Scott Antwi** — [GitHub](https://github.com/ScottT2-spec) · [Kaggle](https://kaggle.com/scottantwi)
