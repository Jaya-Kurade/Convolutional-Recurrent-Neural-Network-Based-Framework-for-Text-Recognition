# CRNN-based OCR with SynthText Dataset

ML-based OCR using CRNN (Convolutional Recurrent Neural Network) trained on the SynthText dataset.

## Project Structure

```
.
├── data/
│   ├── raw/              # SynthText dataset
│   └── processed/
│       └── words/        # Preprocessed word images
│           ├── images/   # Individual word crops (PNG)
│           └── labels.csv # Word labels and metadata
├── models/               # Model architecture files
├── utils/                # Helper functions
├── config.py            # Configuration and hyperparameters
├── main.ipynb           # Main training notebook
└── requirements.txt     # Dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download SynthText dataset (~41GB):
```bash
python utils/download_synthtext.py
```

3. Run the notebook:
```bash
jupyter notebook main.ipynb
```

Dataset source: https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip

## Dataset

- **SynthText**: Synthetic text images from Visual Geometry Group (VGG), Oxford
- Format: MATLAB .mat files with images, text annotations, and bounding boxes
- Preprocessing: Extracts individual word images using word-level bounding boxes
- Output: Word images (PNG) with corresponding text labels in CSV format

## Configuration

Edit `config.py` to adjust:
- Image dimensions (default: 32x128)
- Batch size, learning rate, epochs
- Model architecture parameters
- Character vocabulary

## Workflow

1. Run `main.ipynb` to train the model
2. Preprocessing extracts word images from raw dataset
3. Dataset loader applies transforms and encodes labels
4. CRNN model trained with CTC loss
5. Evaluate and run inference on new images

## Progress

- [x] Dataset download and preprocessing
- [ ] Dataset loader and transforms
- [ ] CRNN model architecture
- [ ] Training loop with CTC loss
- [ ] Inference and evaluation

## Architecture

**CRNN Components:**
- CNN: Feature extraction from images
- RNN: Bidirectional LSTM for sequence modeling
- CTC: Connectionist Temporal Classification for alignment-free training
