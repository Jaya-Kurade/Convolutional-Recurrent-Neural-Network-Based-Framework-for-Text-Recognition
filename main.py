import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.utils import save_image

import pandas as pd
from PIL import Image
import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import string

import config

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
import scipy.io as sio
import csv

def preprocess_dataset(max_images=None, output_dir=config.PROCESSED_DATA_DIR / 'words'):
    """Extract word images from SynthText and create labels CSV."""
    
    # Create directories
    images_path = output_dir / 'images'
    images_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'labels.csv'
    
    print(f"Loading dataset from {config.SYNTHTEXT_MAT}")
    data = sio.loadmat(str(config.SYNTHTEXT_MAT))
    
    imnames = data['imnames'][0]
    txt = data['txt'][0]
    wordBB = data['wordBB'][0]
    
    total_images = len(imnames) if max_images is None else min(max_images, len(imnames))
    print(f"Processing {total_images} images...")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'label', 'source_image', 'word_idx'])
        
        word_counter = 0
        
        for img_idx in tqdm(range(total_images), desc="Processing"):
            img_path = config.RAW_DATA_DIR / 'SynthText' / imnames[img_idx][0]
            img = cv2.imread(str(img_path))
            
            if img is None:
                continue
            
            # Split text into words
            text_data = txt[img_idx]
            words = []
            for text in text_data:
                words.extend([w.strip() for w in str(text).replace('\n', ' ').split() if w.strip()])
            
            # Get word bounding boxes
            wbb = wordBB[img_idx]
            num_boxes = wbb.shape[2] if wbb.ndim == 3 else 0
            num_words = min(len(words), num_boxes)
            
            for word_idx in range(num_words):
                corners = wbb[:, :, word_idx]
                x_min = max(0, int(np.floor(corners[0, :].min())))
                y_min = max(0, int(np.floor(corners[1, :].min())))
                x_max = min(img.shape[1], int(np.ceil(corners[0, :].max())))
                y_max = min(img.shape[0], int(np.ceil(corners[1, :].max())))
                
                cropped = img[y_min:y_max, x_min:x_max]
                
                if cropped.size == 0:
                    continue
                
                filename = f"{word_counter:06d}.png"
                cv2.imwrite(str(images_path / filename), cropped)
                
                writer.writerow([filename, words[word_idx], imnames[img_idx][0], word_idx])
                word_counter += 1
    
    print(f"\nDone! Created {word_counter} word images")
    print(f"Output: {output_dir}")
    return output_dir

# Run preprocessing
output_dir = preprocess_dataset(max_images=1000)
class OCRDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform):
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row.filename).convert('L')
        img = self.transform(img)
        label = row.label
        return img, label

transform = T.Compose([
    T.Grayscale(),
    T.Resize((32, 128)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])

dataset = OCRDataset(
    csv_path="data/processed/words/labels.csv",
    img_dir="data/processed/words/images",
    transform=transform
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

batch_means, batch_stds = [], []
count = 0

for i, (imgs, labels) in enumerate(tqdm(loader, total=len(loader))):
    batch_means.append(imgs.mean().item())
    batch_stds.append(imgs.std().item())
    count += imgs.size(0)

print(f"\nTotal images processed: {count}")
print(f"Expected total images: {len(dataset)}")
print("Dataset Mean:", sum(batch_means)/len(batch_means))
print("Dataset Std:", sum(batch_stds)/len(batch_stds))

# Visual check
plt.figure(figsize=(12, 6))
for i in range(min(8, len(imgs))):
    plt.subplot(2, 4, i+1)
    plt.imshow(imgs[i][0].numpy(), cmap='gray')
    plt.title(labels[i])
    plt.axis('off')
plt.show()

print("\nSaving transformed dataset to disk...")

# Create dir
output_dir = Path("data/processed/words/final_images")
output_dir.mkdir(parents=True, exist_ok=True)
csv_path = "data/processed/words/final_labels.csv"

# Write all transformed samples
with open(csv_path, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])

    for i, (img, label) in enumerate(tqdm(dataset, total=len(dataset))):
        filename = f"{i:06}.png"
        unnorm = img.clone()
        unnorm = unnorm * 0.5 + 0.5   
        unnorm = torch.clamp(unnorm, 0, 1)
        
        save_image(unnorm, output_dir / filename)
        writer.writerow([filename, label])

print(f"\nSaved {len(dataset)} images to {output_dir}")
print(f"Labels written to {csv_path}")

df = pd.read_csv("data/processed/words/final_labels.csv")
print("Saved:", len(df), "entries")
print(df.head())

img = Image.open("data/processed/words/final_images/000000.png")
plt.imshow(img, cmap='gray')
plt.title(df.iloc[0].label)
plt.show()
