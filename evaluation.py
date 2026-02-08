# Enable MPS fallback for CTCLoss
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
from pathlib import Path
import glob
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd

import config
from crnn import build_crnn  # your CRNN definition

def load_labels_from_csv(csv_path: Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    # expects columns: filename, label
    return dict(zip(df["filename"], df["label"]))


def encode_label(text: str) -> List[int]:
    """
    Encode text into a list of indices using config.CHAR_TO_IDX.
    Unknown characters are skipped.
    """
    indices = []
    for ch in text:
        if ch in config.CHAR_TO_IDX:
            indices.append(config.CHAR_TO_IDX[ch])
        else:
            # skip char if not in vocabulary
            pass
    return indices


class WordImageDataset(Dataset):
    def __init__(self, images_dir: Path, label_dict: Dict[str, str]):
        super().__init__()
        self.images_dir = Path(images_dir)
        self.image_paths = sorted(glob.glob(str(self.images_dir / "*.png")))
        self.label_dict = label_dict

        self.transform = transforms.Compose([
            transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
            transforms.ToTensor(),  # [1, H, W] since we convert to "L"
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = Path(self.image_paths[idx])
        img = Image.open(img_path).convert("L")  # grayscale

        img_tensor = self.transform(img)  # [1, H, W]

        filename = img_path.name
        if filename not in self.label_dict:
            raise KeyError(f"No label found for image {filename}")

        text_label = self.label_dict[filename]
        label_indices = encode_label(text_label)
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        return img_tensor, label_tensor, text_label, str(img_path)
      def ctc_collate_fn(batch):
    """
    batch: list of (img_tensor, label_tensor, text_label, img_path)

    Returns:
        images: [B, C, H, W]
        labels: 1D concatenated labels
        target_lengths: [B]
        texts: list of strings
        paths: list of paths
    """
    imgs, label_tensors, texts, paths = zip(*batch)

    images = torch.stack(imgs, dim=0)  # [B, C, H, W]
    target_lengths = torch.tensor([len(l) for l in label_tensors], dtype=torch.long)
    if len(label_tensors) > 0:
        labels = torch.cat(label_tensors, dim=0)
    else:
        labels = torch.empty(0, dtype=torch.long)

    return images, labels, target_lengths, list(texts), list(paths)


def save_split(indices, path: Path):
    with open(path, "w") as f:
        json.dump(indices, f)


def load_split(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def create_datasets_and_loaders():
    images_dir = config.PROCESSED_DATA_DIR / "words" / "final_images"
    labels_csv = config.PROCESSED_DATA_DIR / "words" / "labels.csv"

    label_dict = load_labels_from_csv(labels_csv)
    full_dataset = WordImageDataset(images_dir, label_dict)

    n_total = len(full_dataset)
    n_train = int(config.TRAIN_SPLIT * n_total)
    n_val   = int(config.VAL_SPLIT * n_total)
    n_test  = n_total - n_train - n_val

    split_dir = config.PROCESSED_DATA_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    train_path = split_dir / "train_indices.json"
    val_path   = split_dir / "val_indices.json"
    test_path  = split_dir / "test_indices.json"

    if train_path.exists():
        print("Loading saved dataset splits...")
        train_indices = load_split(train_path)
        val_indices   = load_split(val_path)
        test_indices  = load_split(test_path)
    else:
        print("WARNING: split files not found, creating NEW splits (may not match training).")
        g = torch.Generator().manual_seed(config.RANDOM_SEED)
        train_ds_tmp, val_ds_tmp, test_ds_tmp = random_split(
            full_dataset,
            [n_train, n_val, n_test],
            generator=g
        )
        train_indices = train_ds_tmp.indices
        val_indices   = val_ds_tmp.indices
        test_indices  = test_ds_tmp.indices

        save_split(train_indices, train_path)
        save_split(val_indices, val_path)
        save_split(test_indices, test_path)

    # Build Subset objects from saved indices
    train_ds = Subset(full_dataset, train_indices)
    val_ds   = Subset(full_dataset, val_indices)
    test_ds  = Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=ctc_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=ctc_collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False,
                              collate_fn=ctc_collate_fn, num_workers=0)

    print(f"Total: {n_total}, Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
    return full_dataset, train_loader, val_loader, test_loader
def greedy_decode(logits: torch.Tensor, blank_token: int = config.BLANK_TOKEN):
    """
    logits: [T, B, C]
    Returns: list of predicted strings, length B
    """
    log_probs = logits.log_softmax(2)       # [T, B, C]
    max_indices = log_probs.argmax(2)       # [T, B]

    batch_texts = []
    for b in range(max_indices.size(1)):
        seq = max_indices[:, b].tolist()

        # CTC collapse (remove repeats and blanks)
        prev = None
        decoded_indices = []
        for idx in seq:
            if idx != blank_token and idx != prev:
                decoded_indices.append(idx)
            prev = idx

        # Map to characters
        chars = []
        for idx in decoded_indices:
            ch = config.IDX_TO_CHAR.get(idx, "")
            chars.append(ch)
        batch_texts.append("".join(chars))

    return batch_texts


def levenshtein(a: str, b: str) -> int:
    """
    Simple Levenshtein distance between two strings.
    """
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]

    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j

    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # delete
                dp[i][j - 1] + 1,      # insert
                dp[i - 1][j - 1] + cost,  # substitute
            )
    return dp[-1][-1]


def compute_cer_and_word_accuracy(gt_texts: List[str], pred_texts: List[str]):
    assert len(gt_texts) == len(pred_texts)
    total_char_dist = 0
    total_chars = 0
    correct_words = 0

    for gt, pred in zip(gt_texts, pred_texts):
        total_char_dist += levenshtein(gt, pred)
        total_chars += len(gt)
        if gt == pred:
            correct_words += 1

    cer = total_char_dist / max(1, total_chars)
    word_acc = correct_words / max(1, len(gt_texts))
    return cer, word_acc
@torch.no_grad()
def evaluate_model(model, data_loader, criterion, device, split_name: str, save_csv: bool = True):
    model.eval()

    all_gt = []
    all_pred = []
    all_paths = []

    running_loss = 0.0
    num_batches = 0

    for images, labels, target_lengths, texts, paths in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        target_lengths = target_lengths.to(device)

        logits = model(images)  # [T, B, C]
        T_seq, B, C = logits.shape

        log_probs = logits.log_softmax(2)
        input_lengths = torch.full(
            size=(B,),
            fill_value=T_seq,
            dtype=torch.long,
            device=device,
        )

        loss = criterion(log_probs, labels, input_lengths, target_lengths)
        running_loss += loss.item()
        num_batches += 1

        pred_texts = greedy_decode(logits.cpu(), blank_token=config.BLANK_TOKEN)

        all_gt.extend(texts)
        all_pred.extend(pred_texts)
        all_paths.extend(paths)

    avg_loss = running_loss / max(1, num_batches)
    cer, word_acc = compute_cer_and_word_accuracy(all_gt, all_pred)

    print(f"[{split_name}] Loss: {avg_loss:.4f} | CER: {cer:.4f} | Word Acc: {word_acc:.4f}")

    if save_csv:
        out_dir = config.PROCESSED_DATA_DIR / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{split_name.lower()}_predictions.csv"

        df = pd.DataFrame({
            "path": all_paths,
            "ground_truth": all_gt,
            "prediction": all_pred,
        })
        df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")

    # Return metrics if you want to log them
    return avg_loss, cer, word_acc
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

full_dataset, train_loader, val_loader, test_loader = create_datasets_and_loaders()

# Build model and load best checkpoint
model = build_crnn().to(device)

best_model_path = config.MODELS_DIR / "crnn_best.pt"
print("Loading best model from:", best_model_path)
state = torch.load(best_model_path, map_location=device)
model.load_state_dict(state)

ctc_loss = nn.CTCLoss(
    blank=config.BLANK_TOKEN,
    zero_infinity=True,
)

# Evaluate on validation and test sets
val_metrics = evaluate_model(model, val_loader, ctc_loss, device, split_name="Val")
test_metrics = evaluate_model(model, test_loader, ctc_loss, device, split_name="Test")
