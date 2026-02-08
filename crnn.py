import torch
import torch.nn as nn
import config

class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        img_height = config.IMG_HEIGHT
        img_width = config.IMG_WIDTH
        in_channels = config.IMG_CHANNELS
        cnn_feature_dim = config.CNN_FEATURE_DIM
        rnn_hidden_size = config.RNN_HIDDEN_SIZE
        rnn_num_layers = config.RNN_NUM_LAYERS
        rnn_bidirectional = config.RNN_BIDIRECTIONAL
        num_classes = config.NUM_CLASSES

        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_height, img_width)
            dummy_out = self.cnn(dummy)
        _, c_out, h_out, w_out = dummy_out.shape
        self._cnn_out_height = h_out
        self._cnn_out_channels = c_out
        self._cnn_out_dim = c_out * h_out

        self.cnn_proj = nn.Linear(self._cnn_out_dim, cnn_feature_dim)

        # bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=cnn_feature_dim,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            bidirectional=rnn_bidirectional,
        )

        rnn_directions = 2 if rnn_bidirectional else 1

        self.fc = nn.Linear(rnn_hidden_size * rnn_directions, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        B, C, H, W = features.size()
        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.view(B, W, C * H)
        features = self.cnn_proj(features)
        features = features.permute(1, 0, 2)
        rnn_out, _ = self.rnn(features)
        logits = self.fc(rnn_out)
        return logits


def build_crnn() -> CRNN:
    return CRNN()


def save_crnn(model: CRNN, path: str):
    torch.save(model.state_dict(), path)


def load_crnn(path: str, map_location=None) -> CRNN:
    model = build_crnn()
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    return model


#Below is just a little test to see how it works
#My PyTorch is CPU-only, not CUDA, so you can ignore what I did with 'device'
from PIL import Image
import torchvision.transforms as transforms
import os
import glob

def load_real_sample(device):                                               
    images_dir = config.PROCESSED_DATA_DIR / "words" / "final_images"
    image_files = glob.glob(str(images_dir / "*.png"))
    if len(image_files) == 0:
        raise FileNotFoundError("No images found.")

    img_path = image_files[0] 
    print("Using real image:", img_path)

    img = Image.open(img_path).convert("L")

    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor()
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor.to(device)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_crnn().to(device)

    batch_size = 1
    img_height = config.IMG_HEIGHT
    img_width = config.IMG_WIDTH
    channels = config.IMG_CHANNELS

    x = load_real_sample(device)

    logits = model(x)

    print("Input shape: ", x.shape)
    print("Output shape:", logits.shape)
    T, B, C = logits.shape
    print("T (time steps):", T)
    print("B (batch size):", B)
    print("C (num classes):", C, "== config.NUM_CLASSES?", C == config.NUM_CLASSES)


    #OUTPUT: 
    #Using real image: c:\Users\shanj\ComVision\5190-crnn-ocr\data\processed\words\final_images\000000.png
    #Input shape:  torch.Size([1, 1, 32, 128])
    #Output shape: torch.Size([16, 1, 63])
    #T (time steps): 16
    #B (batch size): 1
    #C (num classes): 63 == config.NUM_CLASSES? True

        
