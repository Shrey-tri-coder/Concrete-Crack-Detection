"""Compact CNN for 128x128 grayscale crack classification (shared by training & analysis)."""
import torch
import torch.nn as nn


def conv_bn(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class CrackCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            *conv_bn(1, 32), nn.MaxPool2d(2),                        # 128 -> 64
            *conv_bn(32, 64), nn.MaxPool2d(2),                       # 64 -> 32
            *conv_bn(64, 128), *conv_bn(128, 128), nn.MaxPool2d(2),  # 32 -> 16
            *conv_bn(128, 256), nn.MaxPool2d(2),                     # 16 -> 8
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, 1))

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


# Fixed input rescaling (x/255 - 0.5) / 0.25. These are NOT the dataset statistics (those are
# mean 0.637, std 0.108); exact standardisation is unnecessary because the first convolution has no
# bias and is followed by BatchNorm, which removes any constant offset/scale of the input.
# The trained weights in results/cnn_best.pt assume these values - do not change without retraining.
MEAN, STD = 0.5, 0.25


def to_tensor(batch_uint8):
    x = torch.from_numpy(batch_uint8).float().div_(255.0).unsqueeze(1)
    return (x - MEAN) / STD
