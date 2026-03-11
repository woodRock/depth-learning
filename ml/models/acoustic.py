import torch
import torch.nn as nn
import math

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.unsqueeze(self.dim)

class ConvEncoder(nn.Module):
    """
    2D-CNN Architecture: Optimized for temporal-depth feature detection.
    Treats the 32-ping history as a 32x256 image.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            Unsqueeze(1), # [B, 1, 32, 256]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), # [B, 32, 16, 128]
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2), # [B, 64, 8, 64]
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), # [B, 128, 4, 4]
            nn.Flatten() # [B, 2048]
        )
        
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        # x input is [B, 32 * 256] flat history
        x = x.view(-1, 32, 256)
        return self.net(x)

class TransformerEncoder(nn.Module):
    """
    Temporal Transformer: Treats each of the 32 pings as a token.
    Each token contains 256 depth bins.
    """
    def __init__(self, embed_dim=256, nhead=8, num_layers=4):
        super().__init__()
        
        # Project 256 depth bins down to a 128-dim embedding
        self.ping_proj = nn.Linear(256, 128)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 32, 128))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=nhead, 
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 128, 512),
            nn.ReLU(),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        # x is [B, 32 * 256] -> [B, 32, 256]
        x = x.view(-1, 32, 256)
        x = self.ping_proj(x) + self.pos_encoding
        x = self.transformer(x)
        return self.fc(x)
