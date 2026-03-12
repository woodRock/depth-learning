import torch
import torch.nn as nn
import math

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return x.unsqueeze(self.dim)

class ResBlock2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))

class ConvEncoder(nn.Module):
    """
    Residual 2D-CNN: Optimized for robust temporal-depth feature extraction.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.stem = nn.Sequential(
            Unsqueeze(1), # [B, 1, 32, 256]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.layer1 = ResBlock2d(32)
        self.down1 = nn.MaxPool2d(2) # [B, 32, 16, 128]
        
        self.layer2 = ResBlock2d(32)
        self.down2 = nn.MaxPool2d(2) # [B, 32, 8, 64]
        
        self.layer3 = ResBlock2d(32)
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4)) # [B, 32, 4, 4]
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        x = x.view(-1, 32, 256)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.down1(x)
        x = self.layer2(x)
        x = self.down2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        return self.fc(x)

class TransformerEncoder(nn.Module):
    """
    Deep Transformer with Pre-LayerNorm and Residual projection.
    """
    def __init__(self, embed_dim=256, nhead=8, num_layers=6):
        super().__init__()
        
        self.ping_proj = nn.Linear(256, 128)
        self.pos_encoding = nn.Parameter(torch.zeros(1, 32, 128))
        
        # Pre-LayerNorm Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, 
            nhead=nhead, 
            dim_feedforward=512, 
            batch_first=True,
            dropout=0.3,
            norm_first=True # This is Pre-LayerNorm
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 128, 512),
            nn.LayerNorm(512), # Extra stability
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        # x is [B, 32 * 256] -> [B, 32, 256]
        x = x.view(-1, 32, 256)
        x = self.ping_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        return self.fc(x)
