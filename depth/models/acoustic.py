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
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))

class TemporalAttention(nn.Module):
    """Attention over the temporal (ping) dimension."""
    def __init__(self, embed_dim, nhead=4, dropout=0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [B, T, D]
        residual = x
        x = self.norm(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x)
        return x + residual

class ConvEncoder(nn.Module):
    """
    Improved Residual 2D-CNN with multi-scale features and temporal attention.
    Processes echogram as (B, 3, 32, 256) - treating time and depth as spatial dims.
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        # Stem with larger kernel for better temporal coverage
        self.stem = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Dropout2d(0.1)
        )

        # Layer 1: Fine-scale features
        self.layer1 = ResBlock2d(48, dropout=0.1)
        self.down1 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )

        # Layer 2: Medium-scale features
        self.layer2 = ResBlock2d(48, dropout=0.15)
        self.down2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Layer 3: Coarse-scale features
        self.layer3 = ResBlock2d(64, dropout=0.2)
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )

        # Layer 4: High-level features
        self.layer4 = ResBlock2d(96, dropout=0.2)
        
        # Multi-scale pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 8))  # (T, D) -> (4, 8)
        self.max_pool = nn.AdaptiveMaxPool2d((4, 8))
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(96 * 4 * 8 * 2, 768),
            nn.BatchNorm1d(768),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(768, embed_dim)
        )

    def forward(self, x):
        x = x.view(-1, 3, 32, 256)
        x = self.stem(x)
        x = self.down1(self.layer1(x))
        x = self.down2(self.layer2(x))
        x = self.down3(self.layer3(x))
        x = self.layer4(x)
        
        # Multi-scale pooling
        avg_feat = self.avg_pool(x).flatten(1)
        max_feat = self.max_pool(x).flatten(1)
        x = torch.cat([avg_feat, max_feat], dim=-1)
        
        return self.fusion(x)

class TransformerEncoder(nn.Module):
    """
    Improved Depth-Aware Transformer with:
    - Better depth feature extraction
    - Temporal attention
    - Multi-scale pooling
    """
    def __init__(self, embed_dim=256, nhead=8, num_layers=6):
        super().__init__()

        # 1. Enhanced Depth Feature Extractor
        self.feature_extractor = nn.Sequential(
            Unsqueeze(1),  # [B*32, 1, 256*3]
            nn.Conv1d(1, 48, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(48, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(96, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten()  # 128 * 16 = 2048
        )

        self.ping_proj = nn.Sequential(
            nn.Linear(2048, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        self.pos_encoding = nn.Parameter(torch.zeros(1, 32, 128))
        nn.init.trunc_normal_(self.pos_encoding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True,
            dropout=0.2,
            norm_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-scale aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 512),  # *2 for global + local
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embed_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * 32, 256 * 3)

        x = self.feature_extractor(x)
        x = x.view(batch_size, 32, 2048)
        x = self.ping_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        
        # Global and local features
        x_perm = x.transpose(1, 2)  # [B, D, T]
        global_feat = self.global_pool(x_perm).squeeze(-1)  # [B, D]
        local_feat = x.mean(dim=1)  # [B, D]
        
        x = torch.cat([global_feat, local_feat], dim=-1)
        return self.fc(x)
