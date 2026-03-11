import torch
import torch.nn as nn

class LatentDecoder(nn.Module):
    """
    Decodes a 512-dim embedding back into a 224x224 RGB image.
    Uses Transposed Convolutions to upsample the latent vector.
    """
    def __init__(self, latent_dim=512):
        super().__init__()
        
        # 1. Project latent to a small spatial grid
        # 512 -> (512 * 7 * 7)
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * 7 * 7),
            nn.ReLU()
        )
        
        # 2. Upsample via Transposed Convolutions
        self.decoder = nn.Sequential(
            # [B, 512, 7, 7] -> [B, 256, 14, 14]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            # [B, 256, 14, 14] -> [B, 128, 28, 28]
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            # [B, 128, 28, 28] -> [B, 64, 56, 56]
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # [B, 64, 56, 56] -> [B, 32, 112, 112]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # [B, 32, 112, 112] -> [B, 3, 224, 224]
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid() # Normalize pixel values to [0, 1]
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 512, 7, 7)
        return self.decoder(x)
