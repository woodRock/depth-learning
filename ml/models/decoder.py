import torch
import torch.nn as nn

class LatentDecoder(nn.Module):
    """
    Decodes a [512, 7, 7] spatial latent grid back into a 224x224 RGB image.
    Updated for 7x7 native ResNet features.
    """
    def __init__(self, in_channels=512):
        super().__init__()
        
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
            nn.Sigmoid() 
        )

    def forward(self, x):
        # Input x is [B, 512, 7, 7]
        return self.decoder(x)
