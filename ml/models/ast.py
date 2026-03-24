import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=(32, 256), patch_size=(8, 16), in_channels=3, d_model=128):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.projection = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x) # (B, d_model, h, w)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, d_model)
        return x

class AcousticAST(nn.Module):
    """
    Audio Spectrogram Transformer tailored for acoustic data.
    """
    def __init__(self, d_model=256, nhead=8, num_layers=4, num_classes=4):
        super().__init__()
        
        # 32x256 image -> 8x16 patches -> (4 * 16) = 64 patches
        self.patch_embed = PatchEmbedding(img_size=(32, 256), patch_size=(8, 16), in_channels=3, d_model=d_model)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x can be (B, 32*256*3) or (B, 3, 32, 256)
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.view(batch_size, 32, 256, 3).permute(0, 3, 1, 2)
        elif x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        x = self.transformer(x)
        
        # Output from CLS token
        logits = self.classifier(x[:, 0, :])
        return logits
