import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class CrossModalJEPA(nn.Module):
    def __init__(self, ac_encoder, target_dim=512, embed_dim=256):
        super().__init__()
        
        # 1. Target Encoder (Visual) - FROZEN to prevent collapse
        self.target_encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.target_encoder.fc.in_features
        self.target_encoder.fc = nn.Identity() # Outputs 512-dim features
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        # 2. Context Encoder (Acoustic) - Trainable
        self.context_encoder = ac_encoder
        
        # 3. Predictor - Trainable
        # Maps the acoustic embedding to the visual latent space
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, in_features) # Matches ResNet18 output dim
        )

    def forward(self, vis, ac):
        """Returns the predicted visual latent and the actual visual latent for loss calculation"""
        # Target representation (no gradients)
        with torch.no_grad():
            target_latent = self.target_encoder(vis)
            target_latent = F.normalize(target_latent, p=2, dim=-1)
            
        # Context representation
        context_latent = self.context_encoder(ac)
        
        # Prediction
        predicted_target = self.predictor(context_latent)
        predicted_target = F.normalize(predicted_target, p=2, dim=-1)
        
        return predicted_target, target_latent

    def forward_ac_to_vis_latent(self, ac):
        """Inference time: generate the visual latent space purely from acoustic data"""
        context_latent = self.context_encoder(ac)
        predicted_target = self.predictor(context_latent)
        return F.normalize(predicted_target, p=2, dim=-1)
