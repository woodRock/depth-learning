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
            nn.Linear(512, in_features) 
        )

        # 4. Classifier Head - NEW
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 4) # 4 Classes: Kingfish, Snapper, Cod, Empty
        )

    def forward(self, vis, ac):
        """Returns predicted latent, target latent, and species logits"""
        # Target representation
        with torch.no_grad():
            target_latent = self.target_encoder(vis)
            target_latent = F.normalize(target_latent, p=2, dim=-1)

        # Context representation
        context_latent = self.context_encoder(ac)

        # 1. Prediction for Image Reconstruction
        predicted_target = self.predictor(context_latent)
        predicted_target = F.normalize(predicted_target, p=2, dim=-1)

        # 2. Classification for Accuracy
        species_logits = self.classifier(context_latent)

        return predicted_target, target_latent, species_logits

    def forward_ac_to_vis_latent(self, ac):
        """Inference: generate latent + species prediction"""
        context_latent = self.context_encoder(ac)
        predicted_target = self.predictor(context_latent)
        species_logits = self.classifier(context_latent)
        return F.normalize(predicted_target, p=2, dim=-1), species_logits
