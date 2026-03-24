import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class CrossModalJEPA(nn.Module):
    def __init__(self, ac_encoder, embed_dim=256, use_focal_loss=True):
        super().__init__()

        # 1. Target Encoder (Visual)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.target_encoder = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 2. Context Encoder (Acoustic)
        self.context_encoder = ac_encoder

        # 3. Predictor (Maps to 7x7 visual grid)
        # More expressive predictor with residual connections
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512 * 7 * 7),
            nn.Unflatten(1, (512, 7, 7))
        )

        # 4. Enhanced Classifier Head with label smoothing support
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 768),
            nn.BatchNorm1d(768),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 4)
        )
        
        self.use_focal_loss = use_focal_loss
        if use_focal_loss:
            self.criterion_cls = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.criterion_cls = nn.CrossEntropyLoss()

    def forward(self, vis, ac, labels=None):
        with torch.no_grad():
            target_feat = self.target_encoder(vis)
            target_latent = F.normalize(target_feat, p=2, dim=1)

        context_latent = self.context_encoder(ac)
        predicted_target = self.predictor(context_latent)
        predicted_target = F.normalize(predicted_target, p=2, dim=1)
        species_logits = self.classifier(context_latent)

        return predicted_target, target_latent, species_logits

    def compute_loss(self, predicted_target, target_latent, species_logits, labels):
        """Compute combined JEPA + classification loss with optional label smoothing."""
        # JEPA loss: cosine similarity
        loss_jepa = (1.0 - F.cosine_similarity(predicted_target, target_latent, dim=-1)).mean()
        
        # Classification loss (focal or cross-entropy)
        loss_cls = self.criterion_cls(species_logits, labels)
        
        # Combined loss
        loss = loss_jepa + loss_cls
        
        return loss, loss_jepa, loss_cls

    def forward_ac_to_vis_latent(self, ac):
        context_latent = self.context_encoder(ac)
        predicted_target = self.predictor(context_latent)
        species_logits = self.classifier(context_latent)
        return F.normalize(predicted_target, p=2, dim=1), species_logits
