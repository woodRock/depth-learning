"""
Multi-modal JEPA with SigReg Regularization.

Fuses:
- Multi-modal training (visual + acoustic) from JEPA
- SigReg Gaussian regularization from LeWM
- Optional world reconstruction decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

from .lewm import SIGReg, Embedder, ARPredictor, WorldDecoder


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


class MultimodalJEPASigReg(nn.Module):
    """
    Multi-modal JEPA with SigReg regularization.
    
    Combines:
    - Visual and acoustic encoders (JEPA)
    - Gaussian latent regularization (SigReg from LeWM)
    - Optional world reconstruction decoder
    """
    def __init__(
        self,
        ac_encoder,
        embed_dim=256,
        use_focal_loss=True,
        task: str = "presence",
        use_sigreg=True,
        sigreg_weight=0.1,
        use_decoder=False,
        n_classes=4,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_sigreg = use_sigreg
        self.sigreg_weight = sigreg_weight
        self.use_decoder = use_decoder
        self.task = task
        
        # 1. Visual Target Encoder (frozen)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.target_encoder = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # 2. Acoustic Context Encoder
        self.context_encoder = ac_encoder
        
        # 3. Fusion/Predictor Network
        # Maps acoustic latent to visual latent space
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
        
        # 4. SigReg Gaussian Regularizer (from LeWM)
        if use_sigreg:
            self.sigreg = SIGReg(
                embed_dim=embed_dim,
                n_projections=10,
                sigma=1.0
            )
        
        # 5. Classification Head
        if task == "presence":
            # Multi-label: sigmoid output
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
                nn.Linear(64, n_classes)
            )
            self.criterion_cls = nn.BCEWithLogitsLoss()
        else:
            # Single-label: softmax output
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
                nn.Linear(64, n_classes)
            )
            self.use_focal_loss = use_focal_loss
            if use_focal_loss:
                self.criterion_cls = FocalLoss(alpha=1.0, gamma=2.0)
            else:
                self.criterion_cls = nn.CrossEntropyLoss()
        
        # 6. World Decoder (optional, from LeWM)
        if use_decoder:
            self.decoder = WorldDecoder(embed_dim=embed_dim)
    
    def forward(self, vis, ac, labels=None):
        """
        Forward pass with multi-modal input.
        
        Args:
            vis: Visual images (B, 3, 224, 224)
            ac: Acoustic history (B, 24576) flattened
            labels: Ground truth labels (B, 4) for multi-label or (B,) for single-label
        
        Returns:
            predicted_target: Predicted visual latent
            target_latent: Target visual latent (from frozen encoder)
            species_logits: Classification logits
            recon_img: Reconstructed image (if use_decoder)
            sigreg_loss: SigReg regularization loss (if use_sigreg)
        """
        # Target visual latent (frozen)
        with torch.no_grad():
            target_feat = self.target_encoder(vis)
            target_latent = F.normalize(target_feat, p=2, dim=1)
        
        # Acoustic latent
        context_latent = self.context_encoder(ac)
        
        # Predict visual latent from acoustic
        predicted_target = self.predictor(context_latent)
        predicted_target = F.normalize(predicted_target, p=2, dim=1)
        
        # Classification from acoustic latent
        species_logits = self.classifier(context_latent)
        
        # World reconstruction (optional)
        recon_img = None
        if self.use_decoder:
            recon_img = self.decoder(context_latent)
        
        # SigReg loss (optional)
        sigreg_loss = None
        if self.use_sigreg:
            sigreg_loss = self.sigreg(context_latent)
        
        return predicted_target, target_latent, species_logits, recon_img, sigreg_loss
    
    def compute_loss(
        self,
        predicted_target,
        target_latent,
        species_logits,
        labels,
        recon_img=None,
        target_img=None,
        sigreg_loss=None,
        recon_weight=0.01,
    ):
        """
        Compute combined loss.
        
        Args:
            predicted_target: Predicted visual latent
            target_latent: Target visual latent
            species_logits: Classification logits
            labels: Ground truth labels
            recon_img: Reconstructed image
            target_img: Target image for reconstruction loss
            sigreg_loss: Pre-computed SigReg loss
            recon_weight: Weight for reconstruction loss
        
        Returns:
            total_loss: Combined loss
            loss_jepa: JEPA prediction loss
            loss_cls: Classification loss
            loss_sigreg: SigReg regularization loss
            loss_recon: Reconstruction loss (if applicable)
        """
        # JEPA loss: cosine similarity
        loss_jepa = (1.0 - F.cosine_similarity(predicted_target, target_latent, dim=-1)).mean()
        
        # Classification loss
        if self.task == "presence":
            loss_cls = self.criterion_cls(species_logits, labels)
        else:
            loss_cls = self.criterion_cls(species_logits, labels)
        
        # SigReg loss
        if sigreg_loss is None and self.use_sigreg:
            sigreg_loss = self.sigreg(predicted_target.flatten(1))
        elif sigreg_loss is None:
            sigreg_loss = torch.tensor(0.0, device=predicted_target.device)
        
        # Reconstruction loss (optional)
        loss_recon = torch.tensor(0.0, device=predicted_target.device)
        if self.use_decoder and recon_img is not None and target_img is not None:
            loss_recon = F.mse_loss(recon_img, target_img) * recon_weight
        
        # Combined loss
        total_loss = loss_jepa + loss_cls + self.sigreg_weight * sigreg_loss + loss_recon
        
        return total_loss, loss_jepa, loss_cls, sigreg_loss, loss_recon
    
    def forward_ac_to_vis_latent(self, ac):
        """Acoustic-only inference (for evaluation)."""
        context_latent = self.context_encoder(ac)
        predicted_target = self.predictor(context_latent)
        predicted_target = F.normalize(predicted_target, p=2, dim=1)
        species_logits = self.classifier(context_latent)
        return predicted_target, species_logits
    
    def forward_ac_only(self, ac):
        """Acoustic-only forward pass (returns classification logits only)."""
        context_latent = self.context_encoder(ac)
        species_logits = self.classifier(context_latent)
        return species_logits
