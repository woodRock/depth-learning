"""
LeWM++ (LeWorldModel Plus Plus) - Multi-modal JEPA with SigReg Regularization.

This model combines the best of both architectures:
- Multi-modal contrastive learning from JEPA (visual teacher)
- Gaussian latent space regularization from LeWM (SigReg)
- Flexible task heads (presence, counting, single-label)

Architecture:
    Visual Image → [Frozen ResNet-18] → Visual Latent (target)
    Acoustic → [Acoustic Encoder] → Acoustic Latent
    Acoustic Latent → [Predictor MLP] → Predicted Visual Latent
    Acoustic Latent → [SigReg] → Gaussian Regularization Loss
    Acoustic Latent → [Task Head] → Predictions

Loss:
    L = L_JEPA + L_task + λ_sigreg * L_sigreg + λ_recon * L_recon

Usage:
    from models.lewm_plus import LeWMPlus
    model = LeWMPlus(ac_encoder, task="counting", use_sigreg=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights

from .lewm import SIGReg


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


class LeWMPlus(nn.Module):
    """
    LeWM++: Multi-modal JEPA with SigReg regularization.
    
    Combines:
    - Visual and acoustic encoders (JEPA)
    - Gaussian latent regularization (SigReg from LeWM)
    - Optional world reconstruction decoder
    
    Supports three task types:
    - "presence": Multi-label presence/absence detection (sigmoid output)
    - "single_label": Single-label classification (softmax output)
    - "counting": Count regression (tanh-scaled output for non-negative counts)
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
        max_count=30.0,  # Maximum count for counting task
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_sigreg = use_sigreg
        self.sigreg_weight = sigreg_weight
        self.use_decoder = use_decoder
        self.task = task
        self.max_count = max_count

        # 1. Visual Target Encoder (frozen ResNet-18)
        resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.target_encoder = nn.Sequential(*list(resnet.children())[:-2])
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 2. Acoustic Context Encoder (provided)
        self.context_encoder = ac_encoder

        # 3. Predictor Network (acoustic → visual latent)
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

        # 4. SigReg Gaussian Regularizer
        if use_sigreg:
            self.sigreg = SIGReg(
                embed_dim=embed_dim,
                n_projections=10,
                sigma=1.0
            )

        # 5. Task-specific Head
        if task == "counting":
            # Counting: tanh-scaled output for [0, max_count] range
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
                nn.Linear(64, n_classes),
            )
            self.criterion_cls = nn.MSELoss()
        elif task == "presence":
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

        # 6. Optional World Decoder (for visual reconstruction)
        if use_decoder:
            from .lewm import WorldDecoder
            self.decoder = WorldDecoder(embed_dim=embed_dim)

    def forward(self, vis, ac, labels=None):
        """
        Forward pass with multi-modal input.
        
        Args:
            vis: Visual images (B, 3, 224, 224)
            ac: Acoustic history (B, 24576) flattened
            labels: Ground truth labels
        
        Returns:
            predicted_target: Predicted visual latent
            target_latent: Target visual latent (frozen)
            species_logits: Task-specific logits
            recon_img: Reconstructed image (if use_decoder)
            sigreg_loss: SigReg regularization loss
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

        # Task-specific output
        raw_logits = self.classifier(context_latent)

        if self.task == "counting":
            # Scale output to [0, max_count] using tanh
            species_logits = torch.tanh(raw_logits / 5.0) * self.max_count
            species_logits = species_logits.clamp(min=0)
        else:
            species_logits = raw_logits

        # World reconstruction (optional)
        recon_img = None
        if self.use_decoder:
            recon_img = self.decoder(context_latent)

        # SigReg loss (optional)
        sigreg_loss = None
        if self.use_sigreg:
            context_latent_3d = context_latent.unsqueeze(1)
            sigreg_loss = self.sigreg(context_latent_3d)

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
            species_logits: Task-specific logits
            labels: Ground truth labels
            recon_img: Reconstructed image
            target_img: Target image for reconstruction loss
            sigreg_loss: Pre-computed SigReg loss
            recon_weight: Weight for reconstruction loss
        
        Returns:
            total_loss, loss_jepa, loss_cls, loss_sigreg, loss_recon
        """
        # JEPA loss: cosine similarity
        loss_jepa = (1.0 - F.cosine_similarity(predicted_target, target_latent, dim=-1)).mean()

        # Task-specific loss
        if self.task == "counting":
            # Counting: MSE between predicted and actual counts
            pred_counts = species_logits.clamp(min=0)
            true_counts = labels.clamp(min=0)
            loss_cls = self.criterion_cls(pred_counts, true_counts)
        elif self.task == "presence":
            # Multi-label: BCE with logits
            loss_cls = self.criterion_cls(species_logits, labels)
        else:
            # Single-label: Cross-entropy or focal loss
            loss_cls = self.criterion_cls(species_logits, labels)

        # SigReg loss
        if sigreg_loss is None and self.use_sigreg:
            predicted_target_3d = predicted_target.flatten(1).unsqueeze(1)
            sigreg_loss = self.sigreg(predicted_target_3d)
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
