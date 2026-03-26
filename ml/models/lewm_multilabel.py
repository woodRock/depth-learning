"""
LeWorldModel with Presence/Absence Detection (Multi-label Classification).

This model predicts which species are present anywhere in the 32-ping window,
rather than predicting a single majority class. This is a more concrete task
that doesn't require global temporal integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lewm import SIGReg, Embedder, ARPredictor, WorldDecoder


class LeWorldModelMultiLabel(nn.Module):
    """
    LeWorldModel with multi-label classification head.
    
    Key differences from standard LeWM:
    - Output: (B, n_classes) sigmoid logits instead of single class
    - Loss: Binary Cross-Entropy (BCE) instead of Cross-Entropy
    - Task: "Which species are present?" instead of "Which species dominates?"
    """
    def __init__(
        self,
        input_dim=None,         # Auto-detect from data (24576 for 32 steps, 49152 for 64)
        embed_dim=256,          # Latent embedding dimension
        n_timesteps=None,       # Auto-detect from input_dim
        num_layers=8,           # Number of transformer layers in predictor
        num_heads=8,            # Number of attention heads
        mlp_ratio=4.0,          # MLP hidden dim ratio
        drop=0.1,               # Dropout rate
        n_classes=4,            # Number of classification classes (Kingfish, Snapper, Cod, Empty)
        use_classifier=True,    # Whether to include classification head
        use_decoder=False,      # Whether to include world reconstruction decoder
        task: str = "presence", # "presence" or "counting"
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_classifier = use_classifier
        self.use_decoder = use_decoder
        self.n_classes = n_classes
        self.task = task

        # 1. Embedder (encodes input to latent space, auto-detects timesteps)
        self.embedder = Embedder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_timesteps=n_timesteps
        )

        # 2. Predictor (autoregressive transformer)
        self.predictor = ARPredictor(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop
        )

        # 3. Prediction projection
        self.pred_proj = nn.Linear(embed_dim, embed_dim)

        # 4. Gaussian regularizer
        self.sigreg = SIGReg(embed_dim=embed_dim, n_projections=10, sigma=1.0)

        # 5. Task-specific head
        if use_classifier:
            if task == "counting":
                # Counting: ReLU activation for non-negative counts
                self.classifier = nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(256, n_classes),
                    nn.ReLU()  # Ensure non-negative counts
                )
            else:
                # Presence/absence: linear output for BCEWithLogitsLoss
                self.classifier = nn.Sequential(
                    nn.Linear(embed_dim, 512),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(256, n_classes)
                )

        # 6. World Decoder (optional, for visual reconstruction)
        if use_decoder:
            self.decoder = WorldDecoder(embed_dim=embed_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def encode(self, x):
        """
        Encode input to latent embeddings.

        Args:
            x: (B, input_dim) or (B, T, D) input tensor

        Returns:
            embeddings: (B, T, embed_dim)
        """
        return self.embedder(x)

    def predict(self, embeddings, condition=None):
        """
        Predict future embeddings autoregressively.

        Args:
            embeddings: (B, T, embed_dim) input embeddings
            condition: (B, embed_dim) conditioning vector (optional)

        Returns:
            predictions: (B, T, embed_dim) predicted embeddings
        """
        return self.predictor(embeddings, condition)

    def forward(self, x, condition=None):
        """
        Forward pass: encode and predict.

        Args:
            x: (B, input_dim) input tensor (flattened echogram history)
            condition: (B, embed_dim) conditioning vector (optional)

        Returns:
            pred_emb: (B, T, embed_dim) predicted embeddings
            goal_emb: (B, T, embed_dim) target embeddings (last timestep)
            species_logits: (B, n_classes) multi-label classification logits
            recon_img: (B, 3, H, W) reconstructed image (if use_decoder)
        """
        # Encode input
        embeddings = self.embedder(x)  # (B, T, embed_dim)

        # Predict future embeddings
        pred_emb = self.predict(embeddings, condition)  # (B, T, embed_dim)
        pred_emb = self.pred_proj(pred_emb)

        # Goal embeddings (shifted by 1 timestep for prediction)
        goal_emb = embeddings[:, 1:, :]  # (B, T-1, embed_dim)
        pred_emb = pred_emb[:, :-1, :]   # (B, T-1, embed_dim)

        # Use mean embedding for classification and reconstruction
        mean_emb = embeddings.mean(dim=1)  # (B, embed_dim)

        # Multi-label classification (independent sigmoid for each species)
        if self.use_classifier:
            species_logits = self.classifier(mean_emb)  # (B, n_classes)
        else:
            species_logits = None

        # World reconstruction
        if self.use_decoder:
            recon_img = self.decoder(mean_emb)  # (B, 3, H, W)
        else:
            recon_img = None

        return pred_emb, goal_emb, species_logits, recon_img

    def compute_loss(
        self,
        pred_emb,
        goal_emb,
        species_logits,
        labels,  # Multi-hot (presence) or counts (counting): (B, n_classes)
        recon_img=None,
        target_img=None,
        sigreg_weight=0.1,
        recon_weight=0.01,
        pos_weight=None,  # Optional: weight for positive samples (handle class imbalance)
    ):
        """
        Compute LeWM loss with task-specific objectives.

        Args:
            pred_emb: (B, T, embed_dim) predicted embeddings
            goal_emb: (B, T, embed_dim) target embeddings
            species_logits: (B, n_classes) classification/regression logits
            labels: (B, n_classes) multi-hot (presence) or counts (counting)
            recon_img: (B, 3, H, W) reconstructed image (optional)
            target_img: (B, 3, H, W) target visual image (optional)
            sigreg_weight: weight for SIGReg regularizer
            recon_weight: weight for reconstruction loss
            pos_weight: (n_classes,) weight for positive samples in BCE loss

        Returns:
            total_loss, pred_loss, sigreg_loss, cls_loss, recon_loss
        """
        # 1. Prediction loss (MSE on last timestep)
        pred_loss = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="mean"
        )

        # 2. Gaussian regularizer
        sigreg_loss = self.sigreg(pred_emb)

        # 3. Task-specific classification/counting loss
        if species_logits is not None and labels is not None:
            if self.task == "counting":
                # Counting: MSE with soft clipping via tanh scaling
                # Scale outputs to reasonable range: tanh(x/5) * 30 → soft cap at ~30
                scaled_logits = torch.tanh(species_logits / 5.0) * 30.0
                # Don't clip targets too hard - let model learn the distribution
                labels_soft = labels.clamp(min=0)  # Only ensure non-negative
                cls_loss = F.mse_loss(scaled_logits, labels_soft)
            else:
                # Presence/absence: BCE with logits
                if pos_weight is not None:
                    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(species_logits.device))
                else:
                    bce_criterion = nn.BCEWithLogitsLoss()
                cls_loss = bce_criterion(species_logits, labels)
        else:
            cls_loss = torch.tensor(0.0, device=pred_emb.device)

        # 4. Reconstruction loss (MSE + L1 for sharpness)
        if recon_img is not None and target_img is not None:
            recon_mse = F.mse_loss(recon_img, target_img)
            recon_l1 = F.l1_loss(recon_img, target_img)
            recon_loss = recon_mse + 0.1 * recon_l1
        else:
            recon_loss = torch.tensor(0.0, device=pred_emb.device)

        # Total loss
        total_loss = pred_loss + sigreg_weight * sigreg_loss + cls_loss + recon_weight * recon_loss

        return total_loss, pred_loss, sigreg_loss, cls_loss, recon_loss

    def predict_presence(self, x, threshold=0.5):
        """
        Predict species presence/absence.

        Args:
            x: (B, input_dim) input tensor
            threshold: sigmoid threshold for presence detection

        Returns:
            probs: (B, n_classes) softmax probabilities (single-label)
            predictions: (B,) predicted class indices
        """
        self.eval()
        with torch.no_grad():
            _, _, species_logits, _ = self.forward(x)
            # Use softmax for single-label prediction
            probs = torch.softmax(species_logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        return probs, predictions

    def rollout(self, x, n_steps=10, condition=None):
        """
        Autoregressive rollout for planning/inference.

        Args:
            x: (B, input_dim) initial input
            n_steps: number of steps to rollout
            condition: (B, embed_dim) conditioning vector

        Returns:
            predictions: list of (B, embed_dim) predictions
        """
        predictions = []

        # Initial encoding
        embeddings = self.embedder(x)

        for _ in range(n_steps):
            # Predict next embedding
            pred = self.predict(embeddings, condition)
            pred = self.pred_proj(pred[:, -1:, :])  # (B, 1, embed_dim)

            predictions.append(pred[:, 0, :])

            # Append prediction to embeddings (autoregressive)
            embeddings = torch.cat([embeddings, pred], dim=1)

        return predictions
