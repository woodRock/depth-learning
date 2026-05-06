"""
ViT pretraining (MAE) + ViT-JEPA trainer.

Usage flow:
  1. ViTPretrainTrainer.pretrain(train_loader, epochs) — trains VisualMAE on
     visual frames, saves encoder weights to weights_dir/vit_pretrained.pth.
  2. ViTJEPATrainer.build_model()  — loads those weights into ViTEncoder,
     freezes it, then wires up ViTCrossModalJEPA like the standard JEPA.
     Falls back to random ViT init if no pretrained file is found.
"""

import os
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.logging import get_logger
from .jepa_trainer import JEPATrainer

logger = get_logger(__name__)


class ViTPretrainTrainer:
    """Self-supervised ViT pretraining via MAE on DepthSim visual frames."""

    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device

    def pretrain(self, train_loader: DataLoader, epochs: int = 50) -> str:
        """Train VisualMAE and save encoder weights.

        Args:
            train_loader: Yields (vis, ac, labels); only vis is used.
            epochs:       Number of MAE pretraining epochs.

        Returns:
            Path to saved ViTEncoder state_dict.
        """
        from models.vit_jepa import VisualMAE

        model = VisualMAE(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=384, encoder_depth=6, encoder_heads=6,
            decoder_dim=192, decoder_depth=4, decoder_heads=3,
            mask_ratio=0.75,
        ).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        os.makedirs(self.config.weights_dir, exist_ok=True)
        save_path = os.path.join(self.config.weights_dir, "vit_pretrained.pth")

        best_loss = float('inf')

        print(f"\n--- ViT MAE Pretraining ({epochs} epochs) ---")
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0

            pbar = tqdm(train_loader, desc=f"  Pretrain {epoch+1}/{epochs}")
            for vis, _ac, _labels in pbar:
                vis = vis.to(self.device)

                optimizer.zero_grad()
                pred, mask = model(vis)
                loss = model.compute_loss(vis, pred, mask)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"mae_loss": f"{loss.item():.4f}"})

            scheduler.step()
            avg_loss = total_loss / len(train_loader)

            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.encoder.state_dict(), save_path)
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}  ← best, saved")
            else:
                print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

        print(f"\n✓ ViT pretraining complete. Encoder saved → {save_path}")
        return save_path


class ViTJEPATrainer(JEPATrainer):
    """JEPA trainer that uses a pretrained ViT-Small as the visual teacher.

    Identical training loop to JEPATrainer — only build_model differs.
    """

    def build_model(self) -> nn.Module:
        from models.acoustic import ConvEncoder, TransformerEncoder
        from models.lstm import AcousticLSTM
        from models.ast import AcousticAST
        from models.vit_jepa import ViTEncoder, ViTCrossModalJEPA

        task = getattr(self.config, 'task', 'presence')
        vit_pretrain_path = getattr(self.config, 'vit_pretrain_path', None)

        # Acoustic encoder (same choices as standard JEPA)
        if self.config.model_type == "conv":
            ac_encoder = ConvEncoder(embed_dim=self.config.embed_dim)
        elif self.config.model_type == "lstm":
            ac_encoder = AcousticLSTM(
                input_dim=768, hidden_dim=256, num_classes=self.config.embed_dim
            )
        elif self.config.model_type == "ast":
            ac_encoder = AcousticAST(
                embed_dim=self.config.embed_dim, num_classes=self.config.embed_dim
            )
        else:  # transformer (default)
            ac_encoder = TransformerEncoder(embed_dim=self.config.embed_dim)

        # ViT encoder (same config used during pretraining)
        vit_encoder = ViTEncoder(
            img_size=224, patch_size=16, in_chans=3,
            embed_dim=384, depth=6, num_heads=6,
        )

        if vit_pretrain_path and os.path.exists(vit_pretrain_path):
            state = torch.load(vit_pretrain_path, map_location=self.device, weights_only=True)
            vit_encoder.load_state_dict(state)
            logger.info(f"Loaded pretrained ViT encoder from {vit_pretrain_path}")
        else:
            logger.warning(
                "No pretrained ViT found at %s — using random initialisation",
                vit_pretrain_path
            )

        return ViTCrossModalJEPA(
            ac_encoder=ac_encoder,
            vit_encoder=vit_encoder,
            embed_dim=self.config.embed_dim,
            use_focal_loss=self.config.use_focal_loss,
            task=task,
        ).to(self.device)
