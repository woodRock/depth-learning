"""Training strategies for decoder, fusion, translator, and MAE models."""

from typing import Dict, Any, Optional, Tuple
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb

from .base import BaseTrainer
from data.data import FishDataset, ImageLatentDataset


class DecoderTrainer(BaseTrainer):
    """Trainer for Latent Decoder model."""

    def __init__(self, config: Any, device: torch.device):
        super().__init__(config, device)
        self.jepa = self._load_jepa_model()
        self.jepa.eval()

    def build_model(self) -> nn.Module:
        """Build Latent Decoder."""
        from models.decoder import LatentDecoder
        return LatentDecoder().to(self.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc="Training Decoder")
        
        for img in pbar:
            img = img.to(self.device)
            
            with torch.no_grad():
                target_latent = self.jepa.target_encoder(img)
                target_latent = F.normalize(target_latent, p=2, dim=1)
            
            self.optimizer.zero_grad()
            recon_img = self.model(target_latent)
            
            # Weighted loss
            target_img = (img * 0.225 + 0.456).clamp(0, 1)
            mask = (target_img.mean(dim=1, keepdim=True) > 0.15).float()
            weights = 1.0 + (mask * 9.0)
            
            loss = (weights * (recon_img - target_img)**2).mean()
            loss += 0.1 * F.l1_loss(recon_img, target_img)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return {"loss": total_loss / len(loader)}

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for img in loader:
                img = img.to(self.device)
                target_latent = self.jepa.target_encoder(img)
                target_latent = F.normalize(target_latent, p=2, dim=1)
                recon_img = self.model(target_latent)
                target_img = (img * 0.225 + 0.456).clamp(0, 1)
                loss = F.mse_loss(recon_img, target_img)
                total_loss += loss.item()
        
        return {
            "loss": total_loss / len(loader),
            "acc": -total_loss / len(loader),  # Higher is better for early stopping
            "last_recon": recon_img[0].cpu().clamp(0, 1),
            "last_target": target_img[0].cpu().clamp(0, 1)
        }

    def _load_jepa_model(self) -> nn.Module:
        """Load pre-trained JEPA model for latent generation."""
        from models.jepa import CrossModalJEPA
        from models.acoustic import ConvEncoder, TransformerEncoder
        
        jepa_weights = os.path.join(self.config.weights_dir, "fish_clip_model.pth")
        config_path = os.path.join(self.config.weights_dir, "model_config.json")
        
        if not os.path.exists(jepa_weights):
            # Try looking in subdirectories if weights_dir is just "weights"
            alt_path = os.path.join(self.config.weights_dir, f"jepa_{self.config.dataset}", "fish_clip_model.pth")
            if os.path.exists(alt_path):
                jepa_weights = alt_path
                config_path = os.path.join(os.path.dirname(alt_path), "model_config.json")
            else:
                raise FileNotFoundError(f"JEPA weights not found. Train JEPA first.")
        
        model_type = "conv"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                model_type = cfg.get("model_type", "conv")
        
        ac_encoder = ConvEncoder() if model_type == "conv" else TransformerEncoder()
        jepa = CrossModalJEPA(ac_encoder=ac_encoder).to(self.device)
        jepa.load_state_dict(torch.load(jepa_weights, map_location=self.device, weights_only=True))
        return jepa

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        return -val_metrics["loss"]


class FusionTrainer(BaseTrainer):
    """Trainer for Masked Attention Fusion model."""

    def __init__(self, config: Any, device: torch.device):
        super().__init__(config, device)
        from torchvision import models
        visual_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.visual_backbone = nn.Sequential(*list(visual_model.children())[:-1]).to(self.device)
        self.visual_backbone.eval()
        self.criterion = nn.CrossEntropyLoss()

    def build_model(self) -> nn.Module:
        from models.fusion import MaskedAttentionFusion
        return MaskedAttentionFusion(d_model=256, nhead=8, num_classes=4).to(self.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        train_correct = 0
        train_total = 0
        total_loss = 0
        
        pbar = tqdm(loader, desc="Training Fusion")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
            
            with torch.no_grad():
                vis_feats = self.visual_backbone(vis).squeeze(-1).squeeze(-1)
            
            # Modality dropout
            dropout_prob = getattr(self.config, 'dropout_prob', 0.5)
            mask = (torch.rand(vis_feats.size(0), 1, device=self.device) > dropout_prob).float()
            vis_feats = vis_feats * mask
            
            self.optimizer.zero_grad()
            logits = self.model(vis_feats, ac)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({"acc": f"{100*train_correct/train_total:.1f}%"})
            
        return {"loss": total_loss / len(loader), "acc": train_correct / train_total}

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_fusion_correct = 0
        val_acoustic_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for vis, ac, labels in loader:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
                vis_feats = self.visual_backbone(vis).squeeze(-1).squeeze(-1)
                
                # Fusion
                logits_fusion = self.model(vis_feats, ac, mask_ratio=0.0)
                preds_fusion = torch.argmax(logits_fusion, dim=1)
                val_fusion_correct += (preds_fusion == labels).sum().item()
                
                # Acoustic only
                zero_vis = torch.zeros_like(vis_feats)
                logits_acoustic = self.model(zero_vis, ac, mask_ratio=0.0)
                preds_acoustic = torch.argmax(logits_acoustic, dim=1)
                val_acoustic_correct += (preds_acoustic == labels).sum().item()
                
                val_total += labels.size(0)
        
        return {
            "acc": val_fusion_correct / val_total,
            "acoustic_acc": val_acoustic_correct / val_total
        }

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        return val_metrics["acoustic_acc"]


class TranslatorTrainer(BaseTrainer):
    """Trainer for Acoustic-to-Image Translator model."""

    def __init__(self, config: Any, device: torch.device):
        super().__init__(config, device)
        self.recon_loss_fn = nn.MSELoss()
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225],
        )

    def build_model(self) -> nn.Module:
        from models.transformer_translator import AcousticToImageTransformer
        return AcousticToImageTransformer(
            d_model=getattr(self.config, 'd_model', 256), 
            patch_size=getattr(self.config, 'patch_size', 16),
        ).to(self.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        train_recon_loss = 0
        train_cls_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(loader, desc="Training Translator")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            gen_img, species_logits = self.model(ac)
            
            vis_01 = torch.stack([self.inv_normalize(v) for v in vis])
            recon_loss = self.recon_loss_fn(gen_img, vis_01)
            cls_loss = self.cls_loss_fn(species_logits, labels)
            
            loss = 10.0 * recon_loss + cls_loss
            loss.backward()
            self.optimizer.step()
            
            train_recon_loss += recon_loss.item()
            train_cls_loss += cls_loss.item()
            
            preds = torch.argmax(species_logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            pbar.set_postfix({"recon": f"{recon_loss.item():.4f}", "cls": f"{cls_loss.item():.3f}"})
            
        return {
            "loss": (train_recon_loss + train_cls_loss) / len(loader),
            "recon_loss": train_recon_loss / len(loader),
            "cls_loss": train_cls_loss / len(loader),
            "acc": train_correct / train_total
        }

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        val_recon_loss = 0
        val_cls_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for vis, ac, labels in loader:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
                gen_img, species_logits = self.model(ac)
                
                vis_01 = torch.stack([self.inv_normalize(v) for v in vis])
                recon_loss = self.recon_loss_fn(gen_img, vis_01)
                cls_loss = self.cls_loss_fn(species_logits, labels)
                
                val_recon_loss += recon_loss.item()
                val_cls_loss += cls_loss.item()
                
                preds = torch.argmax(species_logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        return {
            "loss": (val_recon_loss + val_cls_loss) / len(loader),
            "acc": val_correct / val_total,
            "last_recon": gen_img[0].cpu().detach().clamp(0, 1),
            "last_target": vis_01[0].cpu().detach().clamp(0, 1)
        }

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        return -val_metrics["loss"]


class MAETrainer(BaseTrainer):
    """Trainer for Acoustic Masked Autoencoder."""

    def build_model(self) -> nn.Module:
        from models.mae import AcousticMAE
        return AcousticMAE(mask_ratio=getattr(self.config, 'mask_ratio', 0.75)).to(self.device)

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        pbar = tqdm(loader, desc="Training MAE")
        
        for _, ac, _ in pbar:
            ac = ac.to(self.device)
            ac_img = ac.view(-1, 32, 256, 3).permute(0, 3, 1, 2)
            
            self.optimizer.zero_grad()
            pred, mask = self.model(ac)
            
            # Patchify target
            p1, p2 = 8, 16
            target = ac_img.unfold(2, p1, p1).unfold(3, p2, p2)
            target = target.permute(0, 2, 3, 1, 4, 5).contiguous()
            target = target.view(ac_img.size(0), -1, 3 * p1 * p2)
            
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        return {"loss": total_loss / len(loader)}

    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for _, ac, _ in loader:
                ac = ac.to(self.device)
                ac_img = ac.view(-1, 32, 256, 3).permute(0, 3, 1, 2)
                pred, mask = self.model(ac)
                
                p1, p2 = 8, 16
                target = ac_img.unfold(2, p1, p1).unfold(3, p2, p2)
                target = target.permute(0, 2, 3, 1, 4, 5).contiguous()
                target = target.view(ac_img.size(0), -1, 3 * p1 * p2)
                
                loss = (pred - target) ** 2
                loss = loss.mean(dim=-1)
                loss = (loss * mask).sum() / mask.sum()
                total_loss += loss.item()
        
        return {"loss": total_loss / len(loader), "acc": -total_loss / len(loader)}

    def _get_save_score(self, val_metrics: Dict[str, float]) -> float:
        return -val_metrics["loss"]
