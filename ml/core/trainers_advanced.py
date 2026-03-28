"""Training strategies for decoder, fusion, translator, and MAE models."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
import numpy as np

from utils.config import DecoderConfig, FusionConfig, TranslatorConfig, MAEConfig
from data.data import FishDataset, ImageLatentDataset, create_visual_transform, AugmentationConfig


class DecoderTrainer:
    """Trainer for Latent Decoder model."""
    
    def __init__(self, config: DecoderConfig, device: torch.device):
        self.config = config
        self.device = device
    
    def train(self) -> None:
        """Full training pipeline for decoder."""
        from models.jepa import CrossModalJEPA
        from models.acoustic import ConvEncoder, TransformerEncoder
        from models.decoder import LatentDecoder
        
        # Setup data
        aug_config = AugmentationConfig(enabled=self.config.with_aug)
        transform = create_visual_transform(aug_config)
        dataset_path = self._get_dataset_path()
        
        dataloader = DataLoader(
            ImageLatentDataset(dataset_path, transform=transform),
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        # Load pre-trained JEPA
        jepa = self._load_jepa_model()
        jepa.eval()
        
        # Build decoder
        decoder = LatentDecoder().to(self.device)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=self.config.learning_rate)
        
        # Training loop
        for epoch in range(self.config.epochs):
            decoder.train()
            total_loss = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.epochs}")
            for img in pbar:
                img = img.to(self.device)
                
                with torch.no_grad():
                    target_latent = jepa.target_encoder(img)
                    target_latent = F.normalize(target_latent, p=2, dim=1)
                
                optimizer.zero_grad()
                recon_img = decoder(target_latent)
                
                # Weighted loss
                target_img = (img * 0.225 + 0.456).clamp(0, 1)
                mask = (target_img.mean(dim=1, keepdim=True) > 0.15).float()
                weights = 1.0 + (mask * 9.0)
                
                loss = (weights * (recon_img - target_img)**2).mean()
                loss += 0.1 * F.l1_loss(recon_img, target_img)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Logging
            wandb.log({"epoch": epoch + 1, "decoder_loss": total_loss / len(dataloader)})
        
        # Save model
        self._save_model(decoder)
    
    def _load_jepa_model(self) -> nn.Module:
        """Load pre-trained JEPA model."""
        import json
        
        jepa_weights = os.path.join(self.config.weights_dir, "fish_clip_model.pth")
        config_path = os.path.join(self.config.weights_dir, "model_config.json")
        
        if not os.path.exists(jepa_weights):
            raise FileNotFoundError(f"JEPA weights not found at {jepa_weights}")
        
        model_type = "conv"
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                model_type = cfg.get("model_type", "conv")
        
        ac_encoder = ConvEncoder() if model_type == "conv" else TransformerEncoder()
        jepa = CrossModalJEPA(ac_encoder=ac_encoder).to(self.device)
        
        try:
            jepa.load_state_dict(torch.load(jepa_weights, map_location=self.device))
        except RuntimeError as e:
            raise RuntimeError(
                f"Weight mismatch. Please re-run JEPA training first. Error: {e}"
            )
        
        return jepa
    
    def _get_dataset_path(self) -> str:
        """Get absolute path to dataset."""
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dataset", self.config.dataset)
        )
    
    def _save_model(self, decoder: nn.Module) -> None:
        """Save decoder model."""
        os.makedirs(self.config.weights_dir, exist_ok=True)
        torch.save(decoder.state_dict(), os.path.join(self.config.weights_dir, "decoder_model.pth"))


class FusionTrainer:
    """Trainer for Masked Attention Fusion model."""
    
    def __init__(self, config: FusionConfig, device: torch.device):
        self.config = config
        self.device = device
    
    def train(self) -> None:
        """Full training pipeline for fusion model."""
        from models.fusion import MaskedAttentionFusion
        from torchvision import models
        
        # Setup data
        aug_config = AugmentationConfig(enabled=self.config.with_aug)
        transform = create_visual_transform(aug_config)
        dataset_path = self._get_dataset_path()

        train_dataset = FishDataset(
            dataset_path,
            transform=transform,
            mode="train",
        )
        val_dataset = FishDataset(
            dataset_path,
            transform=transform,
            mode="val",
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
        )
        
        # Visual encoder (frozen ResNet50)
        visual_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        visual_backbone = nn.Sequential(*list(visual_model.children())[:-1]).to(self.device)
        visual_backbone.eval()
        
        # Fusion model
        fusion_model = MaskedAttentionFusion(
            d_model=256, 
            nhead=8, 
            num_classes=4,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            fusion_model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=0.01,
        )
        criterion = nn.CrossEntropyLoss()
        
        best_acoustic_acc = 0.0
        
        # Training loop
        for epoch in range(self.config.epochs):
            fusion_model.train()
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for vis, ac, labels in pbar:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
                
                with torch.no_grad():
                    vis_feats = visual_backbone(vis).squeeze(-1).squeeze(-1)
                
                # Modality dropout
                mask = (torch.rand(vis_feats.size(0), 1, device=self.device) > self.config.dropout_prob).float()
                vis_feats = vis_feats * mask
                
                optimizer.zero_grad()
                logits = fusion_model(vis_feats, ac)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                
                preds = torch.argmax(logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                pbar.set_postfix({"acc": f"{100*train_correct/train_total:.1f}%"})
            
            # Validation
            fusion_model.eval()
            val_fusion_correct = 0
            val_acoustic_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for vis, ac, labels in val_loader:
                    vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
                    vis_feats = visual_backbone(vis).squeeze(-1).squeeze(-1)
                    
                    # Fusion
                    logits_fusion = fusion_model(vis_feats, ac, mask_ratio=0.0)
                    preds_fusion = torch.argmax(logits_fusion, dim=1)
                    val_fusion_correct += (preds_fusion == labels).sum().item()
                    
                    # Acoustic only
                    zero_vis = torch.zeros_like(vis_feats)
                    logits_acoustic = fusion_model(zero_vis, ac, mask_ratio=0.0)
                    preds_acoustic = torch.argmax(logits_acoustic, dim=1)
                    val_acoustic_correct += (preds_acoustic == labels).sum().item()
                    
                    val_total += labels.size(0)
            
            fusion_acc = val_fusion_correct / val_total
            acoustic_acc = val_acoustic_correct / val_total
            
            wandb.log({
                "epoch": epoch + 1,
                "train_acc": train_correct / train_total,
                "val_fusion_acc": fusion_acc,
                "val_acoustic_acc": acoustic_acc,
            })
            
            # Save best acoustic-only model
            if acoustic_acc >= best_acoustic_acc:
                best_acoustic_acc = acoustic_acc
                self._save_model(fusion_model)
    
    def _get_dataset_path(self) -> str:
        """Get absolute path to dataset."""
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dataset", self.config.dataset)
        )
    
    def _save_model(self, model: nn.Module) -> None:
        """Save fusion model."""
        os.makedirs(self.config.weights_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.config.weights_dir, "fusion_best.pth"))
        with open(os.path.join(self.config.weights_dir, "model_config.json"), "w") as f:
            json.dump({"model_type": "fusion"}, f)


class TranslatorTrainer:
    """Trainer for Acoustic-to-Image Translator model."""
    
    def __init__(self, config: TranslatorConfig, device: torch.device):
        self.config = config
        self.device = device
    
    def train(self) -> None:
        """Full training pipeline for translator model."""
        from models.transformer_translator import AcousticToImageTransformer
        
        # Setup data
        aug_config = AugmentationConfig(enabled=self.config.with_aug)
        transform = create_visual_transform(aug_config)
        dataset_path = self._get_dataset_path()

        train_loader, val_loader = self._create_data_loaders(
            dataset_path, transform, self.config.batch_size
        )
        
        # Build model
        model = AcousticToImageTransformer(
            d_model=self.config.d_model, 
            patch_size=self.config.patch_size,
        ).to(self.device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        reconstruction_loss_fn = nn.MSELoss()
        classification_loss_fn = nn.CrossEntropyLoss()
        
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225],
        )
        
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.config.epochs):
            model.train()
            train_recon_loss = 0
            train_cls_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for vis, ac, labels in pbar:
                vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                gen_img, species_logits = model(ac)
                
                vis_01 = torch.stack([inv_normalize(v) for v in vis])
                recon_loss = reconstruction_loss_fn(gen_img, vis_01)
                cls_loss = classification_loss_fn(species_logits, labels)
                
                loss = 10.0 * recon_loss + cls_loss
                loss.backward()
                optimizer.step()
                
                train_recon_loss += recon_loss.item()
                train_cls_loss += cls_loss.item()
                
                preds = torch.argmax(species_logits, dim=1)
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)
                
                pbar.set_postfix({"recon": f"{recon_loss.item():.4f}", "cls": f"{cls_loss.item():.3f}"})
            
            # Validation
            model.eval()
            val_recon_loss = 0
            val_cls_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for vis, ac, labels in val_loader:
                    vis, ac, labels = vis.to(self.device), ac.to(self.device), labels.to(self.device)
                    gen_img, species_logits = model(ac)
                    
                    vis_01 = torch.stack([inv_normalize(v) for v in vis])
                    recon_loss = reconstruction_loss_fn(gen_img, vis_01)
                    cls_loss = classification_loss_fn(species_logits, labels)
                    
                    val_recon_loss += recon_loss.item()
                    val_cls_loss += cls_loss.item()
                    
                    preds = torch.argmax(species_logits, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            avg_val_loss = (val_recon_loss + val_cls_loss) / len(val_loader)
            
            wandb.log({
                "epoch": epoch + 1,
                "train_recon_loss": train_recon_loss / len(train_loader),
                "train_cls_loss": train_cls_loss / len(train_loader),
                "train_acc": train_correct / train_total,
                "val_recon_loss": val_recon_loss / len(val_loader),
                "val_cls_loss": val_cls_loss / len(val_loader),
                "val_acc": val_correct / val_total,
                "generated_image": wandb.Image(gen_img[0].cpu().detach().permute(1, 2, 0).numpy()),
            })
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self._save_model(model)
            
            scheduler.step()
    
    def _create_data_loaders(
        self,
        dataset_path: str,
        transform: transforms.Compose,
        batch_size: int,
    ) -> tuple:
        """Create train and validation loaders."""
        full_dataset = FishDataset(dataset_path, transform=transform)

        total_frames = len(full_dataset)
        train_split = int(0.8 * total_frames)
        indices = list(range(total_frames))
        train_indices = indices[:train_split]
        val_indices = indices[train_split:]

        train_ds = torch.utils.data.Subset(
            FishDataset(dataset_path, transform=transform, mode="train"),
            train_indices,
        )
        val_ds = torch.utils.data.Subset(
            FishDataset(dataset_path, transform=transform, mode="val"),
            val_indices,
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    
    def _get_dataset_path(self) -> str:
        """Get absolute path to dataset."""
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dataset", self.config.dataset)
        )
    
    def _save_model(self, model: nn.Module) -> None:
        """Save translator model."""
        os.makedirs(self.config.weights_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.config.weights_dir, "translator_best.pth"))


class MAETrainer:
    """Trainer for Acoustic Masked Autoencoder."""
    
    def __init__(self, config: MAEConfig, device: torch.device):
        self.config = config
        self.device = device
    
    def train(self) -> None:
        """Full training pipeline for MAE."""
        from models.mae import AcousticMAE
        
        # Setup data
        aug_config = AugmentationConfig(enabled=self.config.with_aug)
        transform = create_visual_transform(aug_config)
        dataset_path = self._get_dataset_path()

        dataset = FishDataset(
            dataset_path,
            transform=transform,
            mode="train",
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        
        # Build model
        model = AcousticMAE(mask_ratio=self.config.mask_ratio).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate, 
            weight_decay=0.05,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs
        )
        
        # Training loop
        for epoch in range(self.config.epochs):
            model.train()
            total_loss = 0
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
            for _, ac, _ in pbar:
                ac = ac.to(self.device)
                ac_img = ac.view(-1, 32, 256, 3).permute(0, 3, 1, 2)
                
                optimizer.zero_grad()
                pred, mask = model(ac)
                
                # Patchify target
                p1, p2 = 8, 16
                target = ac_img.unfold(2, p1, p1).unfold(3, p2, p2)
                target = target.permute(0, 2, 3, 1, 4, 5).contiguous()
                target = target.view(ac_img.size(0), -1, 3 * p1 * p2)
                
                loss = (pred - target) ** 2
                loss = loss.mean(dim=-1)
                loss = (loss * mask).sum() / mask.sum()
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            avg_loss = total_loss / len(loader)
            wandb.log({"epoch": epoch + 1, "mae_loss": avg_loss})
            
            if (epoch + 1) % 10 == 0:
                self._save_model(model, epoch + 1)
            
            scheduler.step()
    
    def _get_dataset_path(self) -> str:
        """Get absolute path to dataset."""
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "dataset", self.config.dataset)
        )
    
    def _save_model(self, model: nn.Module, epoch: int) -> None:
        """Save MAE model."""
        os.makedirs(self.config.weights_dir, exist_ok=True)
        torch.save(
            model.state_dict(), 
            os.path.join(self.config.weights_dir, f"mae_epoch_{epoch}.pth"),
        )
