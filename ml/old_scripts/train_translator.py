import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np
import wandb
import json
from dotenv import load_dotenv

# Import our modular models
from models.transformer_translator import AcousticToImageTransformer
from train import FishDataset

# Load environment variables
load_dotenv()

def train():
    parser = argparse.ArgumentParser(description="Train Acoustic-to-Image Translator")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--dataset", type=str, default="easy", choices=["easy", "medium", "hard"],
                        help="Dataset difficulty level (default: easy)")
    parser.add_argument("--with-aug", action="store_true", default=False,
                        help="Enable data augmentation (default: disabled)")
    args = parser.parse_args()

    run = wandb.init(
        entity="victoria-university-of-wellington",
        project="depth-learning",
        config={
            "learning_rate": args.lr,
            "architecture": "AcousticToImageTransformer",
            "dataset": f"Synthetic-Fish-Sim-V3-{args.dataset.capitalize()}",
            "difficulty": args.dataset,
            "augmentation": "disabled" if not args.with_aug else "enabled",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "d_model": 256,
            "patch_size": 16,
        },
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting Translator Training on {device} ---")
    print(f"--- Using dataset difficulty: {args.dataset} ---")
    print(f"--- Data augmentation: {'disabled' if not args.with_aug else 'enabled'} ---")

    # Visual augmentation - configurable
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", args.dataset))
    full_dataset = FishDataset(dataset_path, transform=transform, use_augmentation=args.with_aug)

    total_frames = len(full_dataset)
    train_split = int(0.8 * total_frames)
    indices = list(range(total_frames))
    train_indices = indices[:train_split]
    val_indices = indices[train_split:]

    train_ds = torch.utils.data.Subset(
        FishDataset(dataset_path, transform=transform, mode="train", use_augmentation=args.with_aug),
        train_indices
    )
    val_ds = torch.utils.data.Subset(
        FishDataset(dataset_path, transform=transform, mode="val", use_augmentation=False),
        val_indices
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    model = AcousticToImageTransformer(d_model=config.d_model, patch_size=config.patch_size).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    # Loss functions
    reconstruction_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()

    # Inverse normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        train_recon_loss = 0
        train_cls_loss = 0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
            optimizer.zero_grad()
            
            gen_img, species_logits = model(ac)
            
            # Reconstruction loss (MSE)
            vis_01 = torch.stack([inv_normalize(v) for v in vis])
            recon_loss = reconstruction_loss_fn(gen_img, vis_01)
            
            cls_loss = classification_loss_fn(species_logits, labels)
            
            loss = 10.0 * recon_loss + cls_loss # Weight reconstruction more
            
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
                vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
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
        print(f"Epoch {epoch+1}: Val Recon: {val_recon_loss/len(val_loader):.4f} | Val Acc: {100*val_correct/val_total:.1f}%")
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_recon_loss": train_recon_loss / len(train_loader),
            "train_cls_loss": train_cls_loss / len(train_loader),
            "train_acc": train_correct / train_total,
            "val_recon_loss": val_recon_loss / len(val_loader),
            "val_cls_loss": val_cls_loss / len(val_loader),
            "val_acc": val_correct / val_total,
            # Log one sample image
            "generated_image": wandb.Image(gen_img[0]),
            "ground_truth": wandb.Image(vis_01[0])
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/translator_best.pth")
            print("--> Saved Best Model")

        scheduler.step()

    run.finish()

if __name__ == "__main__":
    train()
