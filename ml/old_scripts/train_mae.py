import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
import numpy as np
from dotenv import load_dotenv

from models.mae import AcousticMAE
from train import FishDataset

load_dotenv()

def train():
    parser = argparse.ArgumentParser(description="Train Acoustic Masked Autoencoder")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--dataset", type=str, default="easy", choices=["easy", "medium", "hard"],
                        help="Dataset difficulty level (default: easy)")
    parser.add_argument("--with-aug", action="store_true", default=False,
                        help="Enable data augmentation (default: disabled)")
    args = parser.parse_args()

    run = wandb.init(
        project="depth-learning",
        config={
            **vars(args),
            "dataset": f"Synthetic-Fish-Sim-V3-{args.dataset.capitalize()}",
            "difficulty": args.dataset,
            "augmentation": "disabled" if not args.with_aug else "enabled"
        },
        job_type="pretrain-mae"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", args.dataset))
    print(f"--- Using dataset difficulty: {args.dataset} ---")
    print(f"--- Data augmentation: {'disabled' if not args.with_aug else 'enabled'} ---")

    # Visual augmentation - configurable
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FishDataset(dataset_path, transform=transform, mode="train", use_augmentation=args.with_aug)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = AcousticMAE(mask_ratio=args.mask_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")
        
        for _, ac, _ in pbar:
            ac = ac.to(device)
            ac_img = ac.view(-1, 32, 256, 3).permute(0, 3, 1, 2)
            
            optimizer.zero_grad()
            pred, mask = model(ac)
            
            # Loss: MSE on masked patches only
            # pred: (B, L, C*p1*p2)
            # target: we need to patchify the original image
            
            # Patchify target
            p1, p2 = 8, 16
            target = ac_img.unfold(2, p1, p1).unfold(3, p2, p2)
            # target: (B, 3, 4, 16, 8, 16) -> (B, 4, 16, 3, 8, 16)
            target = target.permute(0, 2, 3, 1, 4, 5).contiguous()
            target = target.view(ac_img.size(0), -1, 3 * p1 * p2)
            
            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1) # (B, L)
            loss = (loss * mask).sum() / mask.sum() # average loss on masked patches
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(loader)
        wandb.log({"epoch": epoch+1, "mae_loss": avg_loss})
        
        if (epoch + 1) % 10 == 0:
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), f"weights/mae_epoch_{epoch+1}.pth")

    run.finish()

if __name__ == "__main__":
    train()
