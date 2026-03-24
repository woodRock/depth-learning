import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
import wandb
import json
import random
from dotenv import load_dotenv

from models.fusion import MaskedAttentionFusion
from train import FishDataset

load_dotenv()

def train():
    parser = argparse.ArgumentParser(description="Train Masked Attention Fusion with Modality Dropout")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.5, help="Probability of dropping the visual modality")
    args = parser.parse_args()

    run = wandb.init(
        project="depth-learning",
        config=args,
        job_type="train-fusion"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # 1. Visual Encoder (Pre-trained ResNet50)
    visual_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    visual_backbone = nn.Sequential(*list(visual_model.children())[:-1])
    visual_backbone.to(device)
    visual_backbone.eval()
    
    # 2. Fusion Model
    fusion_model = MaskedAttentionFusion(d_model=256, nhead=8, num_classes=4).to(device)
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = FishDataset(dataset_path, transform=transform, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    val_dataset = FishDataset(dataset_path, transform=transform, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_acoustic_acc = 0.0

    for epoch in range(args.epochs):
        fusion_model.train()
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
            
            with torch.no_grad():
                vis_feats = visual_backbone(vis).squeeze(-1).squeeze(-1) # (B, 2048)
            
            # MODALITY DROPOUT: Zero out visual features for some samples
            # This forces the model to learn to classify using ONLY acoustic data too.
            mask = (torch.rand(vis_feats.size(0), 1, device=device) > args.dropout_prob).float()
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

        # Validation (Test both scenarios)
        fusion_model.eval()
        val_fusion_correct = 0
        val_acoustic_only_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for vis, ac, labels in val_loader:
                vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
                vis_feats = visual_backbone(vis).squeeze(-1).squeeze(-1)
                
                # Case A: Full Fusion (Images + Sonar)
                logits_fusion = fusion_model(vis_feats, ac, mask_ratio=0.0)
                preds_fusion = torch.argmax(logits_fusion, dim=1)
                val_fusion_correct += (preds_fusion == labels).sum().item()
                
                # Case B: Acoustic Only (Zeros for Visual) - THIS MATCHES DEPLOYMENT
                zero_vis = torch.zeros_like(vis_feats)
                logits_acoustic = fusion_model(zero_vis, ac, mask_ratio=0.0)
                preds_acoustic = torch.argmax(logits_acoustic, dim=1)
                val_acoustic_only_correct += (preds_acoustic == labels).sum().item()
                
                val_total += labels.size(0)

        fusion_acc = val_fusion_correct / val_total
        acoustic_acc = val_acoustic_only_correct / val_total
        
        print(f"Epoch {epoch+1}: Val Fusion Acc: {100*fusion_acc:.1f}% | Val Acoustic Acc: {100*acoustic_acc:.1f}%")
        wandb.log({
            "epoch": epoch+1, 
            "train_acc": train_correct/train_total, 
            "val_fusion_acc": fusion_acc,
            "val_acoustic_acc": acoustic_acc
        })

        # Save the model that performs best on ACOUSTIC-ONLY data
        if acoustic_acc >= best_acoustic_acc:
            best_acoustic_acc = acoustic_acc
            os.makedirs("weights", exist_ok=True)
            torch.save(fusion_model.state_dict(), "weights/fusion_best.pth")
            with open("weights/model_config.json", "w") as f:
                json.dump({"model_type": "fusion"}, f)
            print(f"--> Saved Best Model (Acoustic Val Acc: {100*best_acoustic_acc:.1f}%)")

    run.finish()

if __name__ == "__main__":
    train()
