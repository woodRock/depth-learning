import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
from glob import glob
from tqdm import tqdm
import math
import wandb
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class FishDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.visual_files = sorted(glob(os.path.join(data_dir, "*_visual.png")))
        self.transform = transform

    def __len__(self):
        return len(self.visual_files)

    def __getitem__(self, idx):
        vis_path = self.visual_files[idx]
        ac_path = vis_path.replace("_visual.png", "_acoustic.png")
        
        vis_img = Image.open(vis_path).convert("RGB")
        ac_img = Image.open(ac_path).convert("RGB")
        
        if self.transform:
            vis_img = self.transform(vis_img)
            ac_img = self.transform(ac_img)
            
        return vis_img, ac_img

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DualEncoderCLIP(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # 1. Visual Stream (Pre-trained)
        self.vis_backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.vis_backbone.fc.in_features
        self.vis_backbone.fc = nn.Identity() # Remove default head
        self.vis_proj = ProjectionHead(in_features, embed_dim)
        
        # 2. Acoustic Stream (Pre-trained)
        self.ac_backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.ac_backbone.fc = nn.Identity()
        self.ac_proj = ProjectionHead(in_features, embed_dim)
        
        # 3. Temperature (Initialized to a sensible value)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, vis, ac):
        vis_feat = self.vis_backbone(vis)
        ac_feat = self.ac_backbone(ac)
        
        vis_emb = F.normalize(self.vis_proj(vis_feat), p=2, dim=-1)
        ac_emb = F.normalize(self.ac_proj(ac_feat), p=2, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        logits_per_vis = logit_scale * vis_emb @ ac_emb.t()
        logits_per_ac = logits_per_vis.t()
        
        return logits_per_vis, logits_per_ac

def train():
    run = wandb.init(
        entity="victoria-university-of-wellington",
        project="depth-learning",
        config={
            "learning_rate": 5e-4,
            "architecture": "Dual-ResNet18-MLP",
            "dataset": "Synthetic-Fish-Sim-V2",
            "epochs": 50,
            "embed_dim": 256,
            "batch_size": 32,
        },
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting Refined Training on {device} ---")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(), # Add augmentation
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    dataset = FishDataset(dataset_path, transform=transform)
    
    if len(dataset) < config.batch_size:
        print(f"Error: Not enough data. Found {len(dataset)} samples.")
        return

    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    
    model = DualEncoderCLIP(embed_dim=config.embed_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for vis, ac in pbar:
            vis, ac = vis.to(device), ac.to(device)
            optimizer.zero_grad()
            
            logits_per_vis, logits_per_ac = model(vis, ac)
            labels = torch.arange(len(vis), device=device)
            
            loss_v = F.cross_entropy(logits_per_vis, labels)
            loss_a = F.cross_entropy(logits_per_ac, labels)
            loss = (loss_v + loss_a) / 2
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "temp": f"{model.logit_scale.exp().item():.2f}"})
            
            wandb.log({
                "batch_loss": loss.item(), 
                "temperature": model.logit_scale.exp().item()
            })
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/fish_clip_model.pth")
    print("Training complete! Model saved to ml/weights/fish_clip_model.pth")
    run.finish()

if __name__ == "__main__":
    train()
