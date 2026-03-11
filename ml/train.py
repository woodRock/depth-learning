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
from dotenv import load_dotenv

# Import our modular models
from models.acoustic import ConvEncoder, TransformerEncoder
from models.jepa import CrossModalJEPA

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
        history_path = vis_path.replace("_visual.png", "_history.bin")
        
        vis_img = Image.open(vis_path).convert("RGB")
        
        # Load 32-ping history (32 * 256 bytes)
        with open(history_path, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
            # Ensure it's exactly 32*256 (pad or crop if needed)
            expected_size = 32 * 256
            if len(raw_data) < expected_size:
                raw_data = np.pad(raw_data, (0, expected_size - len(raw_data)))
            elif len(raw_data) > expected_size:
                raw_data = raw_data[:expected_size]
                
            history_data = torch.from_numpy(raw_data).float() / 255.0
        
        if self.transform:
            vis_img = self.transform(vis_img)
            
        return vis_img, history_data

def train():
    parser = argparse.ArgumentParser(description="Train Fish CLIP Model")
    parser.add_argument("--model", type=str, choices=["conv", "transformer"], default="conv", help="Acoustic encoder type")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    run = wandb.init(
        entity="victoria-university-of-wellington",
        project="depth-learning",
        config={
            "learning_rate": args.lr,
            "architecture": f"CLIP-{args.model}",
            "dataset": "Synthetic-Fish-Sim-V2",
            "epochs": args.epochs,
            "embed_dim": 256,
            "batch_size": 32,
        },
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting {args.model.upper()} Training on {device} ---")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
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
    
    # Select Acoustic Encoder
    if args.model == "conv":
        ac_encoder = ConvEncoder(embed_dim=config.embed_dim)
    else:
        ac_encoder = TransformerEncoder(embed_dim=config.embed_dim)

    model = CrossModalJEPA(ac_encoder=ac_encoder, embed_dim=config.embed_dim).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for vis, ac in pbar:
            vis, ac = vis.to(device), ac.to(device)
            optimizer.zero_grad()
            
            predicted_target, target_latent = model(vis, ac)
            
            # Use Cosine Similarity for latent alignment (more stable for unit-normalized vectors)
            # Loss = 1 - CosineSimilarity (0 is perfect, 2 is worst)
            cosine_sim = F.cosine_similarity(predicted_target, target_latent, dim=-1)
            loss = (1.0 - cosine_sim).mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "sim": f"{cosine_sim.mean().item():.3f}"})
            
            wandb.log({
                "batch_loss": loss.item(),
                "avg_sim": cosine_sim.mean().item()
            })
            
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch + 1, "avg_loss": avg_loss, "lr": optimizer.param_groups[0]["lr"]})

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/fish_clip_model.pth")
    
    # Save a small config file so serve.py knows which model to load
    with open("weights/model_config.json", "w") as f:
        import json
        json.dump({"model_type": args.model}, f)

    print(f"Training complete! Model ({args.model}) saved to ml/weights/fish_clip_model.pth")
    run.finish()

if __name__ == "__main__":
    train()
