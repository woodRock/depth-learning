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
from models.acoustic import ConvEncoder, TransformerEncoder
from models.jepa import CrossModalJEPA

# Load environment variables
load_dotenv()

class FishDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        all_visuals = sorted(glob(os.path.join(data_dir, "*_visual.png")))
        self.visual_files = []
        self.transform = transform
        self.species_map = {"Kingfish": 0, "Snapper": 1, "Cod": 2}

        # Ensure all required files exist for each sample
        print(f"Validating dataset in {data_dir}...")
        for v_path in all_visuals:
            h_path = v_path.replace("_visual.png", "_history.bin")
            m_path = v_path.replace("_visual.png", "_meta.json")
            if os.path.exists(h_path) and os.path.exists(m_path):
                self.visual_files.append(v_path)
        
        print(f"Found {len(self.visual_files)} complete samples out of {len(all_visuals)} visual files.")

    def __len__(self):
        return len(self.visual_files)

    def __getitem__(self, idx):
        vis_path = self.visual_files[idx]
        history_path = vis_path.replace("_visual.png", "_history.bin")
        meta_path = vis_path.replace("_visual.png", "_meta.json")
        
        vis_img = Image.open(vis_path).convert("RGB")
        
        # Load 32-ping history
        with open(history_path, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
            history_data = torch.from_numpy(raw_data[:32*256]).float() / 255.0
        
        # Load Label
        with open(meta_path, "r") as f:
            meta = json.load(f)
            label = self.species_map.get(meta["dominant_species"], 0)

        if self.transform:
            vis_img = self.transform(vis_img)
            
        return vis_img, history_data, label

def train():
    parser = argparse.ArgumentParser(description="Train Fish JEPA Model")
    parser.add_argument("--model", type=str, choices=["conv", "transformer"], default="conv", help="Acoustic encoder type")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    args = parser.parse_args()

    run = wandb.init(
        entity="victoria-university-of-wellington",
        project="depth-learning",
        config={
            "learning_rate": args.lr,
            "architecture": f"JEPA-{args.model}",
            "dataset": "Synthetic-Fish-Sim-V2",
            "epochs": args.epochs,
            "embed_dim": 256,
            "batch_size": 32,
        },
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting {args.model.upper()} Multi-Task Training on {device} ---")
    
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

    # TRAIN/VAL SPLIT (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    
    # Select Acoustic Encoder
    if args.model == "conv":
        ac_encoder = ConvEncoder(embed_dim=config.embed_dim)
    else:
        ac_encoder = TransformerEncoder(embed_dim=config.embed_dim)

    model = CrossModalJEPA(ac_encoder=ac_encoder, embed_dim=config.embed_dim).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")
        
        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
            optimizer.zero_grad()
            
            predicted_target, target_latent, species_logits = model(vis, ac)
            loss_jepa = (1.0 - F.cosine_similarity(predicted_target, target_latent, dim=-1)).mean()
            loss_cls = F.cross_entropy(species_logits, labels)
            loss = loss_jepa + (0.5 * loss_cls)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(species_logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{100*train_correct/train_total:.1f}%"})

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_sim = 0
        
        with torch.no_grad():
            for vis, ac, labels in val_loader:
                vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
                predicted_target, target_latent, species_logits = model(vis, ac)
                
                sim = F.cosine_similarity(predicted_target, target_latent, dim=-1).mean()
                loss_jepa = (1.0 - sim).mean()
                loss_cls = F.cross_entropy(species_logits, labels)
                
                val_loss += (loss_jepa + 0.5 * loss_cls).item()
                val_sim += sim.item()
                preds = torch.argmax(species_logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        avg_val_sim = val_sim / len(val_loader)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1} Results: Train Acc: {100*train_correct/train_total:.1f}% | Val Acc: {100*avg_val_acc:.1f}% | Val Sim: {avg_val_sim:.3f}")
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": avg_val_acc,
            "val_sim": avg_val_sim
        })

        # Save Best Model
        if avg_val_acc >= best_val_acc:
            best_val_acc = avg_val_acc
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/fish_clip_model.pth")
            with open("weights/model_config.json", "w") as f:
                json.dump({"model_type": args.model}, f)
            print(f"--> Saved New Best Model (Val Acc: {100*best_val_acc:.1f}%)")

    run.finish()

if __name__ == "__main__":
    train()
