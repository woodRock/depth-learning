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
from models.lstm import AcousticLSTM
from models.ast import AcousticAST
from models.jepa import CrossModalJEPA

# Load environment variables
load_dotenv()

class FishDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode="train"):
        self.data_dir = data_dir
        self.mode = mode
        all_visuals = sorted(glob(os.path.join(data_dir, "*_visual.png")))
        self.transform = transform
        self.species_map = {"Kingfish": 0, "Snapper": 1, "Cod": 2, "Empty": 3}

        valid_samples = []
        # Filter complete samples first
        for v_path in all_visuals:
            h_path = v_path.replace("_visual.png", "_history.bin")
            m_path = v_path.replace("_visual.png", "_meta.json")
            if os.path.exists(h_path) and os.path.exists(m_path):
                valid_samples.append(v_path)

        # CLASS BALANCING (Only for Training)
        if mode == "train":
            fish_samples = []
            empty_samples = []
            for v_path in valid_samples:
                m_path = v_path.replace("_visual.png", "_meta.json")
                with open(m_path, "r") as f:
                    if json.load(f)["dominant_species"] == "Empty":
                        empty_samples.append(v_path)
                    else:
                        fish_samples.append(v_path)

            # Keep all fish, but limit empty frames to match total fish count
            np.random.shuffle(empty_samples)
            num_to_keep = min(len(empty_samples), len(fish_samples))
            self.visual_files = fish_samples + empty_samples[:num_to_keep]
            print(f"Balanced Dataset: {len(fish_samples)} Fish frames, {num_to_keep} Empty frames.")
        else:
            self.visual_files = valid_samples

    def __len__(self):
        return len(self.visual_files)

    def __getitem__(self, idx):
        vis_path = self.visual_files[idx]
        history_path = vis_path.replace("_visual.png", "_history.bin")
        meta_path = vis_path.replace("_visual.png", "_meta.json")

        vis_img = Image.open(vis_path).convert("RGB")

        # Load 32-ping history (32, 256, 3)
        with open(history_path, "rb") as f:
            raw_data = np.frombuffer(f.read(), dtype=np.uint8).copy()
            # Ensure it's exactly 32 * 256 * 3
            expected = 32 * 256 * 3
            data = raw_data[:expected].reshape(32, 256, 3).astype(np.float32) / 255.0

        # ACOUSTIC AUGMENTATION - Enhanced
        if self.mode == "train":
            # 1. Temporal Jitter (shift pings)
            shift = np.random.randint(-3, 4)
            data = np.roll(data, shift, axis=0)

            # 2. Spatial Flip (Flip Depth Axis)
            if np.random.random() > 0.5:
                data = np.flip(data, axis=1).copy()

            # 3. Channel Gain Variation (Simulate sensor calibration noise)
            gain = 1.0 + (np.random.rand(3) - 0.5) * 0.15  # +/- 7.5% gain per freq
            data = (data * gain).clip(0, 1)

            # 4. Speckle Noise (realistic sonar noise)
            noise_scale = np.random.uniform(0.01, 0.04)
            noise = np.random.normal(0, noise_scale, data.shape)
            data = (data + noise).clip(0, 1)

            # 5. PING DROPPING (simulates transmission loss)
            drop_count = np.random.randint(0, 5)
            if drop_count > 0:
                indices = np.random.choice(32, drop_count, replace=False)
                data[indices, :, :] = 0.0
            
            # 6. Temporal Masking (drop consecutive pings)
            if np.random.random() > 0.6:
                mask_start = np.random.randint(0, 28)
                mask_len = np.random.randint(2, 5)
                data[mask_start:mask_start + mask_len, :, :] = 0.0
            
            # 7. Depth-dependent noise (more noise at depth)
            depth_gradient = np.linspace(0.5, 1.5, 256).reshape(1, -1, 1)
            depth_noise = np.random.normal(0, 0.02, data.shape) * depth_gradient
            data = (data + depth_noise).clip(0, 1)
            
            # 8. Random occlusion (simulate shadows)
            if np.random.random() > 0.7:
                occlude_depth = np.random.randint(50, 200)
                occlude_height = np.random.randint(10, 30)
                data[occlude_depth:occlude_depth + occlude_height, :, :] *= 0.3

        history_tensor = torch.from_numpy(data.flatten()).float()

        # Load Label
        with open(meta_path, "r") as f:
            meta = json.load(f)
            label = self.species_map.get(meta["dominant_species"], 3)  # Default to Empty

        if self.transform:
            vis_img = self.transform(vis_img)

        return vis_img, history_tensor, label

def train():
    parser = argparse.ArgumentParser(description="Train Fish JEPA Model")
    parser.add_argument("--model", type=str, choices=["conv", "transformer", "lstm", "ast"], default="transformer", help="Acoustic encoder type")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--use-focal-loss", action="store_true", default=True, help="Use focal loss for classification")
    args = parser.parse_args()

    run = wandb.init(
        entity="victoria-university-of-wellington",
        project="depth-learning",
        config={
            "learning_rate": args.lr,
            "architecture": f"JEPA-{args.model}",
            "dataset": "Synthetic-Fish-Sim-V3-Improved",
            "epochs": args.epochs,
            "embed_dim": 256,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "use_focal_loss": args.use_focal_loss,
        },
    )
    config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting {args.model.upper()} Multi-Task Training on {device} ---")

    # Enhanced visual augmentation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
        transforms.RandomRotation(degrees=5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    # DATASET CREATION
    full_dataset = FishDataset(dataset_path, transform=transform)

    if len(full_dataset) < config.batch_size:
        print(f"Error: Not enough data. Found {len(full_dataset)} samples.")
        return

    # BLOCK SPLIT (No Shuffling before split!)
    # First 80% of time-series for training, last 20% for validation
    total_frames = len(full_dataset)
    train_split = int(0.8 * total_frames)

    indices = list(range(total_frames))
    train_indices = indices[:train_split:2]  # Subsample every 2nd frame for diversity
    val_indices = indices[train_split:]

    train_ds = torch.utils.data.Subset(FishDataset(dataset_path, transform=transform, mode="train"), train_indices)
    val_ds = torch.utils.data.Subset(FishDataset(dataset_path, transform=transform, mode="val"), val_indices)

    print(f"Dataset Split: {len(train_ds)} Train samples, {len(val_ds)} Val samples.")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    # Select Acoustic Encoder
    if args.model == "conv":
        ac_encoder = ConvEncoder(embed_dim=config.embed_dim)
    elif args.model == "transformer":
        ac_encoder = TransformerEncoder(embed_dim=config.embed_dim)
    elif args.model == "lstm":
        # LSTM output is (B, hidden_dim * 2), so we project it to embed_dim
        ac_encoder = AcousticLSTM(input_dim=768, hidden_dim=256, num_classes=config.embed_dim)
    elif args.model == "ast":
        # AST CLS output is (B, d_model), we use that as embedding
        ac_encoder = AcousticAST(d_model=config.embed_dim, num_classes=config.embed_dim)
    else:
        # Fallback for old default behavior or if logic is simplified
        ac_encoder = TransformerEncoder(embed_dim=config.embed_dim)

    model = CrossModalJEPA(
        ac_encoder=ac_encoder, 
        embed_dim=config.embed_dim,
        use_focal_loss=args.use_focal_loss
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    
    # Cosine annealing with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    warmup_epochs = 5

    best_val_acc = 0.0
    best_val_sim = 0.0

    for epoch in range(config.epochs):
        # Adjust learning rate for warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.learning_rate * (epoch + 1) / warmup_epochs

        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0
        train_loss_jepa = 0
        train_loss_cls = 0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]")

        for vis, ac, labels in pbar:
            vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
            optimizer.zero_grad()

            predicted_target, target_latent, species_logits = model(vis, ac)
            
            # Use compute_loss method which handles focal loss
            loss, loss_jepa, loss_cls = model.compute_loss(predicted_target, target_latent, species_logits, labels)

            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            train_loss += loss.item()
            train_loss_jepa += loss_jepa.item()
            train_loss_cls += loss_cls.item()
            preds = torch.argmax(species_logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.3f}", 
                "acc": f"{100*train_correct/train_total:.1f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        val_loss_jepa = 0
        val_loss_cls = 0
        val_correct = 0
        val_total = 0
        val_sim = 0

        # Per-class accuracy tracking
        class_correct = [0, 0, 0, 0]
        class_total = [0, 0, 0, 0]

        with torch.no_grad():
            for vis, ac, labels in val_loader:
                vis, ac, labels = vis.to(device), ac.to(device), labels.to(device)
                predicted_target, target_latent, species_logits = model(vis, ac)

                loss, loss_jepa, loss_cls = model.compute_loss(predicted_target, target_latent, species_logits, labels)

                val_loss += loss.item()
                val_loss_jepa += loss_jepa.item()
                val_loss_cls += loss_cls.item()
                
                sim = F.cosine_similarity(predicted_target, target_latent, dim=-1).mean()
                val_sim += sim.item()
                
                preds = torch.argmax(species_logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Per-class accuracy
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    if preds[i] == label:
                        class_correct[label] += 1
                    class_total[label] += 1

        avg_train_loss = train_loss / len(train_loader)
        avg_train_loss_jepa = train_loss_jepa / len(train_loader)
        avg_train_loss_cls = train_loss_cls / len(train_loader)
        avg_train_acc = train_correct / train_total
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_loss_jepa = val_loss_jepa / len(val_loader)
        avg_val_loss_cls = val_loss_cls / len(val_loader)
        avg_val_acc = val_correct / val_total
        avg_val_sim = val_sim / len(val_loader)

        scheduler.step()

        # Per-class accuracy logging
        class_acc = {
            f"val_acc_{name}": class_correct[i] / max(class_total[i], 1) * 100
            for i, name in enumerate(["Kingfish", "Snapper", "Cod", "Empty"])
        }
        
        print(f"Epoch {epoch+1} Results: "
              f"Train Acc: {100*avg_train_acc:.1f}% | "
              f"Val Acc: {100*avg_val_acc:.1f}% | "
              f"Val Sim: {avg_val_sim:.3f}")
        for name, acc in class_acc.items():
            print(f"  {name}: {acc:.1f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_loss_jepa": avg_train_loss_jepa,
            "train_loss_cls": avg_train_loss_cls,
            "train_acc": avg_train_acc,
            "val_loss": avg_val_loss,
            "val_loss_jepa": avg_val_loss_jepa,
            "val_loss_cls": avg_val_loss_cls,
            "val_acc": avg_val_acc,
            "val_sim": avg_val_sim,
            "lr": optimizer.param_groups[0]["lr"],
            **class_acc
        })

        # Save Best Model (by combined accuracy + similarity)
        combined_score = avg_val_acc * 0.7 + avg_val_sim * 0.3
        if combined_score >= best_val_acc * 0.7 + best_val_sim * 0.3:
            best_val_acc = avg_val_acc
            best_val_sim = avg_val_sim
            os.makedirs("weights", exist_ok=True)
            torch.save(model.state_dict(), "weights/fish_clip_model.pth")
            with open("weights/model_config.json", "w") as f:
                json.dump({"model_type": args.model, "config": dict(config)}, f)
            print(f"--> Saved New Best Model (Val Acc: {100*best_val_acc:.1f}%, Sim: {best_val_sim:.3f})")

    run.finish()

if __name__ == "__main__":
    train()
