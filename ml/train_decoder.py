import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
from tqdm import tqdm
import wandb

from models.jepa import CrossModalJEPA
from models.acoustic import ConvEncoder, TransformerEncoder
from models.decoder import LatentDecoder

class ImageLatentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.visual_files = sorted(glob(os.path.join(data_dir, "*_visual.png")))
        self.transform = transform

    def __len__(self):
        return len(self.visual_files)

    def __getitem__(self, idx):
        img = Image.open(self.visual_files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img

def train_decoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Training Decoder on {device} ---")
    
    wandb.init(project="depth-learning", job_type="decoder-train")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # We need the pre-trained JEPA model to get the target latents
    jepa_weights = "weights/fish_clip_model.pth"
    config_path = "weights/model_config.json"
    
    if not os.path.exists(jepa_weights):
        print(f"Error: JEPA weights not found at {jepa_weights}. Please train JEPA first.")
        return

    # Determine model type
    import json
    model_type = "conv"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            model_type = cfg.get("model_type", "conv")

    print(f"Using {model_type.upper()} acoustic encoder for JEPA context.")
    ac_encoder = ConvEncoder() if model_type == "conv" else TransformerEncoder()
    
    jepa = CrossModalJEPA(ac_encoder=ac_encoder).to(device)
    
    # Load weights with strict=False or informative error
    try:
        jepa.load_state_dict(torch.load(jepa_weights, map_location=device))
    except RuntimeError as e:
        print(f"\nFATAL ERROR: Weight mismatch. Your {jepa_weights} likely contains old CLIP weights.")
        print("Please re-run JEPA training first: python3 ml/train.py --model transformer\n")
        return
        
    jepa.eval()
    
    decoder = LatentDecoder().to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    dataloader = DataLoader(ImageLatentDataset(dataset_path, transform=transform), batch_size=16, shuffle=True)
    
    for epoch in range(20):
        decoder.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/20")
        
        for img in pbar:
            img = img.to(device)
            
            # 1. Get Visual Latent from frozen ResNet
            with torch.no_grad():
                target_latent = jepa.target_encoder(img)
                target_latent = F.normalize(target_latent, p=2, dim=-1)
            
            # 2. Reconstruct image from latent
            optimizer.zero_grad()
            recon_img = decoder(target_latent)
            
            # Simple MSE on pixels (denormalize img for better visual loss if needed)
            # For simplicity, we compare against normalized pixels [0,1]
            # ResNet normalized input needs to be denormalized to compare with Sigmoid output
            
            loss = criterion(recon_img, (img * 0.225 + 0.456)) # Approximate denorm
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Log a sample to wandb
        if epoch % 5 == 0:
            sample_recon = recon_img[0].cpu().detach().permute(1, 2, 0).numpy()
            wandb.log({"reconstruction": wandb.Image(sample_recon)})

    torch.save(decoder.state_dict(), "weights/decoder_model.pth")
    print("Decoder saved to ml/weights/decoder_model.pth")

if __name__ == "__main__":
    train_decoder()
