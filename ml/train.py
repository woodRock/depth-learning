import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from glob import glob
from tqdm import tqdm
import math

class FishDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        # Find all visual frames
        self.visual_files = sorted(glob(os.path.join(data_dir, "*_visual.png")))
        self.transform = transform

    def __len__(self):
        return len(self.visual_files)

    def __getitem__(self, idx):
        vis_path = self.visual_files[idx]
        # Map visual frame to its paired acoustic frame
        ac_path = vis_path.replace("_visual.png", "_acoustic.png")
        
        # Load and ensure 3-channel RGB
        vis_img = Image.open(vis_path).convert("RGB")
        ac_img = Image.open(ac_path).convert("RGB")
        
        if self.transform:
            vis_img = self.transform(vis_img)
            ac_img = self.transform(ac_img)
            
        return vis_img, ac_img

class DualEncoderCLIP(nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        
        # 1. Visual Stream (ResNet18)
        self.vis_encoder = models.resnet18(weights=None)
        self.vis_encoder.fc = nn.Linear(self.vis_encoder.fc.in_features, embed_dim)
        
        # 2. Acoustic Stream (ResNet18)
        # Note: We use a 2D CNN because the echogram is treated as an image
        self.ac_encoder = models.resnet18(weights=None)
        self.ac_encoder.fc = nn.Linear(self.ac_encoder.fc.in_features, embed_dim)
        
        # 3. Learnable temperature parameter for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, vis, ac):
        # Extract features and L2 normalize
        vis_emb = F.normalize(self.vis_encoder(vis), p=2, dim=-1)
        ac_emb = F.normalize(self.ac_encoder(ac), p=2, dim=-1)
        
        # Compute cosine similarity scaled by temperature
        logit_scale = self.logit_scale.exp()
        logits_per_vis = logit_scale * vis_emb @ ac_emb.t()
        logits_per_ac = logits_per_vis.t()
        
        return logits_per_vis, logits_per_ac

def train():
    # Setup Device (Supports Apple Silicon MPS, CUDA, or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"--- Starting Training on {device} ---")
    
    # Standard ResNet Image Normalization
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize to fit ResNet inputs
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))
    print(f"Loading dataset from: {dataset_path}")
    
    dataset = FishDataset(dataset_path, transform=transform)
    print(f"Found {len(dataset)} paired samples.")
    
    if len(dataset) == 0:
        print("Error: No data found. Make sure you recorded data using the Bevy sim first.")
        return

    # Use drop_last=True because contrastive loss needs a batch > 1 to compare against
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    
    model = DualEncoderCLIP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for vis, ac in pbar:
            vis, ac = vis.to(device), ac.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits_per_vis, logits_per_ac = model(vis, ac)
            
            # InfoNCE Loss (Symmetric Cross Entropy)
            # The correct label for index i is i (the diagonal of the similarity matrix)
            labels = torch.arange(len(vis), device=device)
            loss_v = F.cross_entropy(logits_per_vis, labels)
            loss_a = F.cross_entropy(logits_per_ac, labels)
            
            # Combine losses
            loss = (loss_v + loss_a) / 2
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Average Loss: {avg_loss:.4f}")

    # Save the trained model
    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/fish_clip_model.pth")
    print("Training complete! Model saved to ml/weights/fish_clip_model.pth")

if __name__ == "__main__":
    train()
