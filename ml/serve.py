import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms, models
from PIL import Image
import io
import uvicorn
import numpy as np
import math

# Re-define the model architecture
class DualEncoderCLIP(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.vis_encoder = models.resnet18(weights=None)
        self.vis_encoder.fc = torch.nn.Linear(self.vis_encoder.fc.in_features, embed_dim)
        self.ac_encoder = models.resnet18(weights=None)
        self.ac_encoder.fc = torch.nn.Linear(self.ac_encoder.fc.in_features, embed_dim)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward_ac(self, ac):
        return F.normalize(self.ac_encoder(ac), p=2, dim=-1)

# App Setup
app = FastAPI()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Transformation (Same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Global model & prototype storage
model = None
prototypes = {} # Species Name -> Embedding

@app.on_event("startup")
async def load_model():
    global model, prototypes
    weights_path = "weights/fish_clip_model.pth"
    if not os.path.exists(weights_path):
        print(f"ERROR: Model weights not found at {weights_path}. Please train first.")
        return

    model = DualEncoderCLIP()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded and serving on {device}!")

    # Pre-calculate species "prototypes"
    # In a real CLIP-like setup, we'd use text or averaged visual images.
    # For now, we'll use placeholder labels for the sim to identify.
    species_list = ["Kingfish", "Snapper", "Cod"]
    # (In a more advanced setup, you'd feed in representative visual images here)
    for s in species_list:
        prototypes[s] = torch.randn(1, 256).to(device) # Placeholder until first visual feed

@app.post("/predict_acoustic")
async def predict(file: UploadFile = File(...)):
    """Receives an acoustic ping image and returns species probabilities."""
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ac_emb = model.forward_ac(tensor)
        
        # Compare against prototypes
        scores = {}
        for species, proto_emb in prototypes.items():
            # Cosine similarity
            sim = (ac_emb @ proto_emb.t()).item()
            scores[species] = sim
            
    # Return sorted results
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
