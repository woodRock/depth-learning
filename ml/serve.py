import os
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms, models
from PIL import Image
import io
import uvicorn
import math
import json
from glob import glob

# Re-define the model architecture
class DualEncoderCLIP(torch.nn.Module):
    def __init__(self, embed_dim=256):
        super().__init__()
        self.vis_encoder = models.resnet18(weights=None)
        self.vis_encoder.fc = torch.nn.Linear(self.vis_encoder.fc.in_features, embed_dim)
        self.ac_encoder = models.resnet18(weights=None)
        self.ac_encoder.fc = torch.nn.Linear(self.ac_encoder.fc.in_features, embed_dim)
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward_vis(self, vis):
        return F.normalize(self.vis_encoder(vis), p=2, dim=-1)

    def forward_ac(self, ac):
        return F.normalize(self.ac_encoder(ac), p=2, dim=-1)

# App Setup
app = FastAPI()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = None
prototypes = {}

@app.on_event("startup")
async def load_model():
    global model, prototypes
    weights_path = "weights/fish_clip_model.pth"
    if not os.path.exists(weights_path):
        print("ERROR: Model weights not found. Please train first.")
        return

    model = DualEncoderCLIP()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded on {device}. Learning prototypes from dataset...")

    # LEARN PROTOTYPES from visual examples in the dataset
    dataset_path = "../dataset"
    species_samples = {} # species -> list of embeddings

    meta_files = glob(os.path.join(dataset_path, "*_meta.json"))
    for meta_path in meta_files:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            species = meta["dominant_species"]
            
            # Load corresponding visual image
            vis_path = meta_path.replace("_meta.json", "_visual.png")
            if os.path.exists(vis_path):
                img = Image.open(vis_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.forward_vis(tensor)
                    if species not in species_samples:
                        species_samples[species] = []
                    species_samples[species].append(emb)

    # Average the embeddings to create a "Visual Prototype" for each species
    for species, embs in species_samples.items():
        if len(embs) > 0:
            avg_emb = torch.mean(torch.stack(embs), dim=0)
            prototypes[species] = F.normalize(avg_emb, p=2, dim=-1)
            print(f"  - Learned prototype for {species} (from {len(embs)} samples)")

@app.post("/predict_acoustic")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        ac_emb = model.forward_ac(tensor)
        scores = {}
        for species, proto_emb in prototypes.items():
            sim = (ac_emb @ proto_emb.t()).item()
            scores[species] = sim
            
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
