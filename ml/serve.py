import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File
from torchvision import transforms
from PIL import Image
import io
import uvicorn
import json
import numpy as np
import base64
from glob import glob

# Import modular models
from models.acoustic import ConvEncoder, TransformerEncoder
from models.jepa import CrossModalJEPA
from models.decoder import LatentDecoder

app = FastAPI()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = None
decoder = None
prototypes = {}
history_buffer = None

@app.on_event("startup")
async def load_model():
    global model, decoder, prototypes, history_buffer
    weights_path = "weights/fish_clip_model.pth"
    decoder_path = "weights/decoder_model.pth"
    config_path = "weights/model_config.json"
    
    history_buffer = np.zeros((32, 256), dtype=np.float32)
    
    # 1. Load JEPA Model
    model_type = "conv"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            model_type = cfg.get("model_type", "conv")

    ac_encoder = ConvEncoder() if model_type == "conv" else TransformerEncoder()
    model = CrossModalJEPA(ac_encoder=ac_encoder)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 2. Load Decoder
    decoder = LatentDecoder()
    if os.path.exists(decoder_path):
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        print("Decoder loaded successfully.")
    decoder.to(device)
    decoder.eval()
    
    # 3. Learn Prototypes
    dataset_path = "../dataset"
    species_samples = {} 
    meta_files = glob(os.path.join(dataset_path, "*_meta.json"))
    for meta_path in meta_files:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            species = meta["dominant_species"]
            vis_path = meta_path.replace("_meta.json", "_visual.png")
            if os.path.exists(vis_path):
                img = Image.open(vis_path).convert("RGB")
                tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.target_encoder(tensor)
                    if species not in species_samples:
                        species_samples[species] = []
                    species_samples[species].append(emb)

    for species, embs in species_samples.items():
        if len(embs) > 0:
            avg_emb = torch.mean(torch.stack(embs), dim=0)
            prototypes[species] = F.normalize(avg_emb, p=2, dim=-1)

@app.post("/predict_acoustic")
async def predict(file: UploadFile = File(...)):
    global history_buffer
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    img_np = np.array(image)
    latest_ping_v = img_np[:, -1, 0].astype(np.float32) / 255.0
    history_buffer = np.roll(history_buffer, -1, axis=0)
    history_buffer[-1, :] = latest_ping_v
    
    history_flat = torch.from_numpy(history_buffer.flatten()).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # A. Predict Latent
        vis_latent = model.forward_ac_to_vis_latent(history_flat)
        
        # B. Generate Image from Latent
        gen_img_tensor = decoder(vis_latent) # [1, 3, 224, 224]
        
        # C. Convert to base64 string for Bevy
        gen_img_np = (gen_img_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(gen_img_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # D. Calculate Similarity Scores
        scores = {}
        for species, proto_emb in prototypes.items():
            sim = (vis_latent @ proto_emb.t()).item()
            scores[species] = sim
            
    return {
        "predictions": sorted(scores.items(), key=lambda x: x[1], reverse=True),
        "generated_image": img_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
