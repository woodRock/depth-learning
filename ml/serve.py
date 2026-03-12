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
history_buffer = None
ping_count = 0
prediction_history = [] # For smoothing

@app.on_event("startup")
async def load_model():
    global model, decoder, history_buffer, ping_count, prediction_history
    weights_path = "weights/fish_clip_model.pth"
    decoder_path = "weights/decoder_model.pth"
    config_path = "weights/model_config.json"
    
    history_buffer = np.zeros((32, 256), dtype=np.float32)
    ping_count = 0
    prediction_history = []
    
    # Load JEPA Model
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

    # Load Decoder
    decoder = LatentDecoder()
    if os.path.exists(decoder_path):
        decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    decoder.to(device)
    decoder.eval()
    
    print(f"Server ready with {model_type.upper()} model + Smoothing.")

@app.post("/predict_acoustic")
async def predict(file: UploadFile = File(...)):
    global history_buffer, ping_count, prediction_history
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    img_np = np.array(image)
    latest_ping_v = img_np[:, -1, 0].astype(np.float32) / 255.0
    
    history_buffer = np.roll(history_buffer, -1, axis=0)
    history_buffer[-1, :] = latest_ping_v
    ping_count += 1
    
    # Warm-up phase
    if ping_count < 16:
        return {
            "predictions": [("Warm-up...", 0.0)],
            "generated_image": ""
        }

    history_flat = torch.from_numpy(history_buffer.flatten()).unsqueeze(0).to(device)
    
    with torch.no_grad():
        vis_latent, species_logits = model.forward_ac_to_vis_latent(history_flat)
        
        # Generation
        gen_img_tensor = decoder(vis_latent)
        gen_img_np = (gen_img_tensor.squeeze().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(gen_img_np)
        buffered = io.BytesIO()
        pil_img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # C. Direct Classification + Smoothing
        probs = F.softmax(species_logits, dim=1).squeeze()
        prediction_history.append(probs)
        if len(prediction_history) > 10: 
            prediction_history.pop(0)
        
        avg_probs = torch.stack(prediction_history).mean(dim=0)
        
        species_names = ["Kingfish", "Snapper", "Cod", "Empty"]
        scores = []
        for i, name in enumerate(species_names):
            scores.append((name, avg_probs[i].item()))
            
    return {
        "predictions": sorted(scores, key=lambda x: x[1], reverse=True),
        "generated_image": img_base64
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
