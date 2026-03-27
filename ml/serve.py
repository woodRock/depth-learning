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
from data import FishDataset
from models.acoustic import ConvEncoder, TransformerEncoder
from models.jepa import CrossModalJEPA
from models.decoder import LatentDecoder
from models.transformer_translator import AcousticToImageTransformer
from models.lstm import AcousticLSTM
from models.ast import AcousticAST
from models.fusion import MaskedAttentionFusion
from models.mae import AcousticMAE
from models.lewm_multilabel import LeWorldModelMultiLabel

app = FastAPI()
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = None
decoder = None
model_type = "conv"
history_buffer = None
ping_count = 0
prediction_history = [] 

@app.on_event("startup")
async def load_model():
    global model, decoder, model_type, history_buffer, ping_count, prediction_history
    weights_path = "weights/fish_clip_model.pth"
    decoder_path = "weights/decoder_model.pth"
    translator_path = "weights/translator_best.pth"
    config_path = "weights/model_config.json"
    
    # 3-channel history (32 pings, 256 depth, 3 frequencies)
    history_buffer = np.zeros((32, 256, 3), dtype=np.float32)
    ping_count = 0
    prediction_history = []
    
    # Load JEPA Model or others
    model_type = "conv"
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = json.load(f)
            model_type = cfg.get("model_type", "conv")

    if model_type == "translator":
        model = AcousticToImageTransformer(d_model=256, patch_size=16)
        if os.path.exists(translator_path):
            model.load_state_dict(torch.load(translator_path, map_location=device))
        print("Loaded Translator Model.")
    elif model_type == "mae":
        model = AcousticMAE()
        # Search for any mae weights
        mae_weights = glob("weights/mae_epoch_*.pth")
        if mae_weights:
            latest = sorted(mae_weights)[-1]
            model.load_state_dict(torch.load(latest, map_location=device))
            print(f"Loaded MAE Model from {latest}")
        elif os.path.exists("weights/mae_best.pth"):
            model.load_state_dict(torch.load("weights/mae_best.pth", map_location=device))
            print("Loaded MAE Model.")
    elif model_type == "lstm":
        model = AcousticLSTM(input_dim=768, hidden_dim=256, num_classes=4)
        if os.path.exists("weights/lstm_best.pth"): # Assuming a name
            model.load_state_dict(torch.load("weights/lstm_best.pth", map_location=device))
        print("Loaded LSTM Model.")
    elif model_type == "ast":
        model = AcousticAST(d_model=256, num_classes=4)
        if os.path.exists("weights/ast_best.pth"):
            model.load_state_dict(torch.load("weights/ast_best.pth", map_location=device))
        print("Loaded AST Model.")
    elif model_type == "fusion":
        model = MaskedAttentionFusion(d_model=256, nhead=8, num_classes=4)
        if os.path.exists("weights/fusion_best.pth"):
            model.load_state_dict(torch.load("weights/fusion_best.pth", map_location=device))
        print("Loaded Fusion Model.")
    elif model_type == "lewm":
        # LeWorldModel: Acoustic-only, with world reconstruction decoder
        # Load task from config
        task = "presence"  # Default
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                task = cfg.get("task", "presence")
        
        model = LeWorldModelMultiLabel(
            embed_dim=256,
            num_layers=8,  # Match saved weights
            num_heads=8,
            mlp_ratio=4.0,
            drop=0.1,
            n_classes=4,
            use_classifier=True,
            use_decoder=True,  # Enable world reconstruction for visualization
            task=task  # Load correct task head
        )
        if os.path.exists(weights_path):
            # Load with strict=False to handle minor architecture differences
            model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False), strict=False)
            print("Loaded LeWorldModel (LeWM) with World Decoder.")
            print("  Note: Using auto-detect for timestep size (32 or 64)")
            print("  World reconstruction enabled for visualization")
            print(f"  Task: {task.upper()}")
        else:
            print("Warning: No LeWM weights found, using random initialization")
    else:
        ac_encoder = ConvEncoder() if model_type == "conv" else TransformerEncoder()
        model = CrossModalJEPA(ac_encoder=ac_encoder)
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded {model_type.upper()} JEPA Model.")

    model.to(device)
    model.eval()

    # Load Decoder (Only needed for JEPA models with reconstruction)
    if model_type in ["conv", "transformer"]:
        decoder = LatentDecoder()
        if os.path.exists(decoder_path):
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
        decoder.to(device)
        decoder.eval()
    
    print(f"Server ready with {model_type.upper()} architecture.")

@app.post("/predict_acoustic")
async def predict(file: UploadFile = File(...)):
    global prediction_history
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')  # Ensure RGB, not RGBA

    img_np = np.array(image)  # Echogram format from simulation: (256 depth, 512 pings, 3 channels)
    
    print(f"DEBUG: echogram shape: {img_np.shape}")

    # Simulation sends full 512-ping scrolling echogram
    # Training data is 32 pings wide
    # Extract the rightmost (most recent) 32 pings to match training
    if img_np.shape[1] > 32:
        img_np = img_np[:, -32:, :]  # Take last 32 pings
        print(f"DEBUG: cropped to last 32 pings: {img_np.shape}")
    
    # The echogram is now (256, 32, 3) = (depth, pings, channels)
    # Transpose to match training format: (32 pings, 256 depth, 3 channels)
    history_32 = img_np.transpose(1, 0, 2).astype(np.float32) / 255.0
    
    print(f"DEBUG: history_32 shape: {history_32.shape}")

    # Flatten to (1, 32 * 256 * 3) = (1, 24576)
    history_flat = torch.from_numpy(history_32.flatten()).unsqueeze(0).to(device)
    print(f"DEBUG: history_flat shape: {history_flat.shape}")
    
    with torch.no_grad():
        if model_type == "translator":
            gen_img_tensor, species_logits = model(history_flat)
        elif model_type == "mae":
            # RECONSTRUCTION TEST
            pred, mask = model(history_flat)
            # Unpatchify (1, 64, 384) -> (1, 3, 32, 256)
            p1, p2 = 8, 16
            h, w = 4, 16
            gen_img_tensor = pred.view(1, h, w, 3, p1, p2)
            gen_img_tensor = gen_img_tensor.permute(0, 3, 1, 4, 2, 5).contiguous()
            gen_img_tensor = gen_img_tensor.view(1, 3, 32, 256)
            
            # Resize to 224x224 so Bevy doesn't panic
            gen_img_tensor = F.interpolate(gen_img_tensor, size=(224, 224), mode="bilinear", align_corners=False)
            
            # MAE doesn't classify, so return empty/neutral logits
            species_logits = torch.zeros((1, 4)).to(device)
        elif model_type == "lstm":
            species_logits = model(history_flat)
            gen_img_tensor = None # No reconstruction
        elif model_type == "ast":
            species_logits = model(history_flat)
            gen_img_tensor = None
        elif model_type == "fusion":
            # Acoustic-only inference for Fusion model
            # We use zeros for visual features, which is now supported by Modality Dropout training
            vis_feats = torch.zeros((1, 2048)).to(device)
            ac_img = history_flat.view(-1, 32, 256, 3).permute(0, 3, 1, 2)
            species_logits = model(vis_feats, ac_img, mask_ratio=0.0)
            gen_img_tensor = None
        elif model_type == "lewm":
            # LeWorldModel: Acoustic-only with world reconstruction
            _, _, species_logits, recon_img = model(history_flat)
            
            # Get reconstruction if decoder is enabled
            if recon_img is not None:
                # Reconstruction is already (B, 3, 224, 224), clamp and convert
                # AUTO-NORMALIZATION: If image is very dark, boost it for visibility
                # This helps if the model was trained with normalized targets (bug fix applied in trainers.py)
                curr_max = recon_img.max().item()
                if curr_max > 0 and curr_max < 0.3:
                    print(f"DEBUG: Boosting dark LeWM reconstruction (max={curr_max:.3f})")
                    recon_img = recon_img / (curr_max + 1e-8)
                
                gen_img_tensor = recon_img.cpu().clamp(0, 1)
            else:
                gen_img_tensor = None  # Decoder not enabled or no reconstruction
        else:
            # JEPA
            vis_latent, species_logits = model.forward_ac_to_vis_latent(history_flat)
            gen_img_tensor = decoder(vis_latent)
        
        # Generation encoding
        if gen_img_tensor is not None:
            # Use [0] to safely remove batch dim without squeezing single-channel images if they exist
            gen_img_np = (gen_img_tensor[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(gen_img_np)
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
        else:
            # Placeholder or empty if model doesn't support reconstruction
            img_base64 = ""

        # Classification + Smoothing
        # Support both presence (sigmoid) and counting (tanh-scaled)
        if model_type == "lewm":
            # Check model task from model itself
            if hasattr(model, 'task') and model.task == "counting":
                # Counting: model output is tanh-scaled (0-30 range)
                probs = species_logits.squeeze()
                # Apply same tanh scaling as training
                probs = torch.tanh(probs / 5.0) * 30.0
                probs = probs.clamp(0, 30)
            else:
                # Presence/absence: sigmoid for independent probabilities
                probs = torch.sigmoid(species_logits).squeeze()
        else:
            # Other models: softmax for single-label
            probs = F.softmax(species_logits, dim=1).squeeze()

        prediction_history.append(probs)
        if len(prediction_history) > 10:
            prediction_history.pop(0)

        avg_probs = torch.stack(prediction_history).mean(dim=0)
        print(f"DEBUG: raw avg_probs: {avg_probs.tolist()}")

        # Use SPECIES_MAP for consistent ordering
        # Sort by value to get [0, 1, 2, 3] order
        species_names = [name for name, _ in sorted(FishDataset.SPECIES_MAP.items(), key=lambda x: x[1])]
        scores = []
        for i, name in enumerate(species_names):
            scores.append((name, avg_probs[i].item()))
            
    return {
        "predictions": sorted(scores, key=lambda x: x[1], reverse=True),
        "generated_image": img_base64,
        "is_multilabel": model_type == "lewm",
        "task": getattr(model, 'task', 'presence') if model_type == "lewm" else "single_label"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
