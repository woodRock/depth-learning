import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AcousticToImageTransformer(nn.Module):
    def __init__(self, 
                 num_pings=32, 
                 ping_dim=768, # 256 * 3
                 d_model=256, 
                 nhead=8, 
                 num_encoder_layers=6, 
                 num_decoder_layers=6,
                 dim_feedforward=1024,
                 num_classes=4,
                 patch_size=16,
                 image_size=224):
        super().__init__()
        
        self.d_model = d_model
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # --- Encoder ---
        # Initial projection of acoustic pings
        self.ping_projection = nn.Linear(ping_dim, d_model)
        self.encoder_pos_encoding = PositionalEncoding(d_model, max_len=num_pings + 1)
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # --- Decoder ---
        # Learned queries for image patches
        self.patch_queries = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        # Final projection from decoder output to patch pixels
        self.patch_projection = nn.Linear(d_model, 3 * patch_size * patch_size)
        
        # --- Classification Head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, acoustic_data):
        """
        acoustic_data: (B, num_pings * ping_dim) - flattened history
        """
        batch_size = acoustic_data.size(0)
        
        # Reshape acoustic data to (B, num_pings, ping_dim)
        x = acoustic_data.view(batch_size, 32, -1)
        
        # Project to d_model
        x = self.ping_projection(x) # (B, 32, d_model)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, 33, d_model)
        
        # Add positional encoding
        x = self.encoder_pos_encoding(x)
        
        # Encode
        memory = self.encoder(x) # (B, 33, d_model)
        
        # Classification output
        cls_output = memory[:, 0, :]
        class_logits = self.classifier(cls_output)
        
        # Decoder
        # Use patch queries
        queries = self.patch_queries.expand(batch_size, -1, -1)
        queries = self.decoder_pos_encoding(queries)
        
        # Decode attending to acoustic memory (excluding CLS token from memory)
        tgt = self.decoder(queries, memory[:, 1:, :]) # (B, num_patches, d_model)
        
        # Project to patches
        patches = self.patch_projection(tgt) # (B, num_patches, 3 * patch_size * patch_size)
        
        # Reshape patches to image
        # (B, 14*14, 3*16*16) -> (B, 14, 14, 3, 16, 16)
        p = self.patch_size
        h = w = self.image_size // p
        patches = patches.view(batch_size, h, w, 3, p, p)
        # Permute to (B, 3, h, p, w, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        # Reshape to (B, 3, image_size, image_size)
        image = patches.view(batch_size, 3, self.image_size, self.image_size)
        
        return torch.sigmoid(image), class_logits
