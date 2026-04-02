import torch
import torch.nn as nn
from .ast import PatchEmbedding

class AcousticMAE(nn.Module):
    """
    Masked Autoencoder (MAE) for acoustic data.
    """
    def __init__(self, 
                 img_size=(32, 256), 
                 patch_size=(8, 16), 
                 in_channels=3, 
                 embed_dim=256, 
                 encoder_depth=4, 
                 decoder_depth=2, 
                 nhead=8, 
                 mask_ratio=0.75):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # --- Encoder ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        enc_layer = nn.TransformerEncoderLayer(embed_dim, nhead, embed_dim*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, encoder_depth)
        
        # --- Decoder ---
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim // 2))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim // 2))
        
        dec_layer = nn.TransformerEncoderLayer(embed_dim // 2, nhead // 2, embed_dim*2, batch_first=True)
        self.decoder = nn.TransformerEncoder(dec_layer, decoder_depth)
        
        self.reconstruct_head = nn.Linear(embed_dim // 2, in_channels * patch_size[0] * patch_size[1])

    def random_masking(self, x, mask_ratio):
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward(self, x):
        # x: (B, 3, 32, 256) or flattened
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.view(batch_size, 32, 256, 3).permute(0, 3, 1, 2)
        elif x.shape[1] != 3:
            x = x.permute(0, 3, 1, 2)
        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # Append CLS token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Encoder
        x = self.encoder(x)
        
        # --- Decoder ---
        x = self.decoder_embed(x)
        
        # Fill in masked patches
        mask_tokens = self.mask_token.expand(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], -1)
        x_all = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_all = torch.gather(x_all, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        
        # Append CLS
        x = torch.cat([x[:, :1, :], x_all], dim=1)
        x = x + self.decoder_pos_embed
        
        x = self.decoder(x)
        x = self.reconstruct_head(x[:, 1:, :])
        
        return x, mask
