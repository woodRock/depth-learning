import torch
import torch.nn as nn
from .ast import PatchEmbedding

class MaskedAttentionFusion(nn.Module):
    """
    Cross-modal fusion using visual features to attend to masked acoustic patches.
    """
    def __init__(self, d_model=256, nhead=8, num_classes=4):
        super().__init__()
        
        self.acoustic_patch_embed = PatchEmbedding(img_size=(32, 256), patch_size=(8, 16), in_channels=3, d_model=d_model)
        
        # Visual feature projection (assuming visual features come from a CNN like ResNet)
        # ResNet50 avgpool is 2048
        self.visual_projection = nn.Linear(2048, d_model)
        
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, visual_features, acoustic_data, mask_ratio=0.5):
        """
        visual_features: (B, 2048)
        acoustic_data: (B, 3, 32, 256) or flattened
        """
        batch_size = acoustic_data.size(0)
        
        # 1. Acoustic Patches
        if acoustic_data.dim() == 2:
            # Flattened input: (B, 32*256*3) where order is ping->depth->channel
            # Reshape to (B, 32, 256, 3) then permute to (B, 3, 32, 256)
            acoustic_data = acoustic_data.view(batch_size, 32, 256, 3).permute(0, 3, 1, 2)
        elif acoustic_data.dim() == 4 and acoustic_data.shape[1] == 32:
            # (B, 32, 256, 3) -> (B, 3, 32, 256)
            acoustic_data = acoustic_data.permute(0, 3, 1, 2)
        
        acoustic_patches = self.acoustic_patch_embed(acoustic_data) # (B, num_patches, d_model)
        
        # 2. Random Masking (Training only)
        if self.training:
            num_patches = acoustic_patches.size(1)
            len_keep = int(num_patches * (1 - mask_ratio))
            noise = torch.rand(batch_size, num_patches, device=acoustic_patches.device)
            ids_keep = torch.argsort(noise, dim=1)[:, :len_keep]
            acoustic_patches = torch.gather(acoustic_patches, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, acoustic_patches.size(2)))
        
        # 3. Visual Queries
        visual_query = self.visual_projection(visual_features).unsqueeze(1) # (B, 1, d_model)
        
        # 4. Cross-Attention: Visual Query attends to Acoustic Keys/Values
        attn_output, _ = self.cross_attention(query=visual_query, 
                                              key=acoustic_patches, 
                                              value=acoustic_patches)
        
        # 5. Concatenate Visual Query and Attended Acoustic Context
        combined = torch.cat((visual_query.squeeze(1), attn_output.squeeze(1)), dim=1)
        
        logits = self.classifier(combined)
        return logits
