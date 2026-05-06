"""
ViT-based cross-modal JEPA.

Two-stage setup:
  1. Pretrain a ViT encoder with MAE on DepthSim visual frames (no labels).
  2. Freeze the pretrained encoder as the JEPA teacher, replacing ResNet-18.

Architecture: ViT-Small (embed_dim=384, depth=6, heads=6, patch_size=16).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # [B, N, C]


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ViT encoder  (shared between MAE pretraining and JEPA teacher)
# ---------------------------------------------------------------------------

class ViTEncoder(nn.Module):
    """ViT-Small encoder.

    Used two ways:
      - ``forward(x)``  — full-image encoding for JEPA teacher, returns mean of
                          patch tokens → [B, embed_dim].
      - ``encode_masked(x, mask_ratio)``  — MAE mode: encodes only visible
                          patches; returns (encoded, mask, ids_restore).
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=6, num_heads=6):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = embed_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """Full-image forward for JEPA teacher. Returns mean-pooled patch tokens [B, C]."""
        B = x.shape[0]
        tokens = self.patch_embed(x)                          # [B, N, C]
        tokens = tokens + self.pos_embed[:, 1:, :]

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        tokens = torch.cat([cls, tokens], dim=1)              # [B, 1+N, C]

        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)

        return tokens[:, 1:].mean(dim=1)                     # [B, C]

    def encode_masked(self, x, mask_ratio=0.75):
        """MAE encoder: processes only the visible subset of patches.

        Returns:
            encoded   [B, 1+len_keep, embed_dim]  — CLS + visible encodings
            mask      [B, N]                       — 1 = masked, 0 = visible
            ids_restore [B, N]                     — permutation to unshuffle
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)      # [B, N, C]
        N = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, 1:, :]

        # Random masking
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        visible = torch.gather(
            tokens, 1, ids_keep.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
        )

        # Mask: 1 = masked, 0 = visible (in original patch order)
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        cls = self.cls_token.expand(B, -1, -1) + self.pos_embed[:, :1, :]
        visible = torch.cat([cls, visible], dim=1)

        for blk in self.blocks:
            visible = blk(visible)
        visible = self.norm(visible)

        return visible, mask, ids_restore


# ---------------------------------------------------------------------------
# MAE pretraining model
# ---------------------------------------------------------------------------

class VisualMAE(nn.Module):
    """Masked Autoencoder for self-supervised ViT pretraining on visual frames."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, encoder_depth=6, encoder_heads=6,
                 decoder_dim=192, decoder_depth=4, decoder_heads=3,
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.mask_ratio = mask_ratio

        self.encoder = ViTEncoder(img_size, patch_size, in_chans,
                                  embed_dim, encoder_depth, encoder_heads)
        num_patches = self.encoder.num_patches

        # Lightweight decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim))
        self.decoder_blocks = nn.ModuleList(
            [ViTBlock(decoder_dim, decoder_heads) for _ in range(decoder_depth)]
        )
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * in_chans)

        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def patchify(self, imgs):
        """[B, C, H, W] → [B, N, patch_size^2 * C]"""
        p = self.patch_size
        c = self.in_chans
        h = w = imgs.shape[-1] // p
        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = torch.einsum('bchpwq->bhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * c)
        return x

    def forward(self, imgs):
        """Encode visible patches, decode all patches. Returns (pred, mask)."""
        B = imgs.shape[0]

        # Encode visible patches only
        encoded, mask, ids_restore = self.encoder.encode_masked(imgs, self.mask_ratio)
        # encoded: [B, 1+len_keep, embed_dim]

        # Project encoder output to decoder dimension
        encoded_dec = self.decoder_embed(encoded)           # [B, 1+len_keep, decoder_dim]

        # Restore full sequence: visible encodings + mask tokens, then unshuffle
        N = ids_restore.shape[1]
        len_keep = encoded_dec.shape[1] - 1                 # excluding CLS
        n_masked = N - len_keep

        mask_tokens = self.mask_token.expand(B, n_masked, -1)
        x_ = torch.cat([encoded_dec[:, 1:], mask_tokens], dim=1)   # [B, N, decoder_dim]
        x_ = torch.gather(
            x_, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1])
        )                                                            # unshuffle

        # Add decoder positional embeddings
        x_ = x_ + self.decoder_pos_embed[:, 1:, :]
        cls_dec = encoded_dec[:, :1] + self.decoder_pos_embed[:, :1, :]
        x_ = torch.cat([cls_dec, x_], dim=1)               # [B, 1+N, decoder_dim]

        for blk in self.decoder_blocks:
            x_ = blk(x_)
        x_ = self.decoder_norm(x_)

        pred = self.decoder_pred(x_[:, 1:])                 # [B, N, patch_size^2*C]
        return pred, mask

    def compute_loss(self, imgs, pred, mask):
        """Normalised MSE loss computed only on the masked patches."""
        target = self.patchify(imgs)                        # [B, N, patch_size^2*C]

        # Per-patch normalisation (as in the MAE paper)
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1e-6).sqrt()

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)                            # [B, N]
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)   # average over masked patches
        return loss


# ---------------------------------------------------------------------------
# ViT-JEPA: JEPA with pretrained ViT teacher
# ---------------------------------------------------------------------------

class ViTCrossModalJEPA(nn.Module):
    """Cross-modal JEPA where the visual teacher is a pretrained ViT-Small.

    Replaces the ResNet-18 teacher with a ViT encoder whose weights were
    learnt from DepthSim visual frames via MAE (no labels).

    Target representation: mean of patch tokens → [B, VIT_DIM].
    Predictor maps acoustic embedding → [B, VIT_DIM].
    """

    VIT_DIM = 384  # Must match ViTEncoder embed_dim

    def __init__(self, ac_encoder, vit_encoder: ViTEncoder,
                 embed_dim=256, use_focal_loss=True, task="presence"):
        super().__init__()
        from .jepa import FocalLoss

        # 1. Target encoder — pretrained ViT, frozen
        self.target_encoder = vit_encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # 2. Context encoder (acoustic)
        self.context_encoder = ac_encoder

        # 3. Predictor: acoustic embed_dim → ViT dim
        vit_dim = self.VIT_DIM
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, vit_dim),
        )

        # 4. Task-specific classifier head (identical to CrossModalJEPA)
        self.task = task
        if task == "presence":
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 768),
                nn.BatchNorm1d(768),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(768, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 4),
            )
            self.criterion_cls = nn.BCEWithLogitsLoss()
        else:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 768),
                nn.BatchNorm1d(768),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(768, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 4),
            )
            self.criterion_cls = (
                FocalLoss(alpha=1.0, gamma=2.0) if use_focal_loss
                else nn.CrossEntropyLoss()
            )

    def forward(self, vis, ac, labels=None):
        with torch.no_grad():
            target_feat = self.target_encoder(vis)              # [B, VIT_DIM]
            target_latent = F.normalize(target_feat, p=2, dim=-1)

        context_latent = self.context_encoder(ac)               # [B, embed_dim]
        predicted_target = self.predictor(context_latent)       # [B, VIT_DIM]
        predicted_target = F.normalize(predicted_target, p=2, dim=-1)

        species_logits = self.classifier(context_latent)

        return predicted_target, target_latent, species_logits

    def compute_loss(self, predicted_target, target_latent, species_logits, labels):
        loss_jepa = (
            1.0 - F.cosine_similarity(predicted_target, target_latent, dim=-1)
        ).mean()
        loss_cls = self.criterion_cls(species_logits, labels)
        return loss_jepa + loss_cls, loss_jepa, loss_cls

    def forward_ac_to_vis_latent(self, ac):
        context_latent = self.context_encoder(ac)
        predicted_target = self.predictor(context_latent)
        species_logits = self.classifier(context_latent)
        return F.normalize(predicted_target, p=2, dim=-1), species_logits
