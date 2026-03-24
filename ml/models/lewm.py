import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer.
    Regularizes projections to match an isotropic Gaussian distribution.
    """
    def __init__(self, n_projections=10, sigma=1.0):
        super().__init__()
        self.n_projections = n_projections
        self.sigma = sigma
        
        # Random orthogonal projections (fixed)
        self.register_buffer('projection_vectors', torch.randn(n_projections, 1))
        self.projection_vectors.data = F.normalize(self.projection_vectors.data, dim=0)
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: (B, T, D) tensor of embeddings
            
        Returns:
            Regularization loss (scalar)
        """
        B, T, D = embeddings.shape
        
        # Flatten to (B*T, D)
        flat_emb = embeddings.view(-1, D)
        
        # Project to 1D
        projections = torch.matmul(flat_emb, self.projection_vectors.T)  # (B*T, n_projections)
        
        # Compute Epps-Pulley statistic
        # Compare against standard Gaussian
        losses = []
        for i in range(self.n_projections):
            proj = projections[:, i]  # (B*T,)
            
            # Gaussian window
            phi = torch.exp(-0.5 * (proj / self.sigma) ** 2)
            
            # Mean of cosine and sine (should be 0 for Gaussian)
            cos_mean = torch.cos(proj).mean()
            sin_mean = torch.sin(proj).mean()
            
            # Loss: deviation from Gaussian
            loss = (cos_mean ** 2 + sin_mean ** 2)
            losses.append(loss)
        
        return sum(losses) / self.n_projections


class Embedder(nn.Module):
    """
    Embeds input sequences into latent space.
    Input: (B, 32*256*3) flattened echogram history
    Output: (B, T, embed_dim) where T=32 timesteps
    """
    def __init__(self, input_dim=24576, embed_dim=256, n_timesteps=32):
        super().__init__()
        self.n_timesteps = n_timesteps
        self.embed_dim = embed_dim
        
        # Each timestep has 256*3 = 768 features (256 depth bins × 3 channels)
        self.timestep_dim = 256 * 3  # 768
        
        # Project each timestep to embed_dim
        self.timestep_proj = nn.Sequential(
            nn.Linear(self.timestep_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, 24576) flattened input (32 timesteps × 768 features)
            
        Returns:
            embeddings: (B, T, embed_dim)
        """
        B = x.shape[0]
        
        # Reshape to (B, T, timestep_dim) = (B, 32, 768)
        x = x.view(B, self.n_timesteps, self.timestep_dim)
        
        # Project to embed_dim
        x = self.timestep_proj(x)  # (B, T, embed_dim)
        
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with causal masking.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # QKV projection
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    def __init__(self, dim, hidden_dim=None, drop=0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        return self.net(x)


class ConditionalBlock(nn.Module):
    """
    Transformer block with AdaLN-zero modulation for conditioning.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=drop, proj_drop=drop)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        
        # AdaLN modulation (produces shift, scale, gate for both attn and mlp)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        # Initialize modulation weights to zero (residual connection)
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        
    def forward(self, x, condition=None):
        """
        Args:
            x: (B, T, D) input tensor
            condition: (B, D) conditioning vector (optional)
            
        Returns:
            x: (B, T, D) output tensor
        """
        # Get modulation parameters
        if condition is None:
            condition = torch.zeros(x.shape[0], x.shape[-1], device=x.device)
        
        mod = self.adaLN_modulation(condition)
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        
        # Attention with modulation
        x_norm = self.norm1(x)
        x_modulated = x_norm * (1 + scale_attn.unsqueeze(1)) + shift_attn.unsqueeze(1)
        x = x + gate_attn.unsqueeze(1) * self.attn(x_modulated)
        
        # MLP with modulation
        x_norm = self.norm2(x)
        x_modulated = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_modulated)
        
        return x


class ARPredictor(nn.Module):
    """
    Autoregressive Transformer predictor with AdaLN conditioning.
    """
    def __init__(self, embed_dim=256, num_layers=6, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ConditionalBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, condition=None):
        """
        Args:
            x: (B, T, D) input embeddings
            condition: (B, D) conditioning vector (optional)
            
        Returns:
            x: (B, T, D) predicted embeddings
        """
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, condition)
        
        x = self.norm(x)
        return x


class LeWorldModel(nn.Module):
    """
    LeWorldModel (LeWM): Stable End-to-End JEPA from pixels.
    
    Key features:
    - Only 2 loss terms: prediction MSE + Gaussian regularizer
    - Stable training without EMA, pretrained encoders, or auxiliary losses
    - Compact latent space with enforced Gaussian distribution
    """
    def __init__(
        self,
        input_dim=768,          # Input dimension (32*256*3 = 24576 flattened, or use encoder)
        embed_dim=256,          # Latent embedding dimension
        patch_size=16,          # Patch size for embedder
        num_layers=6,           # Number of transformer layers in predictor
        num_heads=8,            # Number of attention heads
        mlp_ratio=4.0,          # MLP hidden dim ratio
        drop=0.1,               # Dropout rate
        n_classes=4,            # Number of classification classes
        use_classifier=True,    # Whether to include classification head
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_classifier = use_classifier
        
        # 1. Embedder (encodes input to latent space)
        self.embedder = Embedder(
            input_dim=24576,  # 32 * 256 * 3 flattened
            embed_dim=embed_dim,
            n_timesteps=32
        )
        
        # 2. Predictor (autoregressive transformer)
        self.predictor = ARPredictor(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop
        )
        
        # 3. Prediction projection
        self.pred_proj = nn.Linear(embed_dim, embed_dim)
        
        # 4. Gaussian regularizer
        self.sigreg = SIGReg(n_projections=10, sigma=1.0)
        
        # 5. Classification head (optional, for your fish classification task)
        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 512),
                nn.BatchNorm1d(512),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, n_classes)
            )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
    def encode(self, x):
        """
        Encode input to latent embeddings.
        
        Args:
            x: (B, input_dim) or (B, T, D) input tensor
            
        Returns:
            embeddings: (B, T, embed_dim)
        """
        return self.embedder(x)
    
    def predict(self, embeddings, condition=None):
        """
        Predict future embeddings autoregressively.
        
        Args:
            embeddings: (B, T, embed_dim) input embeddings
            condition: (B, embed_dim) conditioning vector (optional)
            
        Returns:
            predictions: (B, T, embed_dim) predicted embeddings
        """
        return self.predictor(embeddings, condition)
    
    def forward(self, x, condition=None):
        """
        Forward pass: encode and predict.
        
        Args:
            x: (B, input_dim) input tensor (flattened echogram history)
            condition: (B, embed_dim) conditioning vector (optional)
            
        Returns:
            pred_emb: (B, T, embed_dim) predicted embeddings
            goal_emb: (B, T, embed_dim) target embeddings (last timestep)
            species_logits: (B, n_classes) classification logits (if use_classifier)
        """
        # Encode input
        embeddings = self.embedder(x)  # (B, T, embed_dim)
        
        # Predict future embeddings
        pred_emb = self.predict(embeddings, condition)  # (B, T, embed_dim)
        pred_emb = self.pred_proj(pred_emb)
        
        # Goal embeddings (shifted by 1 timestep for prediction)
        goal_emb = embeddings[:, 1:, :]  # (B, T-1, embed_dim)
        pred_emb = pred_emb[:, :-1, :]   # (B, T-1, embed_dim)
        
        # Classification
        if self.use_classifier:
            # Use mean embedding for classification
            mean_emb = embeddings.mean(dim=1)  # (B, embed_dim)
            species_logits = self.classifier(mean_emb)
        else:
            species_logits = None
        
        return pred_emb, goal_emb, species_logits
    
    def compute_loss(self, pred_emb, goal_emb, species_logits, labels, sigreg_weight=0.1):
        """
        Compute LeWM loss: prediction MSE + Gaussian regularizer + classification.
        
        Args:
            pred_emb: (B, T, embed_dim) predicted embeddings
            goal_emb: (B, T, embed_dim) target embeddings
            species_logits: (B, n_classes) classification logits
            labels: (B,) ground truth labels
            sigreg_weight: weight for SIGReg regularizer
            
        Returns:
            total_loss, pred_loss, sigreg_loss, cls_loss
        """
        # 1. Prediction loss (MSE on last timestep)
        pred_loss = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="mean"
        )
        
        # 2. Gaussian regularizer
        sigreg_loss = self.sigreg(pred_emb)
        
        # 3. Classification loss (cross-entropy)
        if species_logits is not None and labels is not None:
            cls_loss = F.cross_entropy(species_logits, labels)
        else:
            cls_loss = torch.tensor(0.0, device=pred_emb.device)
        
        # Total loss
        total_loss = pred_loss + sigreg_weight * sigreg_loss + cls_loss
        
        return total_loss, pred_loss, sigreg_loss, cls_loss
    
    def rollout(self, x, n_steps=10, condition=None):
        """
        Autoregressive rollout for planning/inference.
        
        Args:
            x: (B, input_dim) initial input
            n_steps: number of steps to rollout
            condition: (B, embed_dim) conditioning vector
            
        Returns:
            predictions: list of (B, embed_dim) predictions
        """
        predictions = []
        
        # Initial encoding
        embeddings = self.embedder(x)
        
        for _ in range(n_steps):
            # Predict next embedding
            pred = self.predict(embeddings, condition)
            pred = self.pred_proj(pred[:, -1:, :])  # (B, 1, embed_dim)
            
            predictions.append(pred[:, 0, :])
            
            # Append prediction to embeddings (autoregressive)
            embeddings = torch.cat([embeddings, pred], dim=1)
        
        return predictions
