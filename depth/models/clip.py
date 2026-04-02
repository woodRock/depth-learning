import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
import math

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class DualEncoderCLIP(nn.Module):
    def __init__(self, ac_encoder, embed_dim=256):
        super().__init__()
        
        # 1. Visual Stream (ResNet18)
        self.vis_backbone = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.vis_backbone.fc.in_features
        self.vis_backbone.fc = nn.Identity()
        self.vis_proj = ProjectionHead(in_features, embed_dim)
        
        # 2. Acoustic Stream (Injected)
        self.ac_encoder = ac_encoder
        
        # 3. Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, vis, ac):
        vis_feat = self.vis_backbone(vis)
        vis_emb = F.normalize(self.vis_proj(vis_feat), p=2, dim=-1)
        
        ac_emb = F.normalize(self.ac_encoder(ac), p=2, dim=-1)
        
        logit_scale = self.logit_scale.exp()
        logits_per_vis = logit_scale * vis_emb @ ac_emb.t()
        logits_per_ac = logits_per_vis.t()
        
        return logits_per_vis, logits_per_ac

    def forward_vis(self, vis):
        feat = self.vis_backbone(vis)
        return F.normalize(self.vis_proj(feat), p=2, dim=-1)

    def forward_ac(self, ac):
        return F.normalize(self.ac_encoder(ac), p=2, dim=-1)
