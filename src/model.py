# src/model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

IMAGES_PER_PHON = 5  # keep in sync with utils.py


class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Parameter(torch.randn(num_heads, 1, dim))
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):  # x: [B, P, D]
        B, P, D = x.size()
        x = x.unsqueeze(1).repeat(1, self.num_heads, 1, 1)      # [B,H,P,D]
        q = self.q.unsqueeze(0).repeat(B, 1, 1, 1)              # [B,H,1,D]
        attn = (q @ x.transpose(2, 3)) / (D ** 0.5)             # [B,H,1,P]
        w = attn.softmax(dim=-1)
        w = self.dropout(w)
        out = w @ x                                             # [B,H,1,D]
        return out.mean(1).squeeze(1)                           # [B,D]


class CNNMultistreamModel(nn.Module):
    def __init__(self, arcface_dim=512, geo_dim=1,
                 embed_dim=128, num_heads=4,
                 frames_per_phon=IMAGES_PER_PHON):
        super().__init__()
        self.E = embed_dim
        self.F = frames_per_phon

        # light temporal head across F frames
        self.cnn3d = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(),
            nn.Conv3d(16, 3, kernel_size=1),
            nn.BatchNorm3d(3), nn.ReLU()
        )

        eff = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.backbone = eff.features
        for n, p in self.backbone.named_parameters():
            # fine-tune last blocks only (matches your training script)
            p.requires_grad = ("6" in n or "7" in n)

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, embed_dim),
            nn.ReLU()
        )

        self.geo_enc = nn.Sequential(nn.Linear(geo_dim, embed_dim), nn.ReLU())
        self.arc_enc = nn.Sequential(nn.Linear(arcface_dim, embed_dim), nn.ReLU())

        self.dropout    = nn.Dropout(0.2)
        self.attn_pool  = MultiHeadAttentionPooling(embed_dim * 3, num_heads)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 128), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, geoms, imgs, arcs, mask=None):
        B, P, F, C, H, W = imgs.shape

        # visual: 3D head over F -> average across time -> EfficientNet features
        x = imgs.view(B * P, C, F, H, W)   # [B*P,3,F,H,W]
        x = self.cnn3d(x).mean(dim=2)      # [B*P,3,H,W]

        v = self.backbone(x)               # -> [B*P,1280,h,w]
        v = self.projector(v)              # -> [B*P,E]
        v = v.view(B, P, self.E)           # [B,P,E]

        # geometry (mean across frames)
        g = self.geo_enc(geoms.mean(dim=2))    # [B,P,E]

        # arcface (mean across frames after linear)
        a = arcs.view(B * P, F, -1)
        a = self.arc_enc(a).mean(dim=1).view(B, P, self.E)

        fused = torch.cat([v, g, a], dim=-1)   # [B,P,3E]
        fused = self.dropout(fused)
        pooled = self.attn_pool(fused)         # [B,3E]
        logits = self.classifier(pooled)       # [B,2]
        return (logits,a)
