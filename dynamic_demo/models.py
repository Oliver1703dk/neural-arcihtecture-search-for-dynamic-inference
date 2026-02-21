"""
Dynamic Inference Demo (CIFAR-10): Big/Little + Learned Routing

This file defines the minimal model components used throughout the demo:

1) SmallCNN(widths=(w1,w2,w3)):
   - A small 3-stage CNN for CIFAR-10 classification.
   - The key design choice is that "widths" are PER-STAGE channel counts,
     not a single global multiplier. This mimics "per-layer / per-block width
     configurations" that NAS would eventually discover.
   - The network also exposes an early feature map ("feat") from the stem stage.
     That early feature is what the routing network uses to estimate whether an
     input is "easy" (can be handled by the Little path) or "hard" (should be
     escalated to the Big path).

2) Router(in_ch):
   - A tiny learned gating network that outputs g(x) ∈ [0,1], interpreted as the
     probability of escalating an input to the Big network.
   - The Router is deliberately learned from features rather than being only a
     confidence threshold. This demonstrates "routing mechanisms beyond confidence
     gating", which is a major gap highlighted in dynamic inference NAS literature.

How this connects to the research framing:
- "Big/Little": We instantiate two SmallCNN models with different per-stage widths:
    Little: (24, 48, 96)  (fast, cheaper)
    Big:    (48, 96, 192) (slower, more accurate)
- "Learned routing": Router learns to map early features to an escalation decision.
- "Hardware-aware objective": Later scripts will use measured latency of the exported
  ONNX models, so routing is optimized for expected real latency (not MACs).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCNN(nn.Module):
    """
    Simple 3-stage CNN for CIFAR-10.
    widths=(w1,w2,w3) are per-stage channel counts (NOT a global multiplier).
    """
    def __init__(self, widths=(24, 48, 96), num_classes=10):
        super().__init__()
        w1, w2, w3 = widths

        self.stem = nn.Sequential(
            nn.Conv2d(3, w1, 3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
            nn.Conv2d(w1, w1, 3, padding=1, bias=False),
            nn.BatchNorm2d(w1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(w1, w2, 3, stride=2, padding=1, bias=False),  # 32->16
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
            nn.Conv2d(w2, w2, 3, padding=1, bias=False),
            nn.BatchNorm2d(w2),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(w2, w3, 3, stride=2, padding=1, bias=False),  # 16->8
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
            nn.Conv2d(w3, w3, 3, padding=1, bias=False),
            nn.BatchNorm2d(w3),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w3, num_classes)

    def forward(self, x, return_feat=False):
        x = self.stem(x)
        feat = x  # early feature (router input)
        x = self.down1(x)
        x = self.down2(x)
        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        if return_feat:
            return logits, feat
        return logits

class Router(nn.Module):
    """
    Hybrid learned routing network.

    Outputs g(x) in [0,1] = probability of escalating to the Big model.

    Inputs:
      - feat: early feature map from the Little network
      - max_prob: Little's max softmax probability (confidence)
      - ent: entropy of Little's softmax distribution (uncertainty)

    Why hybrid:
      - Confidence-only thresholds are a common baseline.
      - Pure learned routing can miss simple calibration signals.
      - Combining both makes the routing mechanism richer and matches the
        research goal of jointly using learned routing + confidence gating.
    """
    def __init__(self, in_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_ch + 2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, feat, max_prob, ent):
        x = self.pool(feat).flatten(1)              # (B, in_ch)
        z = torch.stack([max_prob, ent], dim=1)     # (B, 2)
        x = torch.cat([x, z], dim=1)                # (B, in_ch+2)
        return torch.sigmoid(self.mlp(x)).squeeze(1)