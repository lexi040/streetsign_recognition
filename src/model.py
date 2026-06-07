"""
CNN architecture for traffic sign recognition.

Follows the structure described in the CV4V lecture (Curs 7 – Object Detection):
  Input → Backbone (conv layers + pooling) → Neck (feature fusion) → Head (classification)

The backbone progressively extracts spatial features (edges → shapes → patterns),
each block halving spatial dimensions while doubling channels.
"""

import torch
import torch.nn as nn
from src.dataset import NUM_CLASSES, IMAGE_SIZE


# ── Building blocks ───────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv → BatchNorm → ReLU → (optional MaxPool)."""

    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ── Main model ────────────────────────────────────────────────────────────────

class TrafficSignCNN(nn.Module):
    """
    Simple but effective CNN for 5-class traffic sign recognition.

    Architecture (input 48×48 RGB):
      Backbone
        Block 1: Conv(3→32)  + BN + ReLU + MaxPool  → 24×24×32
        Block 2: Conv(32→64) + BN + ReLU + MaxPool  → 12×12×64
        Block 3: Conv(64→128)+ BN + ReLU            → 12×12×128  (no pool)
        Block 4: Conv(128→128)+BN + ReLU + MaxPool  →  6×6×128

      Neck (multi-scale feature fusion approximation)
        Global Average Pool                          →  1×1×128
        Flatten                                      → 128

      Head (classification)
        FC 128 → 64 → num_classes
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        # Backbone — feature extraction
        self.backbone = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # 48→24
            ConvBlock(32,  64,  pool=True),   # 24→12
            ConvBlock(64,  128, pool=False),  # 12→12  (extra conv, no pool)
            ConvBlock(128, 128, pool=True),   # 12→6
        )

        # Neck — spatial reduction to a fixed-size vector
        self.neck = nn.AdaptiveAvgPool2d(1)   # 6×6 → 1×1 regardless of input size

        # Head — classification
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (class_index, confidence) for a batch."""
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)
        return pred, confidence


# ── Model factory & info ──────────────────────────────────────────────────────

def build_model(device: str | torch.device = "cpu") -> TrafficSignCNN:
    model = TrafficSignCNN()
    return model.to(device)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> None:
    total = count_parameters(model)
    print(f"Model: {model.__class__.__name__}")
    print(f"Trainable parameters: {total:,}")
    print(f"Input  shape: (N, 3, {IMAGE_SIZE}, {IMAGE_SIZE})")
    print(f"Output shape: (N, {NUM_CLASSES})  → classes: {NUM_CLASSES}")
