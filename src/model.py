

import torch
import torch.nn as nn
from src.dataset import NUM_CLASSES, IMAGE_SIZE

#arhitectura cnn pentru recunoasterea semnelor de circulatie

#block de convolutie + batchnorm + relu + (optional) maxpool
class ConvBlock(nn.Module):

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


#modelul complet cu backbone, neck si head
class TrafficSignCNN(nn.Module):
    """
    arhitectura (input 48*48 RGB):
      Backbone
        Block 1: Conv(3-32)  + BN + ReLU + MaxPool  
        Block 2: Conv(32-64) + BN + ReLU + MaxPool  
        Block 3: Conv(64-128)+ BN + ReLU            
        Block 4: Conv(128-128)+BN + ReLU + MaxPool 

      Neck (multi-scale feature fusion approximation)
        Global Average Pool                         
        Flatten                                      

      Head (classification)
        FC 128 -> 64 -> num_classes
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()

        # backbone - feature extraction
        self.backbone = nn.Sequential(
            ConvBlock(3,   32,  pool=True),   # 48 - 24
            ConvBlock(32,  64,  pool=True),   # 24-12
            ConvBlock(64,  128, pool=False),  # 12-12  (extra conv, no pool)
            ConvBlock(128, 128, pool=True),   # 12-6
        )

        #Neck - spatial reduction to a fixed-size vector
        self.neck = nn.AdaptiveAvgPool2d(1)   # 6×6 - 1×1 regardless of input size

        # Head - classification
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
        #returneaza predictiile si confidence score
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = probs.max(dim=1)
        return pred, confidence



#functii de construire a modelului
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
