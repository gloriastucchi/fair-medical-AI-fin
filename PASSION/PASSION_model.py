# === model.py ===
import torch
import torch.nn as nn
import torchvision.models as models

class PassionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(PassionClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class ImpetigoBinaryClassifier(nn.Module):
    def __init__(self):
        super(ImpetigoBinaryClassifier, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 1)
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)  # for BCEWithLogitsLoss
