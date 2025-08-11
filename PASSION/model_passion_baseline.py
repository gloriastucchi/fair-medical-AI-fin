import torch
import torch.nn as nn
import torchvision.models as models

class PassionClassifierNoFIN(nn.Module):
    def __init__(self, num_classes=4, feature_dim=512, backbone='resnet18'):
        super(PassionClassifierNoFIN, self).__init__()
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            modules = list(base_model.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.feature_dim = base_model.fc.in_features
        else:
            raise NotImplementedError("Only resnet18 backbone is implemented for now.")

        # Solo Linear, come la versione con FIN!
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)         # [B, 512, 1, 1]
        features = features.view(x.size(0), -1)  # [B, 512]
        logits = self.classifier(features)  # [B, num_classes]
        return logits

class ImpetigoBinaryClassifierNoFIN(nn.Module):
    def __init__(self, feature_dim=512, backbone='resnet18'):
        super(ImpetigoBinaryClassifierNoFIN, self).__init__()
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            modules = list(base_model.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.feature_dim = base_model.fc.in_features
        else:
            raise NotImplementedError("Only resnet18 backbone is implemented for now.")

        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(x.size(0), -1)
        logits = self.classifier(features)  # [B, 1]
        return logits.squeeze(1)            # Per BCEWithLogitsLoss
