import torch
import torch.nn as nn
import torchvision.models as models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes):
        super(ChestXrayModel, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # tutto fino al penultimo layer (global avg pooling)
        self.feature_dim = resnet.fc.in_features  # 512
        self.fc = nn.Linear(self.feature_dim, num_classes)  # output layer (solo per baseline)

    def forward(self, x):
        features = self.backbone(x).squeeze(-1).squeeze(-1)  # da [B, 512, 1, 1] → [B, 512]
        return self.fc(features)

    def extract_features(self, x):
        features = self.backbone(x).squeeze(-1).squeeze(-1)
        return features


# Example usage
if __name__ == "__main__":
    model = ChestXrayModel(num_classes=14)  # 15 disease labels - "no finding" esclusa
    print(model)
