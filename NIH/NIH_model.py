import torch
import torch.nn as nn
import torchvision.models as models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes):
        super(ChestXrayModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Example usage
if __name__ == "__main__":
    model = ChestXrayModel(num_classes=14)  # 14 disease labels
    print(model)
