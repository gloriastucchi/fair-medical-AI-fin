import torch
import torch.nn as nn
import torchvision.models as models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14, extract_features_only=True):
        super(ChestXrayModel, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.feature_extractor = densenet.features  # Output: (B, 1024, 7, 7)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)

        self.extract_features_only = extract_features_only
        self.num_features = 1024  # DenseNet121 final feature dimension
        self.classifier = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)                 # -> (B, 1024, 7, 7)
        x = self.relu(x)
        x = self.avg_pool(x)                          # -> (B, 1024, 1, 1)
        x = x.view(x.size(0), -1)                     # -> (B, 1024)

        if self.extract_features_only:
            return x                                  # output per FIN
        else:
            return self.classifier(x)                 # output classificazione



if __name__ == "__main__":
    model = ChestXrayModel(num_classes=14)  # 15 disease labels - "no finding" esclusa
    print(model)
