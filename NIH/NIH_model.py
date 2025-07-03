import torch
import torch.nn as nn
import torchvision.models as models

import torch
import torch.nn as nn
import torchvision.models as models

class ChestXrayModel(nn.Module):
    def __init__(self, num_classes=14):
        super(ChestXrayModel, self).__init__()
        densenet = models.densenet121(pretrained=True)
        num_ftrs = densenet.classifier.in_features
        densenet.classifier = nn.Linear(num_ftrs, num_classes)
        self.model = densenet

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        features = self.model.features(x)
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        return out



# Example usage
if __name__ == "__main__":
    model = ChestXrayModel(num_classes=14)  # 15 disease labels - "no finding" esclusa
    print(model)
