import torch
import torch.nn as nn

class FairIdentityNormalization(nn.Module):
    def __init__(self, num_features, num_groups):
        super(FairIdentityNormalization, self).__init__()
        self.num_groups = num_groups
        self.mean = nn.Parameter(torch.zeros(num_groups, num_features))
        self.std = nn.Parameter(torch.ones(num_groups, num_features))

    def forward(self, x, group_idx):
        mean = self.mean[group_idx]
        std = self.std[group_idx]
        return (x - mean) / (std + 1e-5)  # Avoid division by zero


# Example usage in a model
class FairChestXrayModel(nn.Module):
    def __init__(self, base_model, feature_dim, num_groups):
        super(FairChestXrayModel, self).__init__()
        self.base_model = base_model
        self.fin = FairIdentityNormalization(feature_dim, num_groups)
        self.fc = nn.Linear(feature_dim, 15)  # 15 diseases

    def forward(self, x, identity_group):
        features = self.base_model(x)
        normalized_features = self.fin(features, identity_group)
        return self.fc(normalized_features)
