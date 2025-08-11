import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# --- Fair Identity Normalization ---
class FairIdentityNormalization(nn.Module):
    def __init__(self, feature_dim, num_identities, momentum=0.7):
        super(FairIdentityNormalization, self).__init__()
        self.momentum = momentum
        self.feature_dim = feature_dim
        self.num_identities = num_identities

        # Learnable mean and std for each identity group
        self.mu = nn.Parameter(torch.zeros(num_identities, feature_dim))
        self.tau = nn.Parameter(torch.zeros(num_identities, feature_dim))

    def forward(self, z, a):
        # z: [B, D], a: [B]
        sigma = torch.log1p(torch.exp(self.tau))  # softplus to ensure positivity
        mu_a = self.mu[a]           # [B, D]
        sigma_a = sigma[a]          # [B, D]
        z_hat = (z - mu_a) / sigma_a
        z_final = (1 - self.momentum) * z_hat + self.momentum * z
        return z_final

# --- Model for 4-class classification ---
class PassionClassifierFIN(nn.Module):
    def __init__(self, num_classes=4, feature_dim=512, num_identities=4, backbone='resnet18'):
        super(PassionClassifierFIN, self).__init__()
        
        # Backbone: ResNet18 truncated -> costruzione
        # to extract features without the final fully connected layer
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True) # crea modello preaddestrato
            # prende tutti i layer tranne l'ultimo
            modules = list(base_model.children())[:-1]  # remove final FC
            # crea nua pipeline ordinata dei layer, * per spacchettare la lista
            # in modo che nn.Sequential accetti i layer come argomenti
            self.backbone = nn.Sequential(*modules)
            # salva la dimensione output (numero features)
            self.feature_dim = base_model.fc.in_features
        else:
            raise NotImplementedError("Only resnet18 backbone is implemented for now.")

        self.fin = FairIdentityNormalization(self.feature_dim, num_identities)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, identity):
        # estrazione delle features, passa le immagini attraverso il backbone
        features = self.backbone(x)         # [B, 512, 1, 1]
        # rimuove dimensioni inutili (flatten)
        features = features.view(x.size(0), -1)  # [B, 512]
        # normalizza le features in base all'identità
        normalized = self.fin(features, identity)
        # passa le features normalizzate attraverso il classificatore
        # per ottenere le probabilità di classe
        logits = self.classifier(normalized)
        return logits

# --- Model for binary classification (impetigo) ---
class ImpetigoBinaryClassifierFIN(nn.Module):
    def __init__(self, feature_dim=512, num_identities=4, backbone='resnet18'):
        super(ImpetigoBinaryClassifierFIN, self).__init__()
        
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            modules = list(base_model.children())[:-1]
            self.backbone = nn.Sequential(*modules)
            self.feature_dim = base_model.fc.in_features
        else:
            raise NotImplementedError("Only resnet18 backbone is implemented for now.")

        self.fin = FairIdentityNormalization(self.feature_dim, num_identities)
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x, identity):
        features = self.backbone(x)
        features = features.view(x.size(0), -1)
        normalized = self.fin(features, identity)
        logits = self.classifier(normalized)
        return logits
