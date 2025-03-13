import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from NIH_dataset import ChestXrayDataset, transform
from NIH_model import ChestXrayModel

# Load dataset
CSV_FILE = "Data_Entry_2017_v2020.csv"
IMAGE_FOLDER = "images/"
TEST_LIST = "test_list.txt"

test_dataset = ChestXrayDataset(CSV_FILE, IMAGE_FOLDER, TEST_LIST, transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChestXrayModel(num_classes=14)
model.load_state_dict(torch.load("chestxray_model.pth"))
model.to(device)
model.eval()

# Evaluate model
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.cpu().numpy()
        outputs = torch.sigmoid(model(images)).cpu().numpy()
        y_true.append(labels)
        y_pred.append(outputs)

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Compute AUC score
auc_score = roc_auc_score(y_true, y_pred, average="macro")
print(f"Test AUC Score: {auc_score:.4f}")
