import os
import pandas as pd

# Path to the 'results' directory containing seed folders
results_dir = './results/no_fin/test'

# Initialize a list to store the highest val_auc per seed
best_val_aucs = []
best_val_aucs_0 = []
best_val_aucs_1 = []
best_val_aucs_2 = []

# Iterate through each seed folder in the results directory
for seed_folder in os.listdir(results_dir):
    seed_path = os.path.join(results_dir, seed_folder)
    progress_file = os.path.join(seed_path, 'progress_train.csv')

    # Check if progress_train.csv exists in the folder
    if os.path.isfile(progress_file):
        # Read the CSV file
        df = pd.read_csv(progress_file)

        # Extract the highest val_auc value
        max_val_auc = df['val_auc'].max()
        max_val_auc_0 = df['val_auc_class0'].max() 
        max_val_auc_1 = df['val_auc_class1'].max()
        max_val_auc_2 = df['val_auc_class2'].max()  
        best_val_aucs.append(max_val_auc)
        best_val_aucs_0.append(max_val_auc_0)
        best_val_aucs_1.append(max_val_auc_1)
        best_val_aucs_2.append(max_val_auc_2)
        print(f"Seed {seed_folder}: Best val_auc = {max_val_auc:.4f}")
        print(f"Seed {seed_folder}: Best val_auc for class 0 = {max_val_auc_0:.4f}")
        print(f"Seed {seed_folder}: Best val_auc for class 1 = {max_val_auc_1:.4f}")
        print(f"Seed {seed_folder}: Best val_auc for class 2 = {max_val_auc_2:.4f}")

# Calculate mean and standard deviation of best val_auc values
mean_val_auc = sum(best_val_aucs) / len(best_val_aucs)
mean_val_auc_0 = sum(best_val_aucs_0) / len(best_val_aucs_0)
mean_val_auc_1 = sum(best_val_aucs_1) / len(best_val_aucs_1)
mean_val_auc_2 = sum(best_val_aucs_2) / len(best_val_aucs_2)

std_val_auc = pd.Series(best_val_aucs).std(ddof=1)
std_val_auc_0 = pd.Series(best_val_aucs_0).std(ddof=1)
std_val_auc_1 = pd.Series(best_val_aucs_1).std(ddof=1)
std_val_auc_2 = pd.Series(best_val_aucs_2).std(ddof=1)

print(f"\nMean of best val_auc values: {mean_val_auc:.4f}")
print(f"Standard deviation of best val_auc values: {std_val_auc:.4f}")
print(f"\nMean of best val_auc values for class 0: {mean_val_auc_0:.4f}")
print(f"Standard deviation of best val_auc values for class 0: {std_val_auc_0:.4f}")
print(f"\nMean of best val_auc values for class 1: {mean_val_auc_1:.4f}")
print(f"Standard deviation of best val_auc values for class 1: {std_val_auc_1:.4f}")
print(f"\nMean of best val_auc values for class 2: {mean_val_auc_2:.4f}")
print(f"Standard deviation of best val_auc values for class 2: {std_val_auc_2:.4f}")
