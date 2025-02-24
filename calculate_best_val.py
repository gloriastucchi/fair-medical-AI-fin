import os
import pandas as pd

# Path to the 'results' directory containing seed folders
results_dir = './results'

# Initialize a list to store the highest val_auc per seed
best_val_aucs = []

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
        best_val_aucs.append(max_val_auc)
        print(f"Seed {seed_folder}: Best val_auc = {max_val_auc:.4f}")

# Calculate mean and standard deviation of best val_auc values
mean_val_auc = sum(best_val_aucs) / len(best_val_aucs)
std_val_auc = pd.Series(best_val_aucs).std()

print(f"\nMean of best val_auc values: {mean_val_auc:.4f}")
print(f"Standard deviation of best val_auc values: {std_val_auc:.4f}")
