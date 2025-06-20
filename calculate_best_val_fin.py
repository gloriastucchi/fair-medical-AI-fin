import os
import pandas as pd

# Path to the 'results' directory containing seed folders
results_dir = '/work3/s232437/fair-medical-AI-fin/results/batch6_imgsize200_fin'

# Initialize lists to store the highest values per seed
best_val_aucs = []
best_val_aucs_0 = []
best_val_aucs_1 = []
best_val_aucs_2 = []
best_es_aucs = []
best_overall_accs = []
best_es_accs = []

# Iterate through each seed folder in the results directory
for seed_folder in os.listdir(results_dir):
    seed_path = os.path.join(results_dir, seed_folder)
    progress_file = os.path.join(seed_path, 'progress_train.csv')

    # Check if progress_train.csv exists in the folder
    if os.path.isfile(progress_file):
        # Read the CSV file
        df = pd.read_csv(progress_file)

        # Extract the highest val_auc values
        max_val_auc = df['val_auc'].max()
        max_val_auc_0 = df['val_auc_class0'].max() 
        max_val_auc_1 = df['val_auc_class1'].max()
        max_val_auc_2 = df['val_auc_class2'].max()

        # Extract additional metrics
        max_es_auc = df['val_es_auc'].max()
        max_overall_acc = df['val_acc'].max()
        max_es_acc = df['val_es_acc'].max()

        # Store values
        best_val_aucs.append(max_val_auc)
        best_val_aucs_0.append(max_val_auc_0)
        best_val_aucs_1.append(max_val_auc_1)
        best_val_aucs_2.append(max_val_auc_2)
        best_es_aucs.append(max_es_auc)
        best_overall_accs.append(max_overall_acc)
        best_es_accs.append(max_es_acc)

        # Print results for each seed
        print(f"Seed {seed_folder}: Best val_auc = {max_val_auc:.4f}")
        print(f"Seed {seed_folder}: Best val_auc for class 0 = {max_val_auc_0:.4f}")
        print(f"Seed {seed_folder}: Best val_auc for class 1 = {max_val_auc_1:.4f}")
        print(f"Seed {seed_folder}: Best val_auc for class 2 = {max_val_auc_2:.4f}")
        print(f"Seed {seed_folder}: Best ES-AUC = {max_es_auc:.4f}")
        print(f"Seed {seed_folder}: Best Overall Accuracy = {max_overall_acc:.4f}")
        print(f"Seed {seed_folder}: Best ES-ACC = {max_es_acc:.4f}")

# Calculate mean and standard deviation for each metric
def compute_mean_std(values):
    return sum(values) / len(values), pd.Series(values).std(ddof=1)

mean_val_auc, std_val_auc = compute_mean_std(best_val_aucs)
mean_val_auc_0, std_val_auc_0 = compute_mean_std(best_val_aucs_0)
mean_val_auc_1, std_val_auc_1 = compute_mean_std(best_val_aucs_1)
mean_val_auc_2, std_val_auc_2 = compute_mean_std(best_val_aucs_2)
mean_es_auc, std_es_auc = compute_mean_std(best_es_aucs)
mean_overall_acc, std_overall_acc = compute_mean_std(best_overall_accs)
mean_es_acc, std_es_acc = compute_mean_std(best_es_accs)

# Print aggregated statistics
print(f"\nMean of best val_auc values: {mean_val_auc:.4f} ± {std_val_auc:.4f}")
print(f"Mean of best val_auc values for class 0: {mean_val_auc_0:.4f} ± {std_val_auc_0:.4f}")
print(f"Mean of best val_auc values for class 1: {mean_val_auc_1:.4f} ± {std_val_auc_1:.4f}")
print(f"Mean of best val_auc values for class 2: {mean_val_auc_2:.4f} ± {std_val_auc_2:.4f}")
print(f"\nMean of best ES-AUC values: {mean_es_auc:.4f} ± {std_es_auc:.4f}")
print(f"Mean of best Overall Accuracy: {mean_overall_acc:.4f} ± {std_overall_acc:.4f}")
print(f"Mean of best ES-ACC values: {mean_es_acc:.4f} ± {std_es_acc:.4f}")
