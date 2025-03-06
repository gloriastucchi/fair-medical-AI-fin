import numpy as np
import pandas as pd
import os

# Define the dataset path
dataset_path = "/your/path/to/dataset.npz"  # Change this to your actual file path

# Check if the file exists before loading
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at {dataset_path}")

# Load dataset (modify the key as needed)
data = np.load(dataset_path, allow_pickle=True)

# Check available keys
print("Available keys in the dataset:", data.files)

# Assuming the dataset has a structured array or dict format
df = pd.DataFrame(data["data"])  # Adjust the key if different

# Display the first few rows for reference
print(df.head())

# Define column names (update these based on your dataset structure)
race_column = "race"  
glaucoma_column = "glaucoma"

# Group by race and count glaucoma cases
prevalence = df.groupby(race_column)[glaucoma_column].value_counts().unstack()

# Rename columns for clarity
prevalence.columns = ["Non-Glaucoma", "Glaucoma"]

# Convert to percentages
prevalence_percentage = prevalence.div(prevalence.sum(axis=1), axis=0) * 100

# Print results
print("\nDisease Prevalence per Class:\n", prevalence_percentage)

# Plot the results
prevalence_percentage.plot(kind="bar", stacked=True, title="Disease Prevalence per Class (%)")  # Adjust plot settings as needed
