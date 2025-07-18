import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_folder, image_list, transform=None, subset_size=None, stratify=False):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform

        # Define all 15 possible labels from the full dataset
        self.all_labels = [
            'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
            'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
            'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]


        # print("all labels", self.all_labels)

        # Filter dataset based on train/test split
        with open(image_list, "r") as f:
            self.image_names = set(f.read().splitlines())
        
        self.data = self.data[self.data["Image Index"].isin(self.image_names)]
        #print("data", self.data)

        # Apply subset sampling (only for training, not testing)
        if subset_size and len(self.data) > subset_size:
            self.data = self.data.sample(n=subset_size, random_state=42)

        # One-hot encode labels and ensure all 14 disease columns (exclude "No Finding" at first)
        label_columns = self.data["Finding Labels"].str.get_dummies(sep="|")

        # Define true diseases (exclude "No Finding")
        disease_labels = [l for l in self.all_labels if l != "No Finding"]

        # Ensure all columns exist
        for label in disease_labels:
            if label not in label_columns.columns:
                label_columns[label] = 0

        label_columns = label_columns[disease_labels]

        # Drop samples with no disease (all labels = 0)
        mask = label_columns.sum(axis=1) > 0
        self.data = self.data[mask].reset_index(drop=True)
        label_columns = label_columns[mask].reset_index(drop=True)


        # Derive "No Finding" as: 1 if all 14 disease labels are 0
        #label_columns["No Finding"] = (label_columns.sum(axis=1) == 0).astype(int)

        # Reorder columns to original order
        #label_columns = label_columns[self.all_labels]

        self.labels = label_columns.values

        #print("labels", self.labels)
        #print("self data", self.data)
                # Apply multilabel stratified sampling if requested
        if subset_size and len(self.data) > subset_size:
            if stratify:
                print("📊 Using Multilabel Stratified Sampling")
                msss = MultilabelStratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=42)
                indices, _ = next(msss.split(np.zeros(len(self.labels)), self.labels))
                self.data = self.data.iloc[indices].reset_index(drop=True)
                self.labels = self.labels[indices]
            else:
                print("🎲 Using Random Sampling")
                self.data = self.data.sample(n=subset_size, random_state=42).reset_index(drop=True)
                self.labels = self.labels[:subset_size]
        # Save paths and groups
        self.image_paths = self.data["Image Index"].values
        self.groups = self.data["Patient Gender"].apply(lambda x: 0 if x == "M" else 1).values.astype(np.int64)

        self.calculate_statistics()
        self.plot_class_distributions()

    def calculate_statistics(self):
        # Total number of samples
        total_samples = len(self.data)
        print(f"Total number of samples: {total_samples}")

        # Total number of unique IDs
        unique_ids = self.data['Patient ID'].nunique()
        print(f"Total number of unique IDs: {unique_ids}")

        # Gender distribution
        if "Patient Gender" in self.data.columns:
            male_count = (self.data["Patient Gender"] == 'M').sum()
            female_count = (self.data["Patient Gender"] == 'F').sum()
            print(f"Percentage of males: {(male_count / total_samples) * 100:.2f}%")
            print(f"Percentage of females: {(female_count / total_samples) * 100:.2f}%")

        # Disease prevalence
        print("self labels", self.labels)
        # axes 0 means it will sum the columns
        positive_counts = self.labels.sum(axis=0)
        print("\nTotal number of positive counts:", positive_counts)
        print("\nPrevalence of each disease:")
        for idx, label in enumerate(self.all_labels):
            prevalence = (positive_counts[idx] / total_samples) * 100
            print(f"{label}: {prevalence:.2f}%")

    def plot_class_distributions(self):
        # Overall class distribution
        positive_counts = self.labels.sum(axis=0)
        total_samples = len(self.data)
        prevalence = (positive_counts / total_samples) * 100

        plt.figure(figsize=(12, 6))
        sns.barplot(x=self.all_labels, y=prevalence, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Conditions')
        plt.ylabel('Prevalence (%)')
        plt.title('Overall Class Distribution')
        plt.tight_layout()
        plot_filename = f"dataset_distribution.png"
        output_dir = os.path.join(os.path.dirname(__file__), "dataset_distributions")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, plot_filename))


        # Class distribution by gender
        if "Patient Gender" in self.data.columns:
            gender_groups = self.data.groupby("Patient Gender")
            for gender, group in gender_groups:
                # One-hot encode labels within each gender group
                gender_labels = group["Finding Labels"].str.get_dummies(sep="|")

                # Ensure the columns are in the correct order as defined in self.all_labels
                for label in self.all_labels:
                    if label not in gender_labels.columns:
                        gender_labels[label] = 0  # Add missing labels as zeros
                gender_labels = gender_labels[self.all_labels]  # Reorder columns to match self.all_labels

                # Sum the occurrences of each disease within the gender group
                gender_counts = gender_labels.sum(axis=0)
                #print("gender labels", gender_labels)
                #print("gender counts", gender_counts)

                # Calculate prevalence as a percentage
                gender_prevalence = (gender_counts / len(group)) * 100
                print(gender)
                print("gender prevalence", gender_prevalence)

                # Plotting
                plt.figure(figsize=(12, 6))
                sns.barplot(x=self.all_labels, y=gender_prevalence, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Conditions')
                plt.ylabel('Prevalence (%)')
                plt.title(f'Class Distribution for {"Male" if gender == "M" else "Female"} Patients')
                plt.tight_layout()
                plot_filename = f"gender_distribution_{gender}.png"
                output_dir = os.path.join(os.path.dirname(__file__), "dataset_distributions")
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, plot_filename))
 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]["Image Index"]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")  # Ensure it's RGB

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        identity_group = 0 if self.data.iloc[idx]["Patient Gender"] == 'M' else 1
        identity_group = torch.tensor(identity_group, dtype=torch.long)

        return image, label, identity_group

# Define transformations
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet means
                         [0.229, 0.224, 0.225])  # ImageNet stds
])

if __name__ == "__main__":
    dataset = ChestXrayDataset(
        csv_file="/work3/s232437/fair-medical-AI-fin/NIH/Data_Entry_2017_v2020_.csv",
        image_folder="/work3/s232437/images_full/",
        image_list="/work3/s232437/fair-medical-AI-fin/NIH/train_val_list.txt",
        transform=transform
    )
    print(f"Dataset size: {len(dataset)}")
