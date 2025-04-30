import pandas as pd
from pathlib import Path
import os
import sys

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def merge_cleaned_data():
    """
    Merge the cleaned training features and training labels into one dataset.
    """
    # Define paths
    processed_dir = Path("data/processed")
    
    # Load cleaned datasets
    train_features = pd.read_csv(processed_dir / "cleaned_train_features.csv")
    train_labels = pd.read_csv(processed_dir / "cleaned_train_labels.csv")
    
    # Merge training features and labels
    merged_data = pd.merge(train_features, train_labels, on=['city', 'year', 'weekofyear'])
    
    # Save merged dataset
    merged_data.to_csv(processed_dir / "cleaned_data_merged.csv", index=False)
    print(f"Merged dataset saved to {processed_dir / 'cleaned_data_merged.csv'}")
    print(f"Shape of merged dataset: {merged_data.shape}")

if __name__ == "__main__":
    merge_cleaned_data() 