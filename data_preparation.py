import os
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt

metadata = pd.read_csv("metadata.csv")

# output directories
output_dir = "CVs"
csv_dir = "CSVs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(csv_dir, exist_ok=True)

# divide into train+val and test sets
train_val_data, test_data = train_test_split(
    metadata,
    test_size=0.2,
    random_state=42,
    stratify=metadata["KL"]
)

# Save global test.csv
test_data.to_csv(os.path.join(csv_dir, "test.csv"), index=False)

def plot_distribution(df, title, filename):
    plt.figure(figsize=(6,4))
    df["KL"].value_counts().sort_index().plot(kind="bar")
    plt.title(title)
    plt.xlabel("KL Grade")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X = train_val_data.index.values
y = train_val_data["KL"].values

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    train_fold = train_val_data.iloc[train_idx]
    val_fold = train_val_data.iloc[val_idx]

    print(f"Fold {fold}: Train samples = {len(train_fold)}, Validation samples = {len(val_fold)}")

    # Save CSV files
    train_fold.to_csv(os.path.join(csv_dir, f"fold_{fold}_train.csv"), index=False)
    val_fold.to_csv(os.path.join(csv_dir, f"fold_{fold}_val.csv"), index=False)

    # Plot distribution
    plot_distribution(train_fold, f"Train fold {fold} KL", f"fold_{fold}_train.png")
    plot_distribution(val_fold, f"Validation fold {fold} KL", f"fold_{fold}_val.png")