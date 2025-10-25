from args import get_args
import os
import pandas as pd
import numpy as np
import torch
from dataset import Knee_Xray_dataset
from torch.utils.data import DataLoader
from model import MyModel
from sklearn.metrics import precision_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculate_test_precision(fold=0):
    """Calculate Precision"""
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 加载测试集
    test_csv_path = os.path.join(args.csv_dir, f'fold_{fold}_test.csv') 
    
    if not os.path.exists(test_csv_path):
        test_csv_path = os.path.join(args.csv_dir, f'fold_{fold}_val.csv')
        print(f"Using validation set as test set: {test_csv_path}")

    test_set = pd.read_csv(test_csv_path)
    test_dataset = Knee_Xray_dataset(test_set)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=torch.cuda.is_available())

    # 2. load model
    model_path = os.path.join(args.out_dir, f'best_model_fold_{fold}.pth')

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None

    model = MyModel(backbone=args.backbone).to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Make predictions
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['img'].to(device)
            if inputs.shape[1] == 3:
                inputs = inputs.mean(dim=1, keepdim=True)
            targets = batch['label'].to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())

    # 4. Calculate three types of Precision
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    precision_micro = precision_score(all_targets, all_preds, average='micro', zero_division=0)
    precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)

    # Per-Class Precision
    precision_per_class = precision_score(all_targets, all_preds, average=None, zero_division=0)

    # 5. Print results
    print("\n" + "=" * 60)
    print(f"PRECISION RESULTS - Fold {fold}")
    print("=" * 60)
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Micro Precision:    {precision_micro:.4f}")
    print(f"Weighted Precision: {precision_weighted:.4f}")
    print("\nPrecision per KL Grade:")
    for i, prec in enumerate(precision_per_class):
        print(f"  KL-{i}: {prec:.4f}")

    # 6. Detailed Classification Report
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=[f'KL-{i}' for i in range(5)],
                                digits=4))

    # 7. Save results to file
    results_dir = os.path.join(args.out_dir, "precision_results")
    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(results_dir, f"precision_results_fold_{fold}.txt")
    with open(results_file, 'w') as f:
        f.write("PRECISION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Fold: {fold}\n")
        f.write(f"Macro Precision: {precision_macro:.4f}\n")
        f.write(f"Micro Precision: {precision_micro:.4f}\n")
        f.write(f"Weighted Precision: {precision_weighted:.4f}\n\n")
        f.write("Precision per class:\n")
        for i, prec in enumerate(precision_per_class):
            f.write(f"  KL-{i}: {prec:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(all_targets, all_preds,
                                      target_names=[f'KL-{i}' for i in range(5)],
                                      digits=4))

    print(f"\nResults saved to: {results_file}")

    # 8. Plot confusion matrix
    plot_confusion_matrix(all_targets, all_preds, fold, results_dir)

    return {
        'fold': fold,
        'macro': precision_macro,
        'micro': precision_micro,
        'weighted': precision_weighted,
        'per_class': precision_per_class,
        'predictions': all_preds,
        'targets': all_targets
    }


def plot_confusion_matrix(true_labels, pred_labels, fold, save_dir):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'KL-{i}' for i in range(5)],
                yticklabels=[f'KL-{i}' for i in range(5)])
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    cm_path = os.path.join(save_dir, f"confusion_matrix_fold_{fold}.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved to: {cm_path}")


def calculate_average_precision_across_folds():
    """Calculate average Precision across all folds"""
    args = get_args()

    all_results = []

    for fold in range(5):
        print(f"\nProcessing Fold {fold}...")
        result = calculate_test_precision(fold)
        if result is not None:
            all_results.append(result)

    if not all_results:
        print("No results found!")
        return

    # Calculate average metrics
    macro_avg = np.mean([r['macro'] for r in all_results])
    micro_avg = np.mean([r['micro'] for r in all_results])
    weighted_avg = np.mean([r['weighted'] for r in all_results])

    # Calculate per-class average Precision
    per_class_avg = np.mean([r['per_class'] for r in all_results], axis=0)

    print("\n" + "=" * 60)
    print("AVERAGE PRECISION ACROSS ALL FOLDS")
    print("=" * 60)
    print(f"Average Macro Precision:    {macro_avg:.4f}")
    print(f"Average Micro Precision:    {micro_avg:.4f}")
    print(f"Average Weighted Precision: {weighted_avg:.4f}")
    print("\nAverage Precision per KL Grade:")
    for i, prec in enumerate(per_class_avg):
        print(f"  KL-{i}: {prec:.4f}")

    # Save average results
    results_dir = os.path.join(args.out_dir, "precision_results")
    avg_file = os.path.join(results_dir, "average_precision_results.txt")

    with open(avg_file, 'w') as f:
        f.write("AVERAGE PRECISION RESULTS ACROSS 5 FOLDS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Average Macro Precision: {macro_avg:.4f}\n")
        f.write(f"Average Micro Precision: {micro_avg:.4f}\n")
        f.write(f"Average Weighted Precision: {weighted_avg:.4f}\n\n")
        f.write("Average Precision per class:\n")
        for i, prec in enumerate(per_class_avg):
            f.write(f"  KL-{i}: {prec:.4f}\n")

    print(f"\nAverage results saved to: {avg_file}")

    return all_results



if __name__ == "__main__":
    # Calculate Precision for a single fold
    # result = calculate_test_precision(fold=0)

    # Calculate average Precision across all folds
    all_results = calculate_average_precision_across_folds()