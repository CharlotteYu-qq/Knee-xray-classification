
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_training_metrics(train_losses, val_losses,
                         train_balanced_accs, val_balanced_accs,
                         train_roc_aucs, val_roc_aucs,
                         train_avg_precisions, val_avg_precisions,
                         train_precision_macro, val_precision_macro,
                         train_precision_micro, val_precision_micro,
                         train_precision_weighted, val_precision_weighted,
                         out_dir, fold):
    """
    plot training and validation metrics in a 3x3 grid layout and save the figure.
    """
    try:
        epochs = range(1, len(train_losses) + 1)

        # Create 3x3 subplots to accommodate all metrics
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))

        # 1. Loss curve
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epochs')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Balanced Accuracy curve
        axes[0, 1].plot(epochs, train_balanced_accs, 'b-', label='Training Balanced Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, val_balanced_accs, 'r-', label='Validation Balanced Accuracy', linewidth=2)
        axes[0, 1].set_title('Training and Validation Balanced Accuracy')
        axes[0, 1].set_xlabel('Epochs')
        axes[0, 1].set_ylabel('Balanced Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. ROC-AUC curve
        axes[0, 2].plot(epochs, train_roc_aucs, 'b-', label='Training ROC-AUC', linewidth=2)
        axes[0, 2].plot(epochs, val_roc_aucs, 'r-', label='Validation ROC-AUC', linewidth=2)
        axes[0, 2].set_title('Training and Validation ROC-AUC')
        axes[0, 2].set_xlabel('Epochs')
        axes[0, 2].set_ylabel('ROC-AUC Score')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

        # 4. Average Precision curve
        axes[1, 0].plot(epochs, train_avg_precisions, 'b-', label='Training Average Precision', linewidth=2)
        axes[1, 0].plot(epochs, val_avg_precisions, 'r-', label='Validation Average Precision', linewidth=2)
        axes[1, 0].set_title('Training and Validation Average Precision')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Average Precision Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 5. Macro Precision curve
        axes[1, 1].plot(epochs, train_precision_macro, 'b-', label='Training Precision (Macro)', linewidth=2)
        axes[1, 1].plot(epochs, val_precision_macro, 'r-', label='Validation Precision (Macro)', linewidth=2)
        axes[1, 1].set_title('Training and Validation Precision (Macro)')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Precision Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Micro Precision curve
        axes[1, 2].plot(epochs, train_precision_micro, 'b-', label='Training Precision (Micro)', linewidth=2)
        axes[1, 2].plot(epochs, val_precision_micro, 'r-', label='Validation Precision (Micro)', linewidth=2)
        axes[1, 2].set_title('Training and Validation Precision (Micro)')
        axes[1, 2].set_xlabel('Epochs')
        axes[1, 2].set_ylabel('Precision Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        # 7. Weighted Precision curve
        axes[2, 0].plot(epochs, train_precision_weighted, 'b-', label='Training Precision (Weighted)', linewidth=2)
        axes[2, 0].plot(epochs, val_precision_weighted, 'r-', label='Validation Precision (Weighted)', linewidth=2)
        axes[2, 0].set_title('Training and Validation Precision (Weighted)')
        axes[2, 0].set_xlabel('Epochs')
        axes[2, 0].set_ylabel('Precision Score')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # 8. Comparison of Three Precision Types (Validation)
        axes[2, 1].plot(epochs, val_precision_macro, 'g-', label='Validation Precision (Macro)', linewidth=2)
        axes[2, 1].plot(epochs, val_precision_micro, 'orange', label='Validation Precision (Micro)', linewidth=2)
        axes[2, 1].plot(epochs, val_precision_weighted, 'purple', label='Validation Precision (Weighted)', linewidth=2)
        axes[2, 1].set_title('Comparison of Three Precision Types (Validation)')
        axes[2, 1].set_xlabel('Epochs')
        axes[2, 1].set_ylabel('Precision Score')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # 9. Leave empty or display final values
        axes[2, 2].axis('off')
        # You can display the final summary of values here
        final_text = f'Final Validation Metrics:\n'
        final_text += f'Macro Precision: {val_precision_macro[-1]:.4f}\n'
        final_text += f'Micro Precision: {val_precision_micro[-1]:.4f}\n'
        final_text += f'Weighted Precision: {val_precision_weighted[-1]:.4f}\n'
        final_text += f'Balanced Accuracy: {val_balanced_accs[-1]:.4f}'
        axes[2, 2].text(0.1, 0.5, final_text, fontsize=12, verticalalignment='center')

        # Adjust layout and save
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f'training_metrics_fold_{fold}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'training plot saved: {plot_path}')
        return True

    except Exception as e:
        print(f'plot fail: {e}')
        return False