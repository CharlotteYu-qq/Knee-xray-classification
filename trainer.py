from args import get_args
import torch
import torch.nn as nn
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, precision_score
import numpy as np
from utils import plot_training_metrics
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, fold=0):
    args = get_args()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # move model to device
    model.to(device)

    # metric lists
    train_losses = []
    val_losses = []
    train_balanced_accs = []
    val_balanced_accs = []
    train_roc_aucs = []
    val_roc_aucs = []
    train_avg_precisions = []
    val_avg_precisions = []
    train_precision_macro = []
    val_precision_macro = []
    train_precision_micro = []
    val_precision_micro = []
    train_precision_weighted = []
    val_precision_weighted = []

    best_balanced_accuracy = 0
    best_model_path = ""

    for epoch in range(args.epochs):
        training_loss = 0
        all_train_preds = []
        all_train_targets = []
        all_train_probs = []

        # train model
        model.train()

        for batch in train_loader:
            inputs = batch['img'].to(device)  # move to device
            targets = batch['label'].to(device)  # move to device

            # 3 channels to 1 channel
            if inputs.shape[1] == 3:
                inputs = inputs.mean(dim=1, keepdim=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_train_preds.extend(predicted.cpu().numpy())
            all_train_targets.extend(targets.cpu().numpy())
            all_train_probs.extend(probabilities.cpu().detach().numpy())

        # calculate training metrics
        train_epoch_loss = training_loss / len(train_loader)
        train_balanced_accuracy = balanced_accuracy_score(all_train_targets, all_train_preds)
        
        # calculate Precision
        train_precision_macro_value = precision_score(all_train_targets, all_train_preds, average='macro', zero_division=0)
        train_precision_micro_value = precision_score(all_train_targets, all_train_preds, average='micro', zero_division=0)
        train_precision_weighted_value = precision_score(all_train_targets, all_train_preds, average='weighted', zero_division=0)

        all_train_probs = np.array(all_train_probs)
        all_train_targets = np.array(all_train_targets)

        try:
            train_roc_auc = roc_auc_score(all_train_targets, all_train_probs, multi_class='ovr', average='macro')
        except ValueError as e:
            print(f"ROC-AUC calculation fail: {e}, use default value 0.5")
            train_roc_auc = 0.5

        try:
            train_avg_precision = average_precision_score(all_train_targets, all_train_probs, average='macro')
        except ValueError as e:
            print(f"Average Precision calculation fail: {e}, use default value 0.0")
            train_avg_precision = 0.0

        # save training metrics
        train_losses.append(train_epoch_loss)
        train_balanced_accs.append(train_balanced_accuracy)
        train_roc_aucs.append(train_roc_auc)
        train_avg_precisions.append(train_avg_precision)
        train_precision_macro.append(train_precision_macro_value)
        train_precision_micro.append(train_precision_micro_value)
        train_precision_weighted.append(train_precision_weighted_value)

        print(f'Epoch {epoch+1}: Train Loss: {train_epoch_loss:.4f}, '
              f'Train Balanced Acc: {train_balanced_accuracy:.4f}')

        # validation phase
        val_metrics = validate_model_with_metrics(model, val_loader, criterion)
        val_losses.append(val_metrics['loss'])
        val_balanced_accs.append(val_metrics['balanced_accuracy'])
        val_roc_aucs.append(val_metrics['roc_auc'])
        val_avg_precisions.append(val_metrics['avg_precision'])
        val_precision_macro.append(val_metrics['precision_macro'])
        val_precision_micro.append(val_metrics['precision_micro'])
        val_precision_weighted.append(val_metrics['precision_weighted'])

        print(f'Validation Loss: {val_metrics["loss"]:.4f}, '
              f'Balanced Accuracy: {val_metrics["balanced_accuracy"]:.4f}')

        if val_metrics['balanced_accuracy'] > best_balanced_accuracy:
            best_balanced_accuracy = val_metrics['balanced_accuracy']
            os.makedirs(args.out_dir, exist_ok=True)
            best_model_path = os.path.join(args.out_dir, f'best_model_fold_{fold}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'balanced_accuracy': best_balanced_accuracy,
                **val_metrics
            }, best_model_path)
            print(f'Best model saved! Best balanced accuracy: {best_balanced_accuracy:.4f}')

    # plot
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        success = plot_training_metrics(
            train_losses, val_losses,
            train_balanced_accs, val_balanced_accs,
            train_roc_aucs, val_roc_aucs,
            train_avg_precisions, val_avg_precisions,
            train_precision_macro, val_precision_macro,
            train_precision_micro, val_precision_micro,
            train_precision_weighted, val_precision_weighted,
            args.out_dir, fold
        )
        if success:
            print("Plotting successful!")
        else:
            print("Plotting failed!")
    except Exception as e:
        print(f"Plotting error: {e}")

    return best_balanced_accuracy

def validate_model_with_metrics(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_targets = []
    all_val_probs = []

    with torch.no_grad():
        for batch in val_loader:
            inputs = batch['img'].to(device)  # move to device
            targets = batch['label'].to(device)  # move to device

            # 3 channels to 1 channel
            if inputs.shape[1] == 3:
                inputs = inputs.mean(dim=1, keepdim=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_val_preds.extend(predicted.cpu().numpy())
            all_val_targets.extend(targets.cpu().numpy())
            all_val_probs.extend(probabilities.cpu().numpy())

    val_epoch_loss = val_loss / len(val_loader)
    val_balanced_accuracy = balanced_accuracy_score(all_val_targets, all_val_preds)
    
    # calculate 3 Precision
    val_precision_macro = precision_score(all_val_targets, all_val_preds, average='macro', zero_division=0)
    val_precision_micro = precision_score(all_val_targets, all_val_preds, average='micro', zero_division=0)
    val_precision_weighted = precision_score(all_val_targets, all_val_preds, average='weighted', zero_division=0)

    all_val_probs = np.array(all_val_probs)
    all_val_targets = np.array(all_val_targets)

    try:
        val_roc_auc = roc_auc_score(all_val_targets, all_val_probs, multi_class='ovr', average='macro')
    except ValueError as e:
        print(f"Validate ROC-AUC calculation fail: {e}, use default value 0.5")
        val_roc_auc = 0.5

    try:
        val_avg_precision = average_precision_score(all_val_targets, all_val_probs, average='macro')
    except ValueError as e:
        print(f"Validate Average Precision calculation fail: {e}, use default value 0.0")
        val_avg_precision = 0.0

    return {
        'loss': val_epoch_loss,
        'balanced_accuracy': val_balanced_accuracy,
        'roc_auc': val_roc_auc,
        'avg_precision': val_avg_precision,
        'precision_macro': val_precision_macro,
        'precision_micro': val_precision_micro,
        'precision_weighted': val_precision_weighted
    }