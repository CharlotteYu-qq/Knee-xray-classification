from args import get_args
import os
import pandas as pd
import numpy as np
from dataset import Knee_Xray_dataset
from torch.utils.data import DataLoader
from model import MyModel
from trainer import train_model
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def main():
    # 1.we need some arguments
    args = get_args()

    os.makedirs(args.out_dir, exist_ok=True)

    fold_performances = []

    # 2.iterate among the folds
    for fold in range(5):
        print('='* 50)
        print(f'Training fold: , {fold}')
        print('=' * 50)

        train_csv_path = os.path.join(args.csv_dir, f'fold_{fold}_train.csv')
        val_csv_path = os.path.join(args.csv_dir, f'fold_{fold}_val.csv')
        train_set = pd.read_csv(train_csv_path)
        val_set = pd.read_csv(val_csv_path)

        # 3.prepare datasets
        train_dataset = Knee_Xray_dataset(train_set)
        val_dataset = Knee_Xray_dataset(val_set)

        # 4.create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, pin_memory=torch.cuda.is_available())

        # 5.initialize the model
        model = MyModel(backbone=args.backbone).to(device)

        # 6.train the model
        best_balanced_acc = train_model(model, train_loader, val_loader, fold)

        fold_performances.append(best_balanced_acc)

        print(f'Fold {fold} done! Best balanced accuracy is {best_balanced_acc:.4f}')
        print()

    print('=' * 50)
    print(f'5 fold cross validation done!')
    print(f'Best balanced accuracy for each fold: ')
    for i, acc in enumerate(fold_performances):
        print(f'Fold {i}: {acc:.4f}')

    mean_acc = sum(fold_performances) / len(fold_performances)
    std_acc = np.std(fold_performances) if len(fold_performances) > 1 else 0
    print(f'average balanced accuracy is {mean_acc:.4f} +- {std_acc:.4f}')
    print('=' * 50)




if __name__ == '__main__':
    main()