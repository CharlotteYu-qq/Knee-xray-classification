import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Model training options')

    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'])

    parser.add_argument('--csv_dir', type=str, default='CSVs')
    parser.add_argument('--batch_size', type=int, default=32,
                        choices=[16, 32, 64])

    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--epochs', type=int, default=50)


    parser.add_argument('--out_dir', type=str, default='session')

    args = parser.parse_args()
    return args



