import pandas as pd
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch


import pandas as pd
from torch.utils.data import Dataset
import cv2
import numpy as np
import torch

def read_xray(path):
    xray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # read gray image [H, W]
    return xray

class Knee_Xray_dataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = read_xray(self.dataset['Path'].iloc[idx])  # shape: [H, W]
        label = self.dataset['KL'].iloc[idx]

        # normalize and add channel dimension [1, H, W]
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # from [H, W] to [1, H, W]

        res = {
            'img': torch.from_numpy(img).float(),  # [1, H, W], ensure float type
            'label': label
        }
        return res

