import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import numpy as np
from PIL import Image


class ExemplarSet(Dataset):
    def __init__(self, data=None, targets=None, transform=None):
        self.data = data
        self.targets = targets
        self.transform = None if transform is None else transforms.Compose(transform)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = Image.fromarray(self.data[index].astype(np.uint8))
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

    def update_data_memory(self, new_data, new_targets):
        assert new_data.shape[0] == new_targets.shape[0]
        self.data = new_data if self.data is None else np.concatenate((self.data, new_data))
        self.targets = new_targets if self.targets is None else np.concatenate((self.targets, new_targets))
