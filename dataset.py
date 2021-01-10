import torch.utils.data as data
import numpy as np
from torchvision import transforms


class CoinrunDataset(data.Dataset):
    def __init__(self, filepath, split='train', reduced=False, transform=None):
        x = np.load(filepath)['obs']  # (400, 32, 64, 64, 3)
        self.data = x.reshape(-1, *x.shape[2:])  # (12800, 64, 64, 3)
        np.random.seed(0)
        np.random.shuffle(self.data)
        N = self.data.shape[0]
        if split == 'train':
            self.data = self.data[:int(0.8*N)]
        elif split == 'test':
            self.data = self.data[int(0.8*N):]
        else:
            raise ValueError(f'Unknown value split="{split}"; split must be "train" or "test".')
        if reduced:
            self.data = self.data[:300]

        self.transform = transform if transform else lambda x:x

    def __getitem__(self, index):
        return self.transform(self.data[index])

    def __len__(self):
        return self.data.shape[0]
