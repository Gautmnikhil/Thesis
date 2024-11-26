import numpy as np
import torch
from torch.utils.data import Dataset
from math import ceil

class ProcessedUserLoader(Dataset):
    def __init__(self, data_path, mode, win_size, step):
        # Load the dataset from .npy file
        self.data = np.load(data_path, allow_pickle=True)  # Load .npy file

        # Remove non-numeric columns (e.g., timestamps)
        if isinstance(self.data, np.ndarray) and len(self.data.shape) > 1:
            numeric_data = []
            for row in self.data:
                # Filter out non-numeric values (e.g., strings)
                numeric_data.append([x for x in row if isinstance(x, (int, float))])
            self.data = np.array(numeric_data, dtype=np.float32)

        # Store parameters
        self.mode = mode
        self.win_size = win_size
        self.step = step

        # Split data into train, val, test, threshold segments as required
        num_samples = len(self.data)
        train_end = int(0.6 * num_samples)
        val_end = int(0.2 * num_samples) + train_end
        thre_end = int(0.1 * num_samples) + val_end

        self.train = self.data[:train_end]
        self.val = self.data[train_end:val_end]
        self.thre = self.data[val_end:thre_end]
        self.test = self.data[thre_end:]

        # Shuffle training data for better generalization
        if self.mode == "train":
            np.random.shuffle(self.train)

    def __len__(self):
        # Calculate length based on the mode and step size
        if self.mode == "train":
            return ceil(len(self.train) / self.step)
        elif self.mode == 'val':
            return ceil(len(self.val) / self.step)
        elif self.mode == 'test':
            return ceil(len(self.test) / self.step)
        elif self.mode == 'thre':
            return ceil(len(self.thre) / self.step)
        else:
            return 0

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            data = self.train[index:index + self.win_size]
        elif self.mode == 'val':
            data = self.val[index:index + self.win_size]
        elif self.mode == 'test':
            data = self.test[index:index + self.win_size]
        elif self.mode == 'thre':
            data = self.thre[index:index + self.win_size]
        else:
            return None

        # Padding if the length of data is less than win_size
        if len(data) < self.win_size:
            padding = np.full((self.win_size - len(data), data.shape[1]), -1, dtype=np.float32)  # Use distinct value for padding
            data = np.concatenate((data, padding), axis=0)

        # Make sure the data has the shape [sequence_length, feature_dim]
        data = np.float32(data)
        # Return only input data
        return torch.tensor(data).permute(1, 0)