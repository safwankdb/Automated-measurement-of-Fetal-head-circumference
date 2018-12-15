import torch
import torch.utils.data
import numpy as np
from PIL import Image
import glob

X_size = (572, 572)
y_size = (388, 388)


class HC18(torch.utils.data.Dataset):
    def __init__(self, data_type, transform=None):
        self.X_train = glob.glob('training_set/Train1/*HC.png')
        self.X_val = glob.glob('training_set/Train2/*HC.png')
        self.X_test = glob.glob('training_set/Test/*HC.png')
        self.y_train = glob.glob('training_set/Train1/*HC_Mask.png')
        self.y_val = glob.glob('training_set/Train2/*HC_Mask.png')
        self.y_test = glob.glob('training_set/Test/*HC_Mask.png')
        self.data_type = data_type

    def __len__(self):
        if self.data_type == "train":
            return len(self.X_train)
        elif self.data_type == "val":
            return len(self.X_val)
        elif self.data_type == "test":
            return len(self.X_test)
        else:
            raise ValueError("No such dataset")

    def __getitem__(self, idx):
        if self.data_type == "train":

            X = np.array(Image.open(self.X_train[idx]).resize(
                X_size)).reshape(1, 572, 572)

            y = np.array(Image.open(self.y_train[idx]).convert(
                'L').resize(y_size)).reshape(1, 388, 388)

            print(X.shape, y.shape)
            return torch.from_numpy(X).float(), torch.from_numpy(y).float()
        elif self.data_type == "val":

            X = np.array(Image.open(self.X_val[idx]).resize(
                X_size)).reshape(1, 572, 572)

            y = np.array(Image.open(self.y_val[idx]).convert(
                'L').resize(y_size)).reshape(1, 388, 388)

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()
        elif self.data_type == "test":

            X = np.array(Image.open(self.X_test[idx]).resize(
                X_size)).reshape(1, 572, 572)

            y = np.array(Image.open(self.y_test[idx]).convert(
                'L').resize(y_size)).reshape(1, 388, 388)

            return torch.from_numpy(X).float(), torch.from_numpy(y).float()
        else:
            raise ValueError("No such datset")
