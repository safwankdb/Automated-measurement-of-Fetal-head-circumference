import torch
import torch.utils.data
import numpy as np
from PIL import Image
import glob


class HC18(torch.utils.data.Dataset, typ):
	def __init__(self, transform=None):
		self.X_train = glob.glob('training_set/Train1/*HC.png')
		self.X_val = glob.glob('training_set/Train2/*HC.png')
		self.X_test = glob.glob('training_set/Test/*HC.png')
		self.y_train = glob.glob('training_set/Train1/*HC_Mask.png')
		self.y_val = glob.glob('training_set/Train2/*HC_Mask.png')
		self.y_test = glob.glob('training_set/Test/*HC_Mask.png')
	def __len__(self):
		if typ == "train":
			return len(self.X_train)
		elif typ == "val":
			return len(self.X_val)
		elif typ == "test":
			return len(self.X_test)
		else:
			raise ValueError("No such dataset")
	def __getitem__(self, idx):
		if typ == "train":
			X = np.array(Image.open(self.X_train[idx]))
			y = np.array(Image.open(self.y_train[idx]))
			return torch.from_numpy(X), torch.from_numpy(y)
		elif typ == "val":
			X = np.array(Image.open(self.X_val[idx]))
			y = np.array(Image.open(self.y_val[idx]))
			return torch.from_numpy(X), torch.from_numpy(y)
		elif typ == "test":
			X = np.array(Image.open(self.X_test[idx]))
			y = np.array(Image.open(self.y_test[idx]))
			return torch.from_numpy(X), torch.from_numpy(y)
		else:
			raise ValueError("No such datset")
