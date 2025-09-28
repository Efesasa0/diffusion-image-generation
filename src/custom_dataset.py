import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F

class SpritesDataset(Dataset):
    def __init__(self, images_path, labels_path, transform, null_context):
        self.images = np.load(images_path, allow_pickle=False)
        self.labels = np.load(labels_path, allow_pickle=False)

        self.images_shape = self.images.shape
        self.labels_shape = self.labels.shape

        self.transform = transform
        self.null_context = null_context
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = torch.tensor(self.labels[idx]).to(torch.int64)
        return image, label
    def __getshape__(self):
        return self.images_shape, self.labels_shape

class CIFAR10OneHot(Dataset):
    def __init__(self, base_dataset, num_classes=10):
        self.base = base_dataset
        self.num_classes = num_classes
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        onehot = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return img, onehot