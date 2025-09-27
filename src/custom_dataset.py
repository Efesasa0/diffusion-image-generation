import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

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

sprites_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
])

