import pytorch_lightning as pl
import torch
from PIL import Image
from src.utils import transform_augmentation, transform_inference
from torch.utils.data import DataLoader, Dataset


class Covid19(Dataset):
    def __init__(self, x, y, transforms):
        self.x = x
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.x[idx]
        image = Image.open(image).convert("RGB")
        image = self.transforms(image)

        label = self.y[idx]
        return image, label


class Covid19DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train_x, val_x, test_x, train_y, val_y, test_y):
        super().__init__()
        self.batch_size = batch_size
        self.train_x = train_x
        self.val_x = val_x
        self.test_x = test_x
        self.train_y = train_y
        self.val_y = val_y
        self.test_y = test_y

        # Augmentation policy for training set
        self.augmentation = transform_augmentation
        # Preprocessing steps applied to validation and test set.
        self.transform = transform_inference

        self.num_classes = 3

    def prepare_data(self):
        self.train = Covid19(x=self.train_x, y=self.train_y, transforms=self.augmentation)
        self.valid = Covid19(x=self.val_x, y=self.val_y, transforms=self.transform)
        self.test = Covid19(x=self.test_x, y=self.test_y, transforms=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
