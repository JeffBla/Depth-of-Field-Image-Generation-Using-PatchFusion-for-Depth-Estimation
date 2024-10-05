import os
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import albumentations as A
from albumentations.pytorch import ToTensorV2


class DoFDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir: str = "data/",
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_split: float = 0.8,
        image_size: int = 256,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_split = train_val_split
        self.image_size = image_size

        self.transform = A.Compose([
            A.Resize(height=self.image_size, width=self.image_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2(),
        ])

    def setup(self, stage: Optional[str] = None):
        # Load all data
        full_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "processed"),
            transform=lambda x: self.transform(image=x)["image"])

        # Split into train and validation sets
        train_size = int(len(full_dataset) * self.train_val_split)
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        # Assuming test set is in a separate folder
        test_dataset = ImageFolder(
            root=os.path.join(self.data_dir, "test"),
            transform=lambda x: self.transform(image=x)["image"])
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
