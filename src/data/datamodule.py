import os
from typing import Optional, Tuple
import lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch
import numpy as np

from SyncTransfom import SyncRandomFlip, SyncRandomRotation, SyncColorJitter

class DofDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, augment=True):
        """
        Args:
            root_dir (str): Path to the proc directory containing blur, clear, and depth folders
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Get all image files from blur directory
        self.blur_dir = os.path.join(root_dir, 'blur')
        self.clear_dir = os.path.join(root_dir, 'clear')
        self.depth_dir = os.path.join(root_dir, 'depth')
        
        self.image_files = sorted(os.listdir(self.blur_dir))

        # Define transforms
        # Define synchronized augmentations
        self.augment = augment
        self.sync_transforms = [
            SyncRandomFlip(p=0.5),
            SyncRandomRotation(degrees=10),
        ]

        # Define color augmentation (only for RGB images, not depth)
        self.color_transform = SyncColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        )

        # Define basic transforms
        self.basic_transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load images
        blur_path = os.path.join(self.blur_dir, img_name)
        clear_path = os.path.join(self.clear_dir, img_name)
        depth_path = os.path.join(self.depth_dir, img_name)
        
        blur_img = Image.open(blur_path).convert('RGB')
        clear_img = Image.open(clear_path).convert('RGB')
        depth_img = Image.open(depth_path).convert('L')  # Load as grayscale
        
        if self.transform:
            blur_img = self.transform(blur_img)
            clear_img = self.transform(clear_img)
            depth_img = self.transform(depth_img)

        # Concatenate clear and depth images for input
        real_A = torch.cat([clear_img, depth_img], dim=0)  # [4, H, W] - RGB + Depth
        real_B = blur_img  # [3, H, W] - RGB
        
        return real_A, real_B
    
    def apply_transforms(self, images):
        # Apply synchronized geometric transformations
        if self.augment:
            for transform in self.sync_transforms:
                images = transform(images)
            images = self.color_transform(images)

        tensors = []
        for img in images:
            tensors.append(self.basic_transform(img))
            
        return tensors
    
class DofDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/proc",
        batch_size: int = 8,
        num_workers: int = 4,
        img_size: Tuple[int, int] = (1024, 1024),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
             # Create full dataset
            full_dataset = DofDataset(
                root_dir=os.path.join(self.data_dir,),
                transform=self.transform,
                augment=True
            )
            
            # Calculate lengths for splits
            total_length = len(full_dataset)
            train_length = int(total_length * self.train_val_split[0])
            val_length = total_length - train_length
            
            # Split dataset
            self.train_dataset, self.test_dataset = random_split(
                full_dataset, 
                [train_length, val_length],
                generator=torch.Generator().manual_seed(42)
            )
        elif stage == 'test':
            self.test_dataset = DofDataset(
                os.path.join(self.data_dir, 'test'),
                transform=self.transform
            )
            
        elif stage == 'predict':
            self.predict_dataset = DofDataset(
                os.path.join(self.data_dir, 'test'),
                transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )