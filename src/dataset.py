import ast
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pycocotools import mask as coco_mask
from typing import List, Dict, TypedDict, Callable, Optional


class DataBatch(TypedDict):
    """
    image_id: List[str], len batch_size
    image: torch.Tensor, shape (batch_size, 3, image_size, image_size)
    mask: torch.Tensor, shape (batch_size, num_classes, image_size, image_size)
    """

    image_id: List[str]
    image: torch.Tensor
    mask: torch.Tensor


class HierTextDataset(Dataset):
    """A class represents hiertext dataset"""

    categories = ["paragraph", "line", "word"]

    def __init__(
        self, image_dir: str, label_path: str, transform: Optional[Callable] = None
    ):
        self.image_dir = image_dir
        self.label_df = pd.read_csv(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.label_df.iloc[idx]
        image_id, h, w = row["image_id"], row["height"], row["width"]
        image = Image.open(str(Path(self.image_dir) / f"{image_id}.jpg"))
        image = np.asarray(image)
        mask = []
        for cat in self.categories:
            rle = {"size": [h, w], "counts": ast.literal_eval(row[cat])}
            mask.append(coco_mask.decode(rle))
        mask = np.stack(mask, axis=-1)
        result = dict(image_id=image_id, image=image, mask=mask)
        if self.transform is not None:
            result = self.transform(**result)
        return result


class HierTextDataModule:
    """Handle transform, data loader, sampler, etc."""

    def __init__(
        self,
        dataset_dir: str,
        label_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
    ):
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        augment_transform = [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.5,
                rotate_limit=30,
                p=0.5,
            ),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(p=0.5),
        ]
        postprocess_transform = [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
            ),
            ToTensorV2(transpose_mask=True),
        ]
        self.train_transform = A.Compose(augment_transform + postprocess_transform)
        self.val_transform = A.Compose(postprocess_transform)

    def train_dataloader(self, use_distributed_sampler: bool) -> DataLoader:
        """Get train dataloader

        Args:
            use_distributed_sampler (bool): use distributed sampler or not (for distributed data parallel)

        Returns:
            DataLoader: train dataloader
        """
        train_ds = HierTextDataset(
            image_dir=str(Path(self.dataset_dir) / "train"),
            label_path=str(Path(self.label_dir) / "train.csv"),
            transform=self.train_transform,
        )
        shuffle = False if use_distributed_sampler else True
        sampler = DistributedSampler(train_ds) if use_distributed_sampler else None
        return DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=sampler,
        )

    def val_dataloader(self, use_distributed_sampler: bool) -> DataLoader:
        """Get validation dataloader

        Args:
            use_distributed_sampler (bool): use distributed sampler or not (for distributed data parallel)

        Returns:
            DataLoader: validation dataloader
        """
        val_ds = HierTextDataset(
            image_dir=str(Path(self.dataset_dir) / "validation"),
            label_path=str(Path(self.label_dir) / "validation.csv"),
            transform=self.val_transform,
        )
        shuffle = False
        sampler = DistributedSampler(val_ds) if use_distributed_sampler else None
        return DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            sampler=sampler,
        )
