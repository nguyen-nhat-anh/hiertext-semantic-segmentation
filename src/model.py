import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict

from src.dataset import DataBatch


class HierTextModelModule(nn.Module):
    """A class represents our semantic segmentation model"""

    categories = ["paragraph", "line", "word"]

    def __init__(self, encoder_name: str):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(self.categories),
        )
        self.loss_fn = smp.losses.DiceLoss(mode="multilabel")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: DataBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Perform a single training step

        Args:
            batch (DataBatch): data batch
            batch_idx (int): batch index

        Returns:
            Dict[str, torch.Tensor]: output dict, keys: outputs, loss, images, targets
        """
        images = batch["image"]
        targets = batch["mask"]
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        return dict(outputs=outputs, loss=loss, images=images, targets=targets)

    def validation_step(
        self, batch: DataBatch, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Perform a single validation step

        Args:
            batch (DataBatch): data batch
            batch_idx (int): batch index

        Returns:
            Dict[str, torch.Tensor]: output dict, keys: outputs, loss, images, targets
        """
        images = batch["image"]
        targets = batch["mask"]
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        return dict(outputs=outputs, loss=loss, images=images, targets=targets)
