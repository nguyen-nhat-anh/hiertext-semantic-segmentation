import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Dict

from src.dataset import DataBatch


class HierTextModelModule(nn.Module):
    categories = ["paragraph", "line", "word"]

    def __init__(self, encoder_name):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(self.categories),
        )
        self.loss_fn = smp.losses.DiceLoss(mode="multilabel")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch: DataBatch, batch_idx) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["mask"]
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        return dict(outputs=outputs, loss=loss, images=images, targets=targets)

    def validation_step(self, batch: DataBatch, batch_idx) -> Dict[str, torch.Tensor]:
        images = batch["image"]
        targets = batch["mask"]
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        return dict(outputs=outputs, loss=loss, images=images, targets=targets)
