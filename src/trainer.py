import argparse
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import List, Dict

from src.callback import Callbacks, Callback
from src.dataset import DataBatch, HierTextDataModule
from src.utils import MeterDict, logger


class HierTextModelModule:
    categories = ["paragraph", "line", "word"]

    def __init__(self, encoder_name, device, lr, amp, batch_size, epochs, **kwargs):
        self.device = (
            torch.device(f"cuda:{device}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=3,
            classes=len(self.categories),
        ).to(self.device)
        self.lr = lr
        self.amp = amp if torch.cuda.is_available() else False
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_fn = smp.losses.DiceLoss(mode="multilabel")

        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.stop_training = None  # for early stopping

    def save_weights(self, weights_path):
        torch.save(self.model.state_dict(), weights_path)

    def load_weights(self, weights_path):
        weights = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(weights)

    def configure(self, steps_per_epoch):
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=self.lr
        )
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            pct_start=0.3,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
        )
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.amp
        )  # for mixed precision training

    def training_step(self, batch: DataBatch, batch_idx) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        targets = batch["mask"].to(self.device)
        with torch.cuda.amp.autocast(enabled=self.amp):
            outputs = self.model(images)
            loss = self.loss_fn(outputs, targets)
        return dict(outputs=outputs, loss=loss, images=images, targets=targets)

    def validation_step(self, batch: DataBatch, batch_idx) -> Dict[str, torch.Tensor]:
        images = batch["image"].to(self.device)
        targets = batch["mask"].to(self.device)
        outputs = self.model(images)
        loss = self.loss_fn(outputs, targets)
        return dict(outputs=outputs, loss=loss, images=images, targets=targets)

    def training_epoch(self, loader: DataLoader):
        self.model.train()
        summaries = MeterDict()
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:
            result = self.training_step(batch, step)
            loss, targets = result["loss"], result["targets"]
            # backpropagation
            self.optimizer.zero_grad()
            if self.amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            # lr scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            # update stats
            batch_size = targets.size(0)
            summaries.update("loss", loss.item(), batch_size)
            lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(lr=lr, **summaries.avg)
        summaries_avg = summaries.avg
        return summaries_avg

    @torch.inference_mode()
    def validation_epoch(self, loader: DataLoader):
        self.model.eval()
        summaries = MeterDict()
        pbar = tqdm(enumerate(loader), total=len(loader))
        for step, batch in pbar:
            result = self.validation_step(batch, step)
            outputs, loss, targets = (
                result["outputs"],
                result["loss"],
                result["targets"],
            )
            # update stats
            probs = torch.sigmoid(outputs)
            stats = smp.metrics.get_stats(
                probs, targets, mode="multilabel", threshold=0.5
            )
            iou_per_class: torch.Tensor = smp.metrics.iou_score(
                *stats, reduction=None
            ).mean(axis=0)
            batch_size = targets.size(0)
            summaries.update("loss", loss.item(), batch_size)
            for cat, iou in zip(self.categories, iou_per_class):
                summaries.update(f"{cat}_iou", iou.item(), batch_size)
            summaries.update(f"macro_avg_iou", iou_per_class.mean().item(), batch_size)
            pbar.set_postfix(**summaries.avg)
        summaries_avg = summaries.avg
        return summaries_avg

    def fit(
        self, data_module: HierTextDataModule, epochs: int, callbacks_: List[Callback]
    ):
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()
        self.configure(steps_per_epoch=len(train_loader))
        callbacks = Callbacks(callbacks_)
        callbacks.set_model_module(self)
        callbacks.on_train_begin()
        for epoch in range(1, epochs + 1):
            callbacks.on_epoch_begin(epoch=epoch)
            logger.info("# Epoch {}/{}: #".format(epoch, epochs))
            train_summaries = self.training_epoch(train_loader)
            val_summaries = self.validation_epoch(val_loader)
            callbacks.on_epoch_end(
                epoch=epoch, logs={"val_loss": val_summaries["loss"]}
            )
            if self.stop_training:
                break
        callbacks.on_train_end()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--lr", type=float)
        return parser

    @staticmethod
    def add_trainer_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--device", type=int)
        parser.add_argument("--amp", action="store_true")
        parser.add_argument("--epochs", type=int)
        return parser
