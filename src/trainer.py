import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from tqdm.auto import tqdm
from typing import List

from src.callback import Callbacks, Callback
from src.dataset import HierTextDataModule
from src.utils import MeterDict
from src.model import HierTextModelModule
from src.strategy import (
    Strategy,
    DDPStrategy,
    SingleDeviceStrategy,
    rank_zero_info,
    is_rank_zero,
    rank_zero_only,
)


class HierTextTrainer:
    def __init__(
        self,
        strategy: str,
        devices: List[int],
        callbacks: List[Callback],
        epochs: int,
        amp: bool,
        lr: float,
    ):
        self.strategy: Strategy
        if strategy == "single_device":
            self.strategy = SingleDeviceStrategy(devices[0])
        elif strategy == "ddp":
            self.strategy = DDPStrategy(devices)
        else:
            self.strategy = SingleDeviceStrategy(devices[0])
        self.callbacks = Callbacks(callbacks)
        self.epochs = epochs
        self.amp = amp
        self.lr = lr
        self.use_distributed_sampler = strategy == "ddp"
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.stop_training = None  # for early stopping

    def configure(self, model: HierTextModelModule, steps_per_epoch: int):
        self.optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=self.lr
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

    def training_epoch(
        self, model: HierTextModelModule, loader: DataLoader
    ) -> MeterDict:
        model.train()
        summaries = MeterDict()
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not is_rank_zero())
        for step, batch in pbar:
            batch = self.strategy.batch_to_device(batch)
            with torch.cuda.amp.autocast(enabled=self.amp):
                result = model.training_step(batch, step)
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
        summaries.reduce(self.strategy)
        return summaries

    @torch.inference_mode()
    def validation_epoch(
        self, model: HierTextModelModule, loader: DataLoader
    ) -> MeterDict:
        model.eval()
        summaries = MeterDict()
        pbar = tqdm(enumerate(loader), total=len(loader), disable=not is_rank_zero())
        for step, batch in pbar:
            batch = self.strategy.batch_to_device(batch)
            result = model.validation_step(batch, step)
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
            for cat, iou in zip(model.categories, iou_per_class):
                summaries.update(f"{cat}_iou", iou.item(), batch_size)
            summaries.update("macro_avg_iou", iou_per_class.mean().item(), batch_size)
            pbar.set_postfix(**summaries.avg)
        summaries.reduce(self.strategy)
        return summaries

    def fit(
        self,
        model: HierTextModelModule,
        data_module: HierTextDataModule,
        ckpt_path: str,
    ):
        # attach model to strategy
        self.strategy.connect(model)
        # setup cuda device and initialize process group
        self.strategy.setup_environment()
        # configure model and move it to the device
        self.strategy.setup()
        # get dataloaders
        train_loader = data_module.train_dataloader(self.use_distributed_sampler)
        val_loader = data_module.val_dataloader(self.use_distributed_sampler)
        self.configure(model, steps_per_epoch=len(train_loader))
        self.callbacks.setup(self, model)
        self.callbacks.on_train_begin()
        for epoch in range(1, self.epochs + 1):
            self.callbacks.on_epoch_begin(epoch=epoch)
            rank_zero_info("# Epoch {}/{}: #".format(epoch, self.epochs))
            train_summaries = self.training_epoch(model, train_loader)
            val_summaries = self.validation_epoch(model, val_loader)
            message = (
                "Train loss {:.4f}\nVal loss {:.4f}\nMacro average IOU {:.4f}\n".format(
                    train_summaries.avg["loss"],
                    val_summaries.avg["loss"],
                    val_summaries.avg["macro_avg_iou"],
                )
            )
            message += "\n".join(
                [
                    " {} IOU {:.4f}".format(cat, val_summaries.avg[f"{cat}_iou"])
                    for cat in model.categories
                ]
            )
            rank_zero_info(message)
            self.callbacks.on_epoch_end(
                epoch=epoch, logs={"val_loss": val_summaries.avg["loss"]}
            )
            if self.stop_training:
                break
        self.callbacks.on_train_end()
        self.save_weights(model, ckpt_path)

    @rank_zero_only
    def save_weights(self, model, ckpt_path):
        if isinstance(model, DistributedDataParallel):
            torch.save(model.module.state_dict(), ckpt_path)
        else:
            torch.save(model.state_dict(), ckpt_path)
