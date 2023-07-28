from __future__ import annotations
import numpy as np
from copy import deepcopy
from typing import List, TYPE_CHECKING
from src.strategy import rank_zero_info

if TYPE_CHECKING:
    from src.trainer import HierTextTrainer
    from src.model import HierTextModelModule


class Callback:
    """Base class used to build new callbacks"""

    def __init__(self):
        self.trainer = None
        self.model_module = None

    def setup(self, trainer: HierTextTrainer, model_module: HierTextModelModule):
        """Connect callback to trainer and model

        Args:
            trainer (HierTextTrainer): trainer to connect
            model_module (HierTextModelModule): model to connect
        """
        self.trainer = trainer
        self.model_module = model_module

    def on_train_begin(self, **kwargs):
        """Called when training begins"""
        pass

    def on_train_end(self, **kwargs):
        """Called when training ends"""
        pass

    def on_epoch_begin(self, **kwargs):
        """Called at the start of each epoch"""
        pass

    def on_epoch_end(self, **kwargs):
        """Called at the end of each epoch"""
        pass


class Callbacks:
    """A class that represents a list of callbacks"""

    def __init__(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def setup(self, trainer: HierTextTrainer, model_module: HierTextModelModule):
        """Setup every callback

        Args:
            trainer (HierTextTrainer): trainer to connect
            model_module (HierTextModelModule): model to connect
        """
        for callback in self.callbacks:
            callback.setup(trainer, model_module)

    def on_train_begin(self, **kwargs):
        """Called when training begins"""
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs):
        """Called when training ends"""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_epoch_begin(self, **kwargs):
        """Called at the start of each epoch"""
        for callback in self.callbacks:
            callback.on_epoch_begin(**kwargs)

    def on_epoch_end(self, **kwargs):
        """Called at the end of each epoch"""
        for callback in self.callbacks:
            callback.on_epoch_end(**kwargs)


class EarlyStoppingCallback(Callback):
    """Monitor a metric and stop training when it stops improving."""

    def __init__(self, patience: int = 1, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.is_improvement = np.less if self.mode == "min" else np.greater

        self.best = None
        self.best_epoch = None
        self.best_state_dict = None
        self.wait = None
        self.stopped_epoch = None

    def on_train_begin(self, **kwargs):
        self.best = np.inf if self.mode == "min" else -np.inf
        self.best_epoch = 0
        self.best_state_dict = None
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs, **kwargs):
        current = logs.get(self.monitor)
        self.wait += 1
        if self.is_improvement(current, self.best):
            self.best = current
            self.best_epoch = epoch
            self.best_state_dict = deepcopy(self.model_module.state_dict())
            self.wait = 0

        # Check early stopping condition (only check after the first epoch)
        if self.wait >= self.patience and epoch > 1:
            self.stopped_epoch = epoch
            self.trainer.stop_training = True

    def on_train_end(self, **kwargs):
        if self.stopped_epoch is not None:
            rank_zero_info("Epoch {}: early stopping".format(self.stopped_epoch))
        else:
            rank_zero_info("Training finished without early stopping")
        rank_zero_info(
            "Restore best weight with {} = {:.2f} at epoch {}".format(
                self.monitor, self.best, self.best_epoch
            )
        )
        self.model_module.load_state_dict(self.best_state_dict)
