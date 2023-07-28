from __future__ import annotations
import torch
import logging
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategy import Strategy


def get_logger() -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # formatter = logging.Formatter(
    #     fmt="%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d]%(message)s",
    #     datefmt="%m/%d/%Y %I:%M:%S %p",
    # )
    formatter = logging.Formatter(
        fmt="%(message)s",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger()


class MeterDict:
    def __init__(self):
        self.meter_dict = defaultdict(lambda: AverageMeter())

    @property
    def avg(self):
        return {k: v.avg for k, v in self.meter_dict.items()}

    @property
    def sum(self):
        return {k: v.sum for k, v in self.meter_dict.items()}

    @property
    def count(self):
        return {k: v.count for k, v in self.meter_dict.items()}

    def reset(self, key):
        self.meter_dict[key].reset()

    def update(self, key, val, n=1):
        self.meter_dict[key].update(val, n)

    def reduce(self, strategy: Strategy):
        for k, v in self.meter_dict.items():
            v.reduce(strategy)

    def __repr__(self):
        str_format = "{:<20} {}"
        str_repr = [str_format.format(f"({k})", v) for k, v in self.meter_dict.items()]
        return "\n".join(str_repr)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reduce(self, strategy: Strategy):
        sum_tensor = torch.tensor(self.sum, dtype=torch.float32)
        count_tensor = torch.tensor(self.count, dtype=torch.int64)
        sum_tensor = strategy.reduce(sum_tensor)
        count_tensor = strategy.reduce(count_tensor)
        self.sum = sum_tensor.item()
        self.count = count_tensor.item()
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"Average: {self.avg:.4f} - Sum: {self.sum:.4f} - Count: {self.count}"
