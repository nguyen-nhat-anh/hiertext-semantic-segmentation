import logging
from collections import defaultdict


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

    def __repr__(self):
        return f"Average: {self.avg:.4f} - Sum: {self.sum:.4f} - Count: {self.count}"
