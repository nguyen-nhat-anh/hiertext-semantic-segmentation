import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from functools import wraps
from typing import Dict

from src.utils import logger
from src.model import HierTextModelModule


def _convert_to_int(x):
    return int(x) if x is not None else x


class ClusterEnvironment:
    local_rank = _convert_to_int(os.environ.get("LOCAL_RANK"))
    global_rank = _convert_to_int(os.environ.get("RANK"))
    local_world_size = _convert_to_int(
        os.environ.get("LOCAL_WORLD_SIZE")
    )  # equals to --nproc-per-node specified on torchrun.
    global_world_size = _convert_to_int(os.environ.get("WORLD_SIZE"))
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT")


class Strategy:
    # flow: connect -> setup_environment -> setup
    def connect(self, model: HierTextModelModule):
        self.model = model

    @property
    def root_device(self):
        pass

    def model_to_device(self):
        self.model.to(self.root_device)

    def batch_to_device(self, batch: Dict) -> Dict:
        return {
            k: (self.data_to_device(v) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

    def data_to_device(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self.root_device)

    def setup_environment(self):
        torch.cuda.set_device(self.root_device)

    def setup(self):
        self.model_to_device()

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        pass


class DDPStrategy(Strategy):
    def __init__(self, devices):
        self.devices = [torch.device("cuda", device) for device in devices]

    @property
    def root_device(self):
        return self.devices[ClusterEnvironment.local_rank]

    def setup_environment(self):
        dist.init_process_group(backend="nccl")
        super().setup_environment()

    def setup(self):
        super().setup()
        self.model = DistributedDataParallel(
            self.model, device_ids=[self.root_device.index]
        )

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.data_to_device(tensor)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor


class SingleDeviceStrategy(Strategy):
    def __init__(self, device):
        self.device = torch.device("cuda", device)

    @property
    def root_device(self):
        return self.device

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor


def is_rank_zero():
    return ClusterEnvironment.local_rank is None or ClusterEnvironment.local_rank == 0


def rank_zero_only(func):
    """Wrap a function to call internal function only in rank zero."""

    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if is_rank_zero():
            return func(*args, **kwargs)
        return None

    return wrapped_func


@rank_zero_only
def rank_zero_info(message):
    logger.info(message)
