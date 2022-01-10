import logging
import os
import random
import warnings
from functools import wraps
from typing import Optional

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: Optional[int] = None, workers: bool = False, deterministic: bool = False) -> int:
    """
    ``
    This part of the code comes from PyTorch Lightning.
    https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/seed.py
    ``

    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    In addition, sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)
        rank_zero_warn(f"No correct seed found, seed set to {seed}")

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return seed


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return wrapped_fn


@rank_zero_only
def rank_zero_warn(*args, stacklevel: int = 4, **kwargs):
    warnings.warn(*args, stacklevel=stacklevel, **kwargs)


def reset_seed() -> None:
    """
    Reset the seed to the value that :func:`seed_everything` previously set.
    If :func:`seed_everything` is unused, this function will do nothing.
    """
    seed = os.environ.get("PL_GLOBAL_SEED", None)
    workers = os.environ.get("PL_SEED_WORKERS", False)
    if seed is not None:
        seed_everything(int(seed), workers=bool(workers))


def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s


def check_tensor(data, is_label):
    if not is_label:
        return data if data.ndim <= 4 else data.squeeze()
    return data.long() if data.ndim <= 3 else data.squeeze().long()
