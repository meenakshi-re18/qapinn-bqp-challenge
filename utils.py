"""
utils.py — reproducibility and logging helpers shared across the project.
"""
import os
import random
import logging
import numpy as np
import torch


def set_seed(seed: int):
    """Fix all relevant RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_path: str = None) -> logging.Logger:
    """Console + optional file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger  # avoid duplicate handlers on re-import

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class Timer:
    """Simple context-manager stopwatch."""
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        import time
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self._t0
