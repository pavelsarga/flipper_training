import random
import numpy as np
import torch


def set_device(device: str) -> torch.device:
    """
    Set the device for the torch module
    """
    if "cuda" in device and torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device(device)
    elif device == "mps" and torch.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int) -> torch.Generator:
    """
    Seed all random number generators (Python random, NumPy, PyTorch CPU/CUDA/MPS).
    """
    random.seed(seed)
    np.random.seed(seed)
    rng = torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    return rng
