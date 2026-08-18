"""
Reproducible Random Seed Utilities
Fixes random seeds across Python random, NumPy, PyTorch CPU & GPU to ensure repeatability.
"""

import random
import os
import numpy as np
import torch

def set_seed(seed=42):
    """Sets random seed across all libraries."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"[Seed Initialization] Random seed fixed to: {seed}")

if __name__ == "__main__":
    set_seed(42)
