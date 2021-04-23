import numpy as np
import random
import torch

def set_seeds(args, rank=0):
    np.random.seed(args.seed + rank)  # numpy
    random.seed(args.seed + rank)  # random.random

    torch.manual_seed(args.seed + rank)  # pytorch
    torch.cuda.manual_seed(args.seed + rank)
