import numpy as np

from arguments import get_args
args = get_args()

from utils import set_seeds
set_seeds(args)

for i in range(3):
    b = np.random.random(2)
    print(b)
