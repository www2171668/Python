import argparse
import time

import tensorflow as tf

def Arguments():
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument('--td3', default=True, type=bool)
    parser.add_argument('--seed', default='1', type=str)

    return parser.parse_args()

args = Arguments()

time.sleep(10)
seed = args.seed
print(seed)
