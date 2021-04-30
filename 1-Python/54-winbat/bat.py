# import torch
# import time
#
# ts = time.time()
# device = "cuda:0"
#
# a = torch.rand((5, 10000000), dtype=torch.float32, device=device)
# b = torch.rand((5, 10000000), dtype=torch.float32, device=device)
# ts = time.time()
# for i in range(10000):
#     c = (a - b) ** 2
#     # print(c)
# te = time.time() - ts
# print(f"{te}s")

import argparse
parser = argparse.ArgumentParser()

# Across All
parser.add_argument('--train', action='store_true')  # action=‘store_true’，只要运行时该变量有传参,就将该变量设为True
parser.add_argument('--eval', type=str, default='hhjj')

args = parser.parse_args()
print(args.train)
print(args.eval)