# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/30
"""

import numpy as np
import math


def calc_dist(data, x):
    return np.sqrt(np.sum((data[:-1] - x) ** 2))


def fetch_k_neighbors(datas, x, k):
    k_neighbors = []
    size = 0
    max_dist = -1
    max_index = -1

    for data in datas:
        dist = calc_dist(data, x)

        if size < k:
            k_neighbors.append((data, dist))
            if dist > max_dist:
                max_dist = dist
                max_index = size
            size += 1
        elif max_dist > dist:
            k_neighbors[max_index] = (data, dist)
            max_index = np.argmax(map(lambda t: t[1], k_neighbors))
            max_dist = k_neighbors[max_index][1]

    return k_neighbors


if __name__ == '__main__':
    datas = np.array([
        [5, 3, 1],
        [6, 3, 1],
        [8, 3, 1],
        [2, 3, 0],
        [3, 3, 0],
        [4, 3, 0]
    ])
    k_neighbors = fetch_k_neighbors(datas, x=np.array([7, 3]), k=2)
    print(k_neighbors)
