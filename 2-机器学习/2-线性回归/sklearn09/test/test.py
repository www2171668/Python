# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/26
"""

import numpy as np
import pandas as pd

a = np.array(['aaa', 'aaa', 'bbbb', 'bbbb', 'c', 'd', 'c', 'f'])
aa = np.unique(a)
size_aa = len(aa)
for i in range(size_aa):
    a[a == aa[i]] = i
a = a.astype(np.int)
print(a)
print(aa)


class A:
    def Categorical(self, arr):
        a = arr
        aa = np.unique(a)
        size_aa = len(aa)
        for i in range(size_aa):
            a[a == aa[i]] = i
        a = a.astype(np.int)

        self.codes = a
        self.categories = aa
        # return [a, aa]
        return self


# a = np.array(['aaa', 'aaa', 'bbbb', 'bbbb', 'c', 'd', 'c', 'f'])
# cate = pd.Categorical(a)
# print(type(cate))
# print(cate.codes)
# print(cate.categories)

a = np.array(['aaa', 'aaa', 'bbbb', 'bbbb', 'c', 'd', 'c', 'f'])
A = A()
result = A.Categorical(a)
print(result.codes)
print(result.categories)
