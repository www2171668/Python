# -- encoding:utf-8 --
"""
Create by ibf on 2018/8/28
"""
import numpy as np
import pandas as pd
# import visuals as vs  # Supplementary code
from sklearn.model_selection import ShuffleSplit

# Pretty display for notebooks
# 让结果在notebook中显示

# Load the Boston housing dataset
# 载入波士顿房屋的数据集
data = pd.read_csv('./data/boston_housing.data')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

# Success
# 完成
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
