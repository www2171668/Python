# -- encoding:utf-8 --
"""
Create by ibf on 2018/5/12
"""
# 引入包
import tensorflow as tf
import matplotlib as mpl
from tensorflow.examples.tutorials.mnist import input_data

# 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

'''
    加载数据 - 数字图形预测
    
    mnist:由6万张训练图片和1万张28*28黑白色图片（手写0到9的数字）构成,黑色是一个0-1的浮点数，黑色越深表示数值越靠近1
    TensorFlow将这个数据集和相关操作封装到了库中

    -》input_data.read_data_sets("path",one_hot)
        one_hot:哑编码处理
        如果path路径文件夹里有手写数字数据集的话，就不从官网中下载了，否则就要下载，但是在程序中直接下载，速度会非常非常慢，所以建议在官网上下载好放在文件夹中
        该数据集是4个压缩文件，不能解压，解压是会报错
        
    ①、predict（预测的结果）：单个样本被预测出来是哪个数字的概率
    如[ 1.07476616 -4.54194021 2.98073649 -7.42985344 3.29253793 1.96750617 8.59438515 -6.65950203 1.68721473 -0.9658531 ]，
    分别表示0，1，2，3，4，5，6，7，8，9数字的概率，然后会取一个最大值来作为本次预测的结果，对于这个数组来说，结果是6（8.59438515）
    ②、y（真实结果）：来自MNIST的训练集，每一个图片所对应的真实值，如6表示为：[0 0 0 0 0 1 0 0 0]
'''
mnist = input_data.read_data_sets('data/', one_hot=True)

'''
    mnist属性：
    -》mnist.train.num_examples  输出训练集样本数
    -》~~~~
'''

print(mnist.train.num_examples)   # 训练集样本数目 55000
print(mnist.train.images.shape)   # (55000, 784) 每个样本有784个维度
print(mnist.train.labels.shape)   # (55000, 10) 目标属性有10个
print(mnist.train.labels[0])   # [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.] 加载时对目标属性数据做了哑编码操作；第一个样本属于7

print(mnist.test.num_examples)   # 测试集样本数目 10000

# Extracting data/train-images-idx3-ubyte.gz
# Extracting data/train-labels-idx1-ubyte.gz
# Extracting data/t10k-images-idx3-ubyte.gz
# Extracting data/t10k-labels-idx1-ubyte.gz


