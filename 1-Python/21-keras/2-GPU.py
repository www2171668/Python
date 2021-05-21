import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

# pip install tensorflow-determinism
from tfdeterminism import patch
patch() # 控制kears在gpu上的结果可复现性，放在代码头部即可

#进行配置，每个GPU使用60%上限现存
os.environ["CUDA_VISIBLE_DEVICES"]="1" # 使用编号为1，2号的GPU
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)

# 设置session
KTF.set_session(session )

