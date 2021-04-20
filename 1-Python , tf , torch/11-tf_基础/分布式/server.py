# -- encoding:utf-8 --
"""
Create by ibf on 2018/5/6
"""

"""
    运行命令：
    python 7.1-server.py --job_name=ps --task_index=0   输出：Started server with target: grpc://localhost:33331
    python 7.1-server.py --job_name=ps --task_index=1
    python 7.1-server.py --job_name=work --task_index=0
    python 7.1-server.py --job_name=work --task_index=1
    python 7.1-server.py --job_name=work --task_index=2
"""

import tensorflow as tf

'''
    -》tf.train.ClusterSpec(dic)：构建集群配置信息
      cluster = tf.train.ClusterSpec({"worker": {1: "worker1.example.com:2222"},
                                      "ps": ["ps0.example.com:2222","ps1.example.com:2222"]})
    tensorflow底层代码中，默认使用ps和work分别表示两类不同的工作节点:
        -》ps：变量/张量的初始化、存储相关节点
        -》work: 变量/张量的计算/运算的相关节点
'''

'''
    -》tf.app:Tensorflow的应用（Application）  .flags.FLAGS：配置项
    -》DEFINE_string：定义string类型
        属性名称，属性名要与Server()的job_name保持一致
        default_value：默认值
        docstring：描述信息
        
    -》server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)    
        cluster：集群信息
        job_name：任务名称
        task_index：训练号
        
    -》server.join()：阻塞等待
    
    -》tf.app.run()：运行分布式程序
'''

# 1. 配置服务器相关信息
# ：前面的是IP地址，现在用127.0.0.1模拟单机情况下的分布式地址；多台机器时，更换为机器的IP地址
# ：后面的是端口号，端口号不能相同
ps_hosts = ['127.0.0.1:33331', '127.0.0.1:33332']
work_hosts = ['127.0.0.1:33333', '127.0.0.1:33334', '127.0.0.1:33335']
cluster = tf.train.ClusterSpec({'ps': ps_hosts, 'work': work_hosts})

# 2. 定义运行参数(在运行该python文件的时候指定参数)
tf.app.flags.DEFINE_string('job_name', default_value='work', docstring="One of 'ps' or 'work'")
tf.app.flags.DEFINE_integer('task_index', default_value=0, docstring="Index of task within the job")
FLAGS = tf.app.flags.FLAGS

# 3. 启动服务
def main(_):   # 定义默认main()方法
    print(FLAGS.job_name)
    server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    server.join()

# 4、运行程序
if __name__ == '__main__':
    tf.app.run()   # 底层默认会调用main方法
