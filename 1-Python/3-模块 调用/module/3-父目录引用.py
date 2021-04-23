""""""
# %% 引入非同父目录中的模块
import sys  # * 使用搜索路径变量，变量里包含当前目录
import os
# * 方法一：设为根源
# sys.path.append('..\\')  # * 方法二：添加父文件夹(有时候不一定有用)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # * 方法三：添加父文件夹

import msg.send as MS
MS.sendMsg()

from msg import send
send.sendMsg()
