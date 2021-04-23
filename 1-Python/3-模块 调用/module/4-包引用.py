""""""

# %% 包       创建包时需要创建一个__init__.py 文件，文件中编写 __all__ = ['包中包含的模块名']
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # * 方法三：添加父文件夹

from msg import *
send.sendMsg()
