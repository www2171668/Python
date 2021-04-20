""""""

# %% 引入同级目录中的自定义模块       函数名
from Custom import add
print(add(2, 3))

# %% 引入同父目录中的文件夹下的模块     文件夹名.函数名
import study.Custom2 as C2
print(C2.add(1,2))

from study import Custom2
print(Custom2.add(2, 3))

