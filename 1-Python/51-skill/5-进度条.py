""""""
from tqdm import tqdm

# * 不推荐使用，在服务器上会卡死
for i in tqdm(range(5)):
    print(5)
