import numpy as np
# from torch.utils.tensorboard import SummaryWriter  # 也可以使用 tensorboardX
from tensorboardX import SummaryWriter  # 也可以使用 pytorch 集成的 tensorboard

writer = SummaryWriter('log')
for epoch in range(100):
    # add_scalar/squared:tag
    # np.square(epoch)：ｙ
    # epoch：x
    writer.add_scalar('add_scalar/squared', np.square(epoch), epoch)
    # writer.add_scalars("add_scalars/trigonometric", {'xsinx': epoch * np.sin(epoch/5), 'xcosx': epoch* np.cos(epoch/5), 'xtanx': np.tan(epoch/5)}, epoch)

writer.close()
