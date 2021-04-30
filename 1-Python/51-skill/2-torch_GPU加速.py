import torch
import torch.nn

#%% GPU加速技巧
if torch.cuda.device_count() > 0:								# 查看是否有可用GPU、可用GPU数量
	eval_net = nn.DataParallel(eval_net).to(DEVICE)		# 使用nn.DataParallel函数来用多个GPU来加速训练
	target_net = nn.DataParallel(target_net).to(DEVICE)
	batch_size = args.batch_size * torch.cuda.device_count()	# 批次
else:
	batch_size = args.batch_size
