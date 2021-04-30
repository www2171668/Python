""""""
import numpy as np
import argparse     # 在juypter上使用会报错

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--sparse_num', action='store_true', help='GAT with sparse version or not.')    # default=False
parser.add_argument('--seed_num', default=0.72, help='Random seed.')        # 默认为float
parser.add_argument('--seed_num_float', type=float, default=0.72, help='Random seed.')        # 默认为float
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--hidden_dim', type=np.asarray, help='max size of the hidden layers', default=(400, 300))  # * array不要用type来定义
parser.set_defaults(sparse_num=False)

args = parser.parse_args()
print(args.sparse_num)
print(args.seed_num)
print(args.epochs)
print(args.hidden_dim)

#%% vars()：返回对象的{属性,属性值}
args = vars(args)
print(args['sparse_num'])
print(args['seed_num'])
print(args['seed_num_float'])
print(args['epochs'])
print(args['hidden_dim'])

print(type(args['sparse_num']))
print(type(args['seed_num']))
print(type(float(args['seed_num_float'])))
print(type(args['epochs']))
print(type(args['hidden_dim']))

# x = np.asarray((400,30))
# print(type(x))