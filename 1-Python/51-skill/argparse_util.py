""""""

import argparse     # 在juypter上使用会报错

parser = argparse.ArgumentParser(description='test')

parser.add_argument('--sparse_num', action='store_true', help='GAT with sparse version or not.')    # default=False
parser.add_argument('--seed_num', default=0.72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.set_defaults(sparse_num=False)

args = parser.parse_args()
print(args.sparse_num)
print(args.seed_num)
print(args.epochs)

#%% vars()：返回对象的{属性,属性值}
args = vars(args)
print(args['sparse_num'])
print(args['seed_num'])
print(args['epochs'])
