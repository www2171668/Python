import argparse
import time
import ftest

def Arguments():
    parser = argparse.ArgumentParser()

    # Across All
    parser.add_argument('--td3', default=True, type=bool)
    parser.add_argument('--seed', default='1', type=str)         # action=‘store_true’，只要运行时该变量有传参,如--train,就将该变量设为True


    return parser.parse_args()

args = Arguments()
# if args.td3:
    # path = args.seed + ".txt"
    # f = open(path, 'w')  # 打开文件，若文件不存在系统自动创建。
    # f.write(args.seed + '\n')
    # ftest.CCC(f=f)
    # time.sleep(10)
# f.close()

seed = args.seed
print(seed)

