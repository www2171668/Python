import csv
import os

def csv_save(args, csv_name='data/experiments.csv'):
    if not os.path.exists(csv_name):  # 创建scv
        with open(csv_name, 'w') as f:  # * 写入 w
            w = csv.DictWriter(f, list(args.keys()))
            w.writeheader()  # * 换行，指针回到头部
            w.writerow(args)
    else:
        with open(csv_name, 'a') as f:
            print(list(args.keys()))
            w = csv.DictWriter(f, list(args.keys()))  # list(args.keys())是csv文件的标头列表
            w.writerow(args)  # * 写入所有args信息

dict01 = {'data1': 2, 'data2': 3}
csv_save(dict01)
