import csv
import os

def record_experience_to_csv(args, csv_name='experiments.csv'):
    if os.path.exists(csv_name):
        with open(csv_name, 'a') as f:
            print(list(args.keys()))
            w = csv.DictWriter(f, list(args.keys()))   #　list(args.keys())是csv文件的标头列表
            w.writerow(args)           #* 写入所有args信息
    else:
        with open(csv_name, 'w') as f:  # * 写入 w
            w = csv.DictWriter(f, list(args.keys()))
            w.writeheader()         #* 换行，指针回到头部
            w.writerow(args)


args = {'data1':2,'data2':3}
record_experience_to_csv(args)
exit()
