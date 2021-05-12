""""""
import numpy as np
import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator

# %% 读取：读tensorboard文件的标量值      event_path:event文件路径        scalarName：要操作的标量名称
def readEvent(file_list, scalarName):
    event = event_accumulator.EventAccumulator(file_list)
    event.Reload()
    print(event.Tags())
    print(event.scalars.Keys())
    value = event.scalars.Items(scalarName)
    return value

# %% 存储：将不同的标量数据导入到同一个csv中
def ten_to_diff_excel(file_list, scalarName):
    file_id = 0
    for file in file_list:
        excelName = "log/data" + str(file_id) + ".csv"  # 存储位置

        scalarValue = readEvent(file, scalarName)  # 读取数据
        data = pd.DataFrame(scalarValue, columns=['wall_time', 'step', 'score'])  # 转list为pd
        data = data.drop(labels='wall_time', axis=1)  # 删除wall_time列

        data.to_csv(excelName)  # 存入表，并重命名sheet
        file_id += 1

# %% 存储：将多个pd数据存入同一个csv中
def ten_to_one_excel(file_list, scalarName):
    file_id = 0
    excelName = "log/data.csv"  # 存储位置
    datas = np.array([])
    for file in file_list:
        scalarValue = readEvent(file, scalarName)  # 读取数据
        data = pd.DataFrame(scalarValue)  # 转list为pd
        data = data.drop(labels='wall_time', axis=1)  # 删除wall_time列

        if file_id == 0:
            datas = data
        else:
            datas = np.concatenate((datas, data), axis=0)  # 叠加数据

        file_id += 1

    datas[:, 0] = datas[:, 0] / 1000000 # million
    datas = pd.DataFrame(datas, columns=['step', 'score'])
    datas.to_csv(excelName)  # 存入表，并重命名sheet

# %% 遍历文件 一次只放一个tfevents文件进来
def walkFile(file):
    file_list = []
    for root, dirs, files in os.walk(file):
        for f in files:  # * 遍历文件
            if 'events' in f:  # 只遍历tensorboard文件
                file_list.append(os.path.join(root, f))
    return file_list

if __name__ == "__main__":
    file_list = walkFile("log/")
    scalarName = 'Reward'  # scalar名
    # ten_to_diff_excel(file_list, scalarName)
    ten_to_one_excel(file_list, scalarName)
    print("数据保存成功")
