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

# %% 存储：将不同的标量数据导入到同一个excel中，并
def exp_to_diff_excel(file_list, scalarName):
    file_id = 0
    for file in file_list:
        excelName = "log/data" + str(file_id) + ".xlsx"  # 存储位置
        writer = pd.ExcelWriter(excelName)

        scalarValue = readEvent(file, scalarName)  # 读取数据
        data = pd.DataFrame(scalarValue)  # 转list为pd

        data = data.drop(labels='wall_time', axis=1)  # 删除wall_time列
        data.columns = ['step', 'score']  # 修改列名

        data.to_excel(writer, sheet_name=scalarName)  # 存入表，并重命名sheet
        file_id += 1

        writer.save()

    print("数据保存成功")

# %% 存储：将多个pd数据存入同一个excel中
def exp_to_one_excel(file_list, scalarName):
    excelName = "log/data.xlsx"  # 存储位置
    writer = pd.ExcelWriter(excelName)
    dcit01 = {}
    for file in file_list:
        scalarName = scalarName
        scalarValue = readEvent(file, scalarName)  # 读取数据
        for j in range(len(scalarValue)):
            # print(scalarValue[0][1])
            dcit01.update({'':j,'step':(j+1)*5000})

        print(dcit01)
        exit()

        data = pd.DataFrame(scalarValue)

    # print(len(Value_list))
    # exit()

      # 转list为pd

    # data = data.drop(labels='wall_time', axis=1)  # 删除wall_time列
    # data.columns = ['step', 'score']  # 修改列名

    data.to_excel(writer, sheet_name='Reward')  # 存入表，并重命名sheet

    writer.save()
    print("数据保存成功")

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
    # exp_to_diff_excel(file_list, scalarName)
    exp_to_one_excel(file_list, scalarName)


 # %% 如果要放置在不同的sheet下,则修改名字，excel中sheet名称的命名不能有：/\?*这些符号
# if scalarName == 'reward/Intrinsic Reward':
#     scalarName = 'Intrinsic Reward'
# if scalarName == 'reward/Reward':
#     scalarName = 'Reward'