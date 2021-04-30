""""""
import pandas as pd
import os
from tensorboard.backend.event_processing import event_accumulator

# %% 读取：读tensorboard文件的标量值      event_path:event文件路径        scalarName：要操作的标量名称
def readEvent(tfevents_path, scalarName):
    event = event_accumulator.EventAccumulator(tfevents_path)
    event.Reload()
    print(event.Tags())
    print(event.scalars.Keys())
    value = event.scalars.Items(scalarName)
    return value

# %% 存储：将不同的标量数据导入到同一个excel中，并放置在不同的sheet下
def exportToexcel(tfevents_path, scalarNameList, excelName):
    writer = pd.ExcelWriter(excelName)

    for i in range(len(scalarNameList)):
        scalarName = scalarNameList[i]
        scalarValue = readEvent(tfevents_path, scalarName)  # 读取数据
        data = pd.DataFrame(scalarValue)  # 修改数据类型
        data = data.drop('wall_time', 1)  # 删除wall_time列

        # %% 修改名字，excel中sheet名称的命名不能有：/\?*这些符号
        # if scalarName == 'reward/Intrinsic Reward':
        #     scalarName = 'Intrinsic Reward'
        # if scalarName == 'reward/Reward':
        #     scalarName = 'Reward'

        data.to_excel(writer, sheet_name=scalarName)

    writer.save()
    print("数据保存成功")

# %% 遍历文件 一次只放一个tfevents文件进来
def walkFile(file):
    for root, dirs, files in os.walk(file):
        for f in files:  # * 遍历文件
            return os.path.join(root, f)

if __name__ == "__main__":
    tfevents_path = walkFile("log/")
    scalarNameList = ['Reward']  # scalar名
    excelName = "log/data.xlsx"  # 存储位置
    exportToexcel(tfevents_path, scalarNameList, excelName)
