""""""
import os

# %% 遍历文件夹
def walkFile(file):
    for root, dirs, files in os.walk(file):
        # root 表示当前正在访问的文件夹路径
        # dirs 表示该文件夹下的子目录名list
        # files 表示该文件夹下的文件list

        for f in files:  # * 遍历文件
            print(f)
            # print(os.path.join(root, f))

        for d in dirs:  # * 遍历所有的文件夹
            print(d)
            # print(os.path.join(root, d))

def main():
    walkFile("data/")

if __name__ == '__main__':
    main()
