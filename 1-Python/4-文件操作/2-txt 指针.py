""""""

# %% .tell()  返回文件中指针位置
files = open('python.txt', 'r', encoding='utf-8')
str = files.read(5)
print('当前读取的数据是：' + str)

# 查看文件的指针
position = files.tell()
print('当前的位置是：', position)

str = files.read()
print('当前读取的数据是：' + str)

# 查看文件的指针
position = files.tell()
print('当前的位置是：', position)

files.close()

# %% .seek() 设置指针位置
files = open('python.txt', 'r', encoding='utf-8')
str = files.read(5)
print('当前读取的数据是：' + str)

# 查看文件的指针
position = files.tell()
print('当前的位置是：', position)

# 重新设置文件的指针
files.seek(0, 0)  # 0是回到文件头部
files.seek(2, 0)  # 2是偏移量
files.seek(0, 1)  # 1是停留在当前位置

str = files.read(2)
print('当前读取的数据是：' + str)

# 查看文件的指针
position = files.tell()
print('当前的位置是：', position)

files.close()

files = open('python.txt', 'a+')
files.write('水电费')
files.seek(0)
content = files.read()
print(content)
files.close()
