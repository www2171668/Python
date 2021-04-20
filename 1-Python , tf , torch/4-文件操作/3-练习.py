#-*- conding:utf-8 -*-
''''''
'''
    在文本编辑器中新建一个文件，写几句话，其中每一行都以“In Python you can”打头。
    将这个文件命名为learning_python.txt，并将其存储到该目录中。
    编写一个程序，读取这个文件，将内容打印三次：
    第一次打印时读取整个文件；第二次打印时按行遍历；第三次打印时将各行存储在一个列表中。
'''
# for i in range(3):
#     if i == 0:
#         with open('learning_python.txt','r',encoding='utf-8') as files:
#             content = files.read()
#             print('one')
#             print(content)
#     if i == 1:
#         with open('learning_python.txt','r',encoding='utf-8') as files:
#             print('two')
#             for line in files:
#                 print(line)
#     if i == 2:
#         with open('learning_python.txt','r',encoding='utf-8') as files:
#             content = files.readlines()
#             print('three')
#             print(content)

'''
    读取learning_python.txt 中的每一行，将其中的Python 都替换为另C。
    将修改后的各行都打印到屏幕上。块外打印它们。
'''
# #打开文件，读取内容
# file1 = open('learning_python.txt','r',encoding='utf-8')
# content1 = file1.read()
# file1.close()
#
# #开打文件，写入内容并且读取新的内容
# file2 = open('learning_python.txt','w+',encoding='utf-8')   # w+ 写+读
# file2.write(content1.replace('Python','C')) # -》replace 写入的时候替换内容
# file2.seek(0)  # 重置指针导位置
# content2 = file2.read() # 读取所有的内容
# print(content2)
# file2.close()


'''
    编写一个程序，提示用户输入其名字；用户作出响应后，将其名字写入到文件guest.txt 中
'''
while True:
    name = input('请输入您的姓名:')
    if name == 'n':
        break;
    with open('guest.txt','a+',encoding='utf-8') as files:  #用追加的读写
        files.write(name)
        files.write('\n')
        content = files.read()   #注意对文件操作方法必须在with的代码块中（或者在opne和close之间）
print(content)
