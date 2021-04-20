""""""
#%%
import pickle

'''
    打开方法一：
    -》filename = open ('文件名','打开的模式')
        r:以只读的方式打开；如果文件不存在会提示错误
        w:以只写的方式打开；如果文件不存在则创建，存在则覆盖内容，添加新内容
        a:以只写的方式打开；如果文件不存在则创建新的文件，存在则保存内容，并追加内容
        
        r+:读写；如果文件不存在会提示错误
        w+:读写；如果文件不存在则创建，存在则覆盖源文件内容
        a+:追加读写；将文件指针调到文件的末尾
        
        rb:二进制转正常类型，通常用于pickle.load(open(文件地址，rb))文件打开操作   ★
            pickle.load(open(文件地址，'rb'))
        wb:转二进制后写，通常用于pickle.dump(存入内容,保存地址)文件保存操作 - pickle.dump()只能写二进制   ★
            with open(保存地址, 'wb') as 文件变量名:
                pickle.dump(保存内容, 文件变量名)
        
        -》*.close() 关闭文件

    读取方法一：
    -》file.read(*):读取文件内容
    *：读取指定长度的内容，如果没有则读取所有的数据
    注意读的时候，是从当前位置开始读的，也就是说，如果指针在末尾，就需要用seek（0）调整指针位置
'''
# 打开文件
files = open('python.txt','r',encoding= 'utf-8')   # -》后面加上encoding= 'utf-8'，以规避中文报错显现
# 读取文件内容
content = files.read()  #读取数据保存在content变量当中
# content = files.read(10)  #读取指定长度内容
# 输出读取的内容
print(content)
# 关闭文件
files.close()

#%%
'''
    打开方法二： ★
    -》with open('文件名','打开的模式') as filename:     使用关键字with打开文件，python会自动关闭文件
        文件操作
'''
with open('python.txt','r',encoding= 'utf-8' ) as files:      #这是一个代码块，注意有：
    content = files.read()  # 对该文件的操作都在代码块中进行
    print(content)
print(content)              # 读取出的内容可以在with之外使用


#%%
'''
    读取方法二：
    -》file.readlines()：一次性读取文件内容，返回一个列表，每一行的数据为一个元素（/n也会被包含在行末）
'''
# 打开文件
files = open('python.txt','r',encoding= 'utf-8' )
#读取文件内容
content = files.readlines()

#输出读取的内容
print(content)      #输出['元旦一起喝咖啡，看电影可以吗？想和你一起去开海\n', '不去\n']
#关闭文件
files.close()

#%%
'''
    逐行读取
    读取方法三：直接用for循环遍历file
'''
# 1、用open结合for循环逐行读取
files = open('python.txt','r',encoding= 'utf-8')
i = 1
for line in files:   # 遍历每行内容
    print('这是第%d行内容:%s'%(i,line))
    i+=1
files.close()

#2、用with结合for循环逐行读取
with open('python.txt','r',encoding= 'utf-8') as files:
    i = 1                                     #该部分也可以不
    for line in files:
        print('这是第%d行内容:%s'%(i,line))
        i+=1

#3、用readline()赋值来规避open的代码块（close）
files = open('python.txt','r',encoding= 'utf-8')
contents = files.readlines()   #逐行读取内容
files.close()   #关闭文件

i = 1
for line in contents:
    print('这是第%d行内容:%s'%(i,line))   # 可以规避\n
    i+=1

#4、用readline()赋值来规避with的代码块
with open('python.txt','r',encoding= 'utf-8') as files:
    contents = files.readlines()
    # contents = files.read()       #不能用read
i = 1
for line in contents:
    print('这是第%d行内容:%s'%(i,line))
    i+=1

#%%
'''
    写入文件
    -》.write()   只能写str。不会换行
    -》.writeline()   可以写入list、tuple类型数据。不会换行
'''

# 以写的方式打开一个文件
# files = open('python.txt','w',encoding='utf-8')
files = open('python.txt','a',encoding='utf-8')
content = 'hello,爱你哟'
files.write(content) #写入数据
files.close()

with open('python.txt','a',encoding='utf-8') as files:
    content = '想和你一起去看海'
    files.write(content)

with open('python.txt','a',encoding='utf-8') as files:
    content = ['aaa','bbb','ccc']
    files.writelines(content)

#%%
'''
    -》.tell()  返回文件中指针位置
'''
files = open('python.txt','r',encoding='utf-8')
str = files.read(5)
print('当前读取的数据是：'+str)

#查看文件的指针
position = files.tell()
print('当前的位置是：',position)

str = files.read()
print('当前读取的数据是：'+str)

#查看文件的指针
position = files.tell()
print('当前的位置是：',position)

files.close()

#%%
'''
    -》.seek() 设置指针位置
'''
files = open('python.txt','r',encoding='utf-8')
str = files.read(5)
print('当前读取的数据是：'+str)

# 查看文件的指针
position = files.tell()
print('当前的位置是：',position)

# 重新设置文件的指针
files.seek(0,0)       #0是回到文件头部
files.seek(2,0)       #2是偏移量
files.seek(0,1)       #1是停留在当前位置

str = files.read(2)
print('当前读取的数据是：'+str)

#查看文件的指针
position = files.tell()
print('当前的位置是：',position)

files.close()

files = open('python.txt','a+')
files.write('水电费')
files.seek(0)
content = files.read()
print(content)
files.close()

