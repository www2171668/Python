import time

path = "data/name.txt"
f = open(path,'w')     #打开文件，若文件不存在系统自动创建(目录必须存在)
mean = 12345.564
std =454.5785
for i in range(5):

    f.writelines('mean:{mean:.2f}, std:{std:.2f}'.format(mean=mean,std=std) + '\n')  # write 写入
    f.write('hello22222 word 你好 \n')  # write 写入
    time.sleep(10)


f.close()              #关闭文件