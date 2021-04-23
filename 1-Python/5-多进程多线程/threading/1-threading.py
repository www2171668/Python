#-*- conding:utf-8 -*-
''''''

import time
import threading        #导入模块时，保证脚本名和模块名不同，否则会加载错误

'''
    单线程：依次进行
''' 
# def music(name,loop):
#     for i in range(loop):
#         print('listen music %s %s'%(name,time.ctime()))
#         time.sleep(1)
#
# def movie(name,loop):
#     for i in range(loop):
#         print('look movie %s %s'%(name,time.ctime()))
#         time.sleep(1)
#
# if __name__ == '__main__':
#     music('爱的故事上集',3)
#     movie('晓生克的救赎',4)
#     print('单线程')


'''
    多线程：先运行主线程，再运行子线程。主线程运行时遵循主函数运行顺序
    -》threading.Thread(target,args,name) :target函数属于子线程
    -》.setName() ：线程命名
    
    -》.setDaemon(bool) ：为True时，守护主线程，主线程结束后子线程不再运行
    -》.ident ：线程id号
    -》.join() ：对主线程进行阻塞，等所有的子线程运行结束，再运行主线程
'''
def music(name,loop):
    for i in range(loop):
        print('listen music %s %s %s'%(name,time.ctime(),threading.Thread.getName(t1)))  # 获得线程名称
        time.sleep(1)

def movie(name,loop):
    for i in range(loop):
        print('look movie %s %s %s'%(name,time.ctime(),threading.Thread.getName(t2)))
        time.sleep(1)

#1、创建多线程
t1 = threading.Thread(target= music,args=('爱的故事上集',4))
t1.setName('musicThread')       #设置线程名称方法一
t2 = threading.Thread(target= movie,args=('肖生克的救赎',4),name = 'movieThread')     #设置线程名称方法二

if __name__ == '__main__':
    # 2.守护主线程
    t1.setDaemon(False)
    t2.setDaemon(False)

    # 3.启动线程
    t1.start()
    print('1')
    t2.start()
    print('2')  # 主线程第一次会运行到此，然后开始子线程

    # 4. 对主线程进行阻塞，同时阻塞主函数运行
    t1.join()
    t2.join()

    print('主线程')


'''
    加锁
'''

# balance = 0
#
# def change(n):
#     global balance        #用的是全局变量balance
#     balance+=n
#     balance-=n

# def run_thread(n):
#     for i in range(1000000):
#         change(n)

# lock = threading.Lock()  #获取线程锁
# def run_thread(n):
#     for i in range(1000000):
#         #获取锁
#         lock.acquire()
#         try:
#             change(n)
#         finally:
#             #释放锁
#             lock.release()
#
#
# t1 = threading.Thread(target= run_thread,args=(4,))       #args = 后面必须是一个元组。如果只有一个值，就必须在末尾加，
# t2 = threading.Thread(target= run_thread,args=(8,))
#
# t1.start()
# t2.start()
# t1.join()
# t2.join()
# print(balance)
