#-*- conding:utf-8 -*-
import time
import multiprocessing

'''
    单进程：多函数依次运行，使用普通的函数调用方法
    多进程：多函数同时运行。多进程会在主函数运行结束后才运行
        -》multiprocessing.Process(target,args) ：traget 函数，args 传参
        -》.start() ：通过start方法调用多进程化函数
'''
def work_1(f,n):   # f：文件 n：循环次数
    print('work_1 start')
    for i in range(n):
        with open(f,'a') as fs:
            fs.write('i love pyhton \n')
            time.sleep(1)
    print('work_1 end')

def work_2(f,n):
    print('work_2 start')
    for i in range(n):
        with open(f,'a') as fs:
            fs.write('come on baby \n')
            time.sleep(1)
    print('work_2 end')
#
# if __name__ == '__main__':   # 1、单进程
#     work_1('file.txt',3)
#     work_2('file.txt',3)

if __name__ == '__main__':   # 2、多进程
    p1 = multiprocessing.Process(target=work_1,args = ('file.txt',3))
    p2 = multiprocessing.Process(target=work_2, args=('file.txt', 3))

    p1.start()
    print('1')
    p2.start()
    print('2')

'''
    加锁
'''
# def work_1(f,n,lock):
#     print('work_1 start')
#     lock.acquire()
#     for i in range(n):
#         with open(f,'a') as fs:
#             fs.write('i love pyhton \n')
#             time.sleep(1)
#     print('work_1 end')
#     lock.release()
#
# def work_2(f,n,lock):
#     print('work_2 start')
#     lock.acquire()
#     for i in range(n):
#         with open(f,'a') as fs:
#             fs.write('come on baby \n')
#             time.sleep(1)
#     print('work_2 end')
#     lock.release()
#
# if __name__ == '__main__':
#     lock=multiprocessing.Lock()
#     p1 = multiprocessing.Process(target=work_1,args = ('file.txt',3,lock))
#     p2 = multiprocessing.Process(target=work_2, args=('file.txt', 3,lock))
#
#     p1.start()
#     p2.start()
