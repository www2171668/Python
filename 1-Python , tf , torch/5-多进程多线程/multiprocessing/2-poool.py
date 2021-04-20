#-*- conding:utf-8 -*-
import os
import multiprocessing
import time

# def  work(n):
#    print('run work (%s) ,work id %s'%(n,os.getpid()))
#    time.sleep(2)
#    print('work (%s) stop ,work id %s'%(n,os.getpid()))
#
#
# if __name__=='__main__':
#    print('Parent process %s.' % os.getpid())
#
#    p = multiprocessing.Pool(3)      #创建大小为3的进程池
#    for i in range(4):
#         #创建5个进程，依次进入进程池
#         p.apply(work, args=(i,))
#         p.apply_async(work, args=(i,))
#    p.close()
#    p.join()

#p.apply：
#1、虽然是池，但是却是依次执行进程，所以是进入一个，完成一个，这样的操作同属于一个进程ID

#p.apply_async：
#1、由于进程池为3，前3个进程优先进入进程池，他们属于不同的进程ID
#2、运行时，优先运行第1个进程，此时第4个进程进入进程池，他们同属一个进程ID
#3、运行了第2个进程，第5个进程进入，他们同属一个进程ID
#4、运行第3、4、5个进程，他们属于不同的进程ID


def music(name,loop):
    print(time.ctime())
    for i in range(loop):
        time.sleep(2)
        print('您现在正在听的音乐是%s'%name)

def movie(name,loop):
    print(time.ctime())
    for i in range(loop):
        time.sleep(2)
        print('您现在正在看的电影是%s'%name)

if __name__=='__main__':
    pool=multiprocessing.Pool(2)
    pool.apply_async(func=music,args=('花太香',3))
    pool.apply_async(func=movie,args=('王牌特工',4))
    pool.apply_async(func=music, args=('爱的故事上集', 2))
    pool.close()
    # pool.terminate()
    # 比较危险,不要轻易用,直接杀死进程池
    #join阻塞主进程,当子进程执行完毕的时候会继续往后执行,使用join必须在进程池使用terminate或者close
    pool.join()
    print('结束时间是%s'%time.ctime())
