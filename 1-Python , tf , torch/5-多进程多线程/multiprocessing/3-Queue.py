#-*- conding:utf-8 -*-
import multiprocessing
import time
#queue 跨进程通信

def put(q):
   for value in ['A', 'B', 'C']:
       print ('发送 %s 到 queue...' % value)
       q.put(value)  #通过put发送数据到队列中
       time.sleep(2)

## 读数据进程执行的代码:
def get(q):
   while True:
       value = q.get(True)  #通过get接受队列中的数据
       print ('从 queue 接受 %s .' % value)

if __name__=='__main__':
   # 父进程创建Queue，并传给各个子进程：
   q = multiprocessing.Queue()
   pw = multiprocessing.Process(target=put, args=(q,))
   pr = multiprocessing.Process(target=get, args=(q,))
   # 启动子进程pw，写入:
   pw.start()
   # 启动子进程pr，读取:
   pr.start()
   # 等待pw结束:
   pw.join()
   # 在pr进程里，由于get（）方法处使用的是while True死循环，所以用terminate（）终止程序
   pr.terminate()
