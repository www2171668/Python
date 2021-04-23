""""""
# %% time       time.time()   返回当前时间的时间戳（秒） ★       time.ctime()   以周，月，日，时分秒，年 的形式当前时间

import time

print(time.time())
print(time.ctime())

print('start')
time.sleep(5)  # time.sleep(秒数) ：推迟调用线程的运行
print('stop')

# %% 时间格式化
'''
    -》time.strftime(时间格式，时间元祖) ：时间元祖 转 标准时间
    -》time.strptime(标准时间，时间格式) ：标准时间 转 时间元祖 - 保证时间格式和标准时间保持一致
    -》time.mktime(时间元祖) ：时间元祖 转 时间戳
    -》time.localtime(时间戳) ：时间戳 转 时间元祖 - ()为空是返回当前时间元组
    
    凡是相对时间进行修改操作的，均需要在时间戳的状态下进行 ★
'''
# 1. 获取当前时间戳，并转换为标准时间
formatTime = time.localtime()
times = time.strftime('%Y-%m-%d %H:%M:%S', formatTime)  # \ 年月日，时分秒 ；注意大小写s
print(times)

# 2. 字符串格式更改,将time = "2017-10-10 23:40:00"改为time = "2017/10/10 23:40:00"
times = '2017-10-10 23:40:00'
formatTime = time.strptime(times, '%Y-%m-%d %H:%M:%S')  # \ 先转时间元组
new_formatTime = time.strftime('%Y/%m/%d %H:%M:%S', formatTime)  # \ 再转标准时间
print(new_formatTime)

# 3.
timestamp = time.mktime(formatTime)
print(timestamp)

# %% datetime.timedelta(seconds=秒数)：秒转时分
import datetime

start = time.process_time()    # 获取当前时间
step = 0
for _ in range(100000000):
    step += 1
end = time.process_time()
times = datetime.timedelta(seconds=end - start)
print(times)
