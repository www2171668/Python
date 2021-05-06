#!/bin/bash

# 通过./run_experiments.sh来运行
# wait 执行完上一个程序才会执行下一个

"""
  nohup ./startWebLogic.sh >out.log 2>&1 &
  nohup+最后面的& 是让命令在后台执行
  >out.log 是将信息输出到out.log日志中
  2>&1 是将标准错误信息转变成标准输出，这样就可以将错误信息输出到out.log 日志里面来。
"""

source activate tf

for ((i=0;i<1;i+=1))
do
	nohup python -u AE.py \
  > log/AE"$i".log 2>&1 &   # 在log文件夹中记录log文件
  wait
done

echo "结束......" # echo 输出