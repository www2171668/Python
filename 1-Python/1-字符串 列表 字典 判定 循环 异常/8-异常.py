""""""

# %% 异常      try:        except:
# * NameError 捕获变量命名异常      Exception 捕获所有异常  ★
# 1、-》
try:
    print(aaa)  # \ 如果这句话有错，就会捕获到异常
except NameError:
    print('变量未定义')

# 2、捕获异常的具体信息
try:
    print(aaa)  # 如果这句话有错，就会捕获到异常
except NameError as e:
    print(e)  # 打印具体的异常信息。输出 name 'aaa' is not defined
    print('变量未定义')

# 3、包含多个异常
try:
    print(aaa)
    files = open('aaa.txt', 'r', encoding='utf-8')  # 如果这句话有错，就会捕获到异常
except (NameError, FileNotFoundError) as e:  # 即使同时捕获到两个异常，也只会报错前一个
    print(e)  # 打印具体的异常信息

# 4、
try:
    print(aaa)
    files = open('aaa.txt', 'r', encoding='utf-8')
except Exception as e:
    print(e)

# %% else 或 finally ：不管有没有异常都会执行的代码块
try:
    print('打开文件！')
    files = open('aaa.txt', 'w', encoding='utf-8')
    try:
        files.write('测试一下行不行')
    except:
        print('写入失败')
    else:
        print('写入成功')
    finally:
        print('关闭文件')
        files.close()
except Exception as e:
    print(e)

# %%  assert 判断表达式，在条件为 false 时触发异常
expression = 2 == 3
assert expression

# 等价于
if not expression:
    raise AssertionError
