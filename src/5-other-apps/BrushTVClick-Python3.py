# -*- coding:utf-8 -*-
import os
import time
import random
import webbrowser as web
# count = random.randint(1, 2)  # 产生2,4之间的随机数
count = 200
num = 0
while num < count:
    i = 0
    while i <= 9:
        # 声明代理集合
        # webbrowser_list = ['chrome', 'firefox', 'safari']
        webbrowser_list = ['firefox', 'safari']
        web.get(random.choice(webbrowser_list)).open_new_tab('..')
        i = i + 1
        time.sleep(random.randint(1, 300))
        print(i)
    else:
        print("open browser")
        os.system('taskkill /F /IM TheWorld.exe')  # 杀死进程 TheWorld.exe
        print(num + 1, "times close browser")
    num = num + 1
print("一共刷新： ", num * 10, " 次")
