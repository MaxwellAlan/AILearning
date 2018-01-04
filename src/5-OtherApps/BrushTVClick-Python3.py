#!/usr/bin/python3
# coding: utf-8

import os
import time
import random
import webbrowser as web

# 注册浏览器信息
safari_path = r'D:\Program Files\Safari\Safari.exe'                 # 例如我的：C:\***\***\***\***\Google\Chrome\Application\chrome.exe 
web.register('Safari', None, web.BackgroundBrowser(safari_path))    # 这里的'chrome'可以用其它任意名字，如chrome111，这里将想打开的浏览器保存到'chrome'

firefox_path = r'D:\Program Files\Mozilla Firefox\firefox.exe'                 # 例如我的：C:\***\***\***\***\Google\Chrome\Application\chrome.exe 
web.register('firefox', None, web.BackgroundBrowser(firefox_path))    # 这里的'chrome'可以用其它任意名字，如chrome111，这里将想打开的浏览器保存到'chrome'

chrome_360_path = r'D:\Program Files\360Chrome\Chrome\Application\360chrome.exe'                 # 例如我的：C:\***\***\***\***\Google\Chrome\Application\chrome.exe 
web.register('360chrome', None, web.BackgroundBrowser(chrome_360_path))    # 这里的'chrome'可以用其它任意名字，如chrome111，这里将想打开的浏览器保存到'chrome'

the_world_6_path = r'D:\Program Files\TheWorld6\Application\TheWorld.exe'                 # 例如我的：C:\***\***\***\***\Google\Chrome\Application\chrome.exe 
web.register('TheWorld', None, web.BackgroundBrowser(the_world_6_path))    # 这里的'chrome'可以用其它任意名字，如chrome111，这里将想打开的浏览器保存到'chrome'

iexplore_path = r'C:\Program Files\Internet Explorer\iexplore.exe'                 # 例如我的：C:\***\***\***\***\Google\Chrome\Application\chrome.exe 
web.register('iexplore', None, web.BackgroundBrowser(iexplore_path))    # 这里的'chrome'可以用其它任意名字，如chrome111，这里将想打开的浏览器保存到'chrome'

chrome_path = r'C:\Users\Administrator\AppData\Local\Google\Chrome\Application\chrome.exe'                 # 例如我的：C:\***\***\***\***\Google\Chrome\Application\chrome.exe 
web.register('chrome', None, web.BackgroundBrowser(chrome_path))    # 这里的'chrome'可以用其它任意名字，如chrome111，这里将想打开的浏览器保存到'chrome'

# 设置线程
from multiprocessing.dummy import Pool as ThreadPool

global num, count


def show(newbrowser):
    num = 0
    count = 100
    while num < count:
        i = 0
        while i <= 3:
            # 声明代理集合
            web.get(newbrowser).open('xxx', new=0, autoraise=True)
            
            time.sleep(random.randint(20, 300))
            i = i + 1
            print(i)
        else:
            print("open browser")
            os.system("taskkill /F /IM  %s.exe" % newbrowser)
            # os.system("kill -s 9 `pgrep %s`" % newbrowser)  # 杀死进程 TheWorld.exe
            print(num + 1, "times close browser")
        num = num + 1
        print(newbrowser, u"本次刷新： ", num * 10, " 次")
    print(newbrowser, u"一共刷新： ", num * 10, " 次")


if __name__ == '__main__':
    # <type 'list'>
    list_task = ['chrome', 'firefox', 'Safari', '360chrome', 'TheWorld', 'iexplore']

    # 创建一个的线程池 （设置对应的线程数）
    # 如果task多，那么会等线程池中有空余的线程后，执行后续的任务
    pool = ThreadPool(6)
    # 然后每一个任务，执行对应的方法
    pool.map(show, list_task)
    # print results

    # 关闭池, 等待任务完成
    pool.close()
    pool.join()
    print('done...')
