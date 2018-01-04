#!/usr/bin/python3
# coding: utf-8

'''
Created on 2017-12-18
Update  on 2017-12-18
Author: 片刻
Github: https://github.com/geekhoo/AILearning
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms # transforms用于数据预处理


# 数据预处理：数据归一化，将数据变化[-1, 1]之间。（预处理会帮助我们加快神经网络的训练）
# torchvision输出的是PILImage，值的范围是[0, 1].
# 我们将其转化为tensor数据，并归一化为[-1, 1]。
'''
1. Compose 函数会将多个 transforms 包在一起
2. ToTensor是指把PIL.Image(RGB) 或者numpy.ndarray(H x W x C) 从0到255的值映射到0到1的范围内，并转化成Tensor格式
3. Normalize(mean，std)是通过下面公式实现数据归一化（处理公式：channel =（channel-mean）/std）
'''
transform = transforms.Compose([
                                transforms.ToTensor(), 
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])


# 数据读取
# 测试集，将相对目录./data下的cifar-10-batches-py文件夹中的全部数据（10000张图片作为测试数据）加载到内存中，若download为True时，会自动从网上下载数据并解压
'''
1. root，表示cifar10数据的加载的相对目录
2. train，表示是否加载数据库的训练集，false的时候加载测试集
3. download，表示是否自动下载cifar数据集
4. transform，表示是否需要对数据进行预处理，none为不进行预处理
'''
# trainset = torchvision.datasets.CIFAR10(root='data/3-DeepLearning/cifar10', train=True, download=False, transform=None)
trainset = torchvision.datasets.CIFAR10(root='data/3-DeepLearning/cifar10', train=True, download=False, transform=transform)
testset = torchvision.datasets.CIFAR10(root='data/3-DeepLearning/cifar10', train=False, download=False, transform=transform)

# 将测试集的10000张图片划分成2500份，每份4张图，用于mini-batch输入。
'''
1. batch_size=4，将训练集的50000张图片划分成12500份，每份4张图，用于mini-batch输入。
2. shffule=True，在表示不同批次的数据遍历时，打乱顺序（这个需要在训练神经网络时再来讲）。
3. num_workers=2，表示使用两个子进程来加载数据
'''
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

print(len(trainset))
print(len(trainloader))


# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# # 下面是代码只是为了给小伙伴们显示一个图片例子，让大家有个直觉感受。
# # functions to show an image
# import matplotlib.pyplot as plt
# import numpy as np
# #matplotlib inline
# def imshow(img):
#     img = img / 2 + 0.5 # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # show some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# 创建一个 CNN 模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)     # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即彩色图）,输出为6张特征图, 卷积核为5x5正方形
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()


#  优化器
'''
使用SGD（随机梯度下降）优化，学习率为0.001，动量为0.9

lr=0.01
    [10, 12000] loss: 0.848  Accuracy of the network on the 10000 test images: 62 %
lr=0.001
    [10, 12000] loss: 1.195  Accuracy of the network on the 10000 test images: 57 %
lr=0.01, momentum=0.9
    [10, 12000] loss: 2.033  Accuracy of the network on the 10000 test images: 23 %
lr=0.01, momentum=0.8
    [10, 12000] loss: 1.647  Accuracy of the network on the 10000 test images: 40 %
'''
optimizer = optim.SGD(net.parameters(), lr=0.01)         # [10, 12000] loss: 0.848  Accuracy of the network on the 10000 test images: 62 %
# optimizer = optim.SGD(net.parameters(), lr=0.001)        # [10, 12000] loss: 1.195  Accuracy of the network on the 10000 test images: 57 %
# optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)        # [10, 12000] loss: 1.195  Accuracy of the network on the 10000 test images: 57 %
# , alpha=0.9
optimizer = optim.Adam(net.parameters(), lr=0.01) 
# 损失函数
criterion = nn.CrossEntropyLoss()  # 叉熵损失函数


for epoch in range(10):      # 遍历数据集两次

    running_loss = 0.0
    # enumerate(sequence, [start=0])，i序号，data是数据
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data       # data的结构是：[4x3x32x32的张量,长度4的张量]

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)     # 把input数据从tensor转为variable

        # zero the parameter gradients
        optimizer.zero_grad()                   # 将参数的grad值初始化为0
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)       # 将output和labels使用叉熵计算损失
        loss.backward()                         # 反向传播
        optimizer.step()                        # 用SGD更新参数

        # 每2000批数据打印一次平均loss值
        running_loss += loss.data[0]            # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if i % 2000 == 1999:                    # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0
 
print('Finished Training')

correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    #print outputs.data
    _, predicted = torch.max(outputs.data, 1)       # outputs.data是一个4x10张量，将每一行的最大的那一列的值和序号各自组成一个一维张量返回，第一个是值的张量，第二个是序号的张量。
    total += labels.size(0)
    correct += (predicted == labels).sum()          # 两个一维张量逐行对比，相同的行记为1，不同的行记为0，再利用sum(),求总和，得到相同的个数。
 
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
