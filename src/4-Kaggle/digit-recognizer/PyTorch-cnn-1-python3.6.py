#!/usr/bin/python3
# coding: utf-8

'''
Created on 2017-12-18
Update  on 2017-12-18
Author: 片刻
Github: https://github.com/geekhoo/AILearning
Result: [10,  1000] loss: 0.008  Accuracy of the network on the 10000 test images: 99.140 %
'''
import os

# third-party library
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

# torch.manual_seed(1)    # reproducible
# Hyper Parameters

DOWNLOAD_MNIST = False
# Mnist digits dataset
if not(os.path.exists('data/3-DeepLearning/mnist')) or not os.listdir('data/3-DeepLearning/mnist'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='data/3-DeepLearning/mnist',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
                                                    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=DOWNLOAD_MNIST,
)
test_data = torchvision.datasets.MNIST(
    root='data/3-DeepLearning/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)

# # plot one example
# print(train_data.train_data.size())                 # (60000, 28, 28)
# print(train_data.train_labels.size())               # (60000)
# plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)
BATCH_SIZE = 50
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         # input shape (1, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


cnn = CNN()
# print(cnn)  # net architecture

LR = 0.001              # learning rate
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()                       # the target label is not one-hotted

# training and testing
print(u'开始训练')
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
for epoch in range(EPOCH):
    running_loss = 0.0

    for step, (x, y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader
        b_x = Variable(x)   # batch x
        b_y = Variable(y)   # batch y

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        # 每1000批数据打印一次平均loss值
        running_loss += loss.data[0]        # loss本身为Variable类型，所以要使用data获取其Tensor，因为其为标量，所以取0
        if step % 1000 == 999:              # 每2000批打印一次
            print('[%d, %5d] loss: %.3f' % (epoch+1, step+1, running_loss/1000))
            running_loss = 0.0
print('Finished Training')


correct = 0
total = 0
for img, label in test_loader:
    img = Variable(img, volatile=True)
    label = Variable(label, volatile=True)

    outputs = cnn(img)
    _, predicted = torch.max(outputs[0], 1)
    # print('1-', type(label), '-------', label)
    # print('2-', type(predicted), '-------', predicted)
    total += label.size(0)
    num_correct = (predicted == label).sum()
    correct += num_correct.data[0]

print('Accuracy of the network on the %d test images: %.3f %%' % (total, 100 * correct / total))
