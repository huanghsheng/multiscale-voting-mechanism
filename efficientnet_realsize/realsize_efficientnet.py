# -*- coding: utf-8 -*-
"""
Created on Wed May 26 09:58:17 2021

@author: Prince
"""

import os
from PIL import Image

import torch
import torchvision
import sys
from model import EfficientNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import optim
from torch import nn
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
#import d2lzh_pytorch as d2l
from time import time
import time
import csv
import os
import random
import shutil
import csv
from PIL import Image
from csv import writer


class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, is_train, root):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh:  # 按行循环txt文本中的内容
            line = line.rstrip()  # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            words = line.split()  # 通过指定分隔符对字符串进行切片，默认为所有的空字符，包括空格、换行、制表符等
            imgs.append((words[0], int(words[1])))  # 把txt里的内容读入imgs列表保存，具体是words几要看txt内容而定
            # word[0]是文件名, word[1]为标签
        self.imgs = imgs
        self.is_train = is_train

        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.CenterCrop(400,300),
                # torchvision.transforms.Resize(400,300),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                # torchvision.transforms.CenterCrop(400,300),
                # torchvision.transforms.Resize(400,300),
                torchvision.transforms.ToTensor()])

        # 归一化
        # if self.is_train:
        #     self.train_tsf = torchvision.transforms.Compose([
        #         torchvision.transforms.RandomResizedCrop(224, scale=(0.1, 1), ratio=(0.5, 2)),
        #         torchvision.transforms.ToTensor()
        #     ])
        # else:
        #     self.test_tsf = torchvision.transforms.Compose([
        #         torchvision.transforms.Resize(size=224),
        #         torchvision.transforms.CenterCrop(size=224),
        #         torchvision.transforms.ToTensor(),
        #         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        if self.is_train:
            feature = self.train_tsf(feature)
        else:
            feature = self.test_tsf(feature)
        return feature, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


def READ(num_epochs,device,batch_size,optimizer,loss,net):
    train_k = r'/home/jinfei/PycharmProjects/EfficientNet-Pytorch-master/efficientnet_realsize/400-300-2_train.txt'
    test_k = r'/home/jinfei/PycharmProjects/EfficientNet-Pytorch-master/efficientnet_realsize/400-300-2_test.txt'
    # train_k = r'/home/jinfei/PycharmProjects/EfficientNet-Pytorch-master/efficientnet_realsize/400-300-4_train.txt'
    # test_k = r'/home/jinfei/PycharmProjects/EfficientNet-Pytorch-master/efficientnet_realsize/400-300-4_test.txt'
    #loss_acc_sum,train_acc_sum, test_acc_sum = 0,0,0
    Ktrain_min_l = []
    Ktrain_acc_max_l = []
    Ktest_acc_max_l = []
    #修改train函数，使其返回每一批次的准确率，tarin_ls用列表表示
    train_data = MyDataset(is_train=True, root=train_k)
    test_data = MyDataset(is_train=False, root=test_k)

    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=180, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=180, shuffle=True, num_workers=0)

    loss_min,train_acc_max,test_acc_max=train(train_loader,test_loader, net, loss, optimizer, device, num_epochs)
    Ktrain_min_l.append(loss_min)
    Ktrain_acc_max_l.append(train_acc_max)
    Ktest_acc_max_l.append(test_acc_max)
    #train_acc_sum += train_acc# train函数epoches（即第k个数据集被测试后）结束后，累加
    #test_acc_sum += test_acc#
    #loss_acc_sum+=loss_acc
    #print('fold %d, lose_rmse_max %.4f, train_rmse_max %.4f, test_rmse_max %.4f ' %(i+1, loss_acc,train_acc, test_acc_max_l[i]))
    return sum(Ktrain_min_l)/len(Ktrain_min_l),sum(Ktrain_acc_max_l)/len(Ktrain_acc_max_l),sum(Ktest_acc_max_l)/len(Ktest_acc_max_l)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum,test_l_sum , n = 0.0,0.0, 0
    f = open("//home//jinfei//PycharmProjects//pythoncodes//EfficientNet_b0_test time.txt", "a")
    t1 = time.time()
    with torch.no_grad():
        batch_count = 0
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            l = loss(net(X), y)
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else:
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            batch_count +=1
            n += y.shape[0]
            test_l_sum += l.cpu().item()
        # print("test number :", n)
    t2 = time.time()
    f.write(str(t2 - t1) + " " + '\n')
    print("test time %.4f sec:", t2 - t1)
    return (acc_sum / n),(test_l_sum/batch_count)

def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)

    f1 = open("//home//jinfei//PycharmProjects//pythoncodes//results.txt", "a")
    f2 = open("//home//jinfei//PycharmProjects//pythoncodes//results data.txt", "w")
    start = time.time()
    test_acc_max_l = []
    test_l_min_l = []
    train_acc_max_l = []
    train_l_min_l=[]

    # step_schedule = optim.lr_scheduler.StepLR(step_size=100, gamma=0.8, optimizer=optimizer)
    # reduce_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
    #                                                        verbose=False, threshold=1e-3, threshold_mode='rel',
    #                                                        cooldown=0, min_lr=0, eps=1e-3)
    for epoch in range(num_epochs):
        batch_count = 0
        train_l_sum, train_acc_sum,test_l_sum ,test_acc_sum, n = 0.0, 0.0, 0.0, 0.0, 0
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        # step_schedule.step()
        # reduce_schedule.step(l)

        #至此，一个epoches完成
        test_acc_sum,test_l_sum = evaluate_accuracy(test_iter, net)
        test_l_min_l.append(test_l_sum)
        train_l_min_l.append(train_l_sum/batch_count)
        train_acc_max_l.append(train_acc_sum/n)
        test_acc_max_l.append(test_acc_sum)

        # if (epoch+1)%100==0:
        #     torch.save(net.module.state_dict(), "./weights/b0-{}.pth".format(epoch+1))

        print('epoch %d, train loss %.4f, train acc %.4f, test loss %.4f, test acc %.4f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,test_l_sum, test_acc_sum))

        f1.write('epoch %d, train loss %.4f, train acc %.4f, test loss %.4f, test acc %.4f'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n,test_l_sum, test_acc_sum)+'\n')

        f2.write(str(epoch+1)+" "+str(train_l_sum / batch_count)+" "+str(train_acc_sum / n)+" "+str(test_l_sum)+" "+str(test_acc_sum)+" "+'\n')
    #train_l_min_l.sort()
    #train_acc_max_l.sort()
    index_max=test_acc_max_l.index(max(test_acc_max_l))
    f = open("//home//jinfei//PycharmProjects//pythoncodes//b0_results.txt", "a")
    f.write("train_loss"+"       "+"train_acc"+"      "+"test_loss"+"       "+"test_acc")
    f.write('\n'+str(train_l_min_l[index_max]) + " ;" + str(train_acc_max_l[index_max]) + " ;" +str(test_l_min_l[index_max]) + " ;" + str(test_acc_max_l[index_max]))
    f.close()


    print('train_loss_min %.4f, train acc max%.4f,test_loss_min %.4f, test acc max %.4f, time %.1f sec'
            % ( train_l_min_l[index_max], train_acc_max_l[index_max],test_l_min_l[index_max],test_acc_max_l[index_max], time.time() - start))
    f1.write('train_loss_min %.4f, train acc max%.4f,test_loss_min %.4f, test acc max %.4f, time %.1f sec'
            % ( train_l_min_l[index_max], train_acc_max_l[index_max],test_l_min_l[index_max],test_acc_max_l[index_max], time.time() - start)+'\n')
    f1.close()
    f2.close()
    return train_l_min_l[index_max],train_acc_max_l[index_max],test_acc_max_l[index_max]



batch_size=180
num_epochs=100
lr = 0.01
net = EfficientNet.from_name('efficientnet-b0')

#得到想要继承的模型的预训练参数pretext_model
pretext_model = torch.load('./efficientnet-b0.pth')
net_dict = net.state_dict()
#保留继承的参数中，key(层的名称)存在于mymodel的键值对
state_dict = {k: v for k, v in pretext_model.items() if k in net_dict.keys()}
#更新mymodel的参数
net_dict.update(state_dict)
#导入更新后的参数
net.load_state_dict(net_dict)
net._fc1 = nn.Linear(100, 5)

# net._fc1 = nn.Linear(100, 5)
# pretext_model = torch.load('./pth/b0_disease_model_500~1000ep_0.05.pth')
# net_dict = net.state_dict()
# state_dict = {k: v for k, v in pretext_model.items() if k in net_dict.keys()}
# net_dict.update(state_dict)
# net.load_state_dict(net_dict)




#b0,b1
#net.last_conv = nn.Conv2d(1280, 5, kernel_size=1)

# net._fc4 = nn.Linear(15360, 5)
#b6
#net._fc = nn.Linear(2304, 3)
output_params = list(map(id, net._fc1.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, net.parameters())

optimizer = optim.SGD([{'params': feature_params},
                       {'params': net._fc1.parameters(), 'lr': lr * 10}],
                      lr=lr, weight_decay=0.001)
net=net.cuda(0)
net = torch.nn.DataParallel(net)
loss = torch.nn.CrossEntropyLoss()
loss_k,train_k, valid_k=READ(num_epochs,device,batch_size,optimizer,loss,net)
f=open("//home//jinfei//PycharmProjects//pythoncodes//b0_results.txt","a")
f.write('\n'+"result:"+"\n"+str(loss_k)+" ;"+str(train_k)+" ;"+str(valid_k))
f.close()
print('result: min loss rmse %.5f, max train rmse %.5f,max test rmse %.5f' % (loss_k,train_k, valid_k))
print("Congratulations!!! ")
torch.save(net.module.state_dict(), "///home//jinfei//PycharmProjects//pythoncodes//b0_disease_model_400-300-2.pth")