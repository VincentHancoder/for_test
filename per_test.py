from itertools import count
import sys,os,math,time
import matplotlib.pyplot as plt
from numpy import *
import numpy as np
import pylab as pl

xdim = [(1,1,1),(1,1,0),(1,0,1),(0,1,1)]#数据集
ddim = [1,0,0,0]#二值真值标签
ddim1 = [1,-1,-1,-1]#双极性真值标签
n = np.arange(0,1,0.01)#学习率
emin = 0.005#误差阈值
error = 0#误差
X1 = []
X2 = []

def sigmoid(x):#sigmoid激活函数定义
    return 1/(1+exp(-x))

def hebbian(w,x,d,n):
    x1 = [1,x[0],x[1],x[2]]
    count = 0
    for i in range(100): 
        net = sum([ww*xx for ww,xx in zip(w, x1)])#w指的是[b,w1,w2],x指[1,x[0],x[1]],将w与x组成的坐标元组数组中的对象挨个相乘，最后进行求和
        o = sigmoid(net)#使用sigmoid激活函数作为神经元模型的输出，后续相同就省略
        count += 1
        w1 = [ww+n*o*xx for ww,xx in zip(w,x1)]#每一个更新后的权重是前一个权重加上 学习率（这里默认为1）x 神经元输出值 x 输入向量（公式就是这样定义）
        error = d-o
        if error > emin:
            w = w1
        else:
            break 
    return w,count



wb = [0,0,0,0]#初始权重为0
x1 = [1,1,1,1]
d = 1
for i in range(100):
    print('hebbian')                        # [b, w1, w2 ,w3]
    nn = n[i]
    #for x,d in zip(xdim, ddim):#x、d是xdim、ddim中的数组元素，x代表坐标，d代表类别；而ww、xx代表数组w与x1中的参数；x1数组比数组x多1个1，用于与数组w中b相乘
    wb1,c1 = hebbian(wb,x1,d,nn)
    print(wb1,f'epoch = {c1}')
    X1.append(c1)


pl.plot(n,X1)  #画图
pl.xlabel("lr")  #x轴的标记
pl.ylabel("epoch")  #y轴的标记
pl.title("epoch_lr")  #图的标题
pl.show()   #显示图
plt.savefig('/home/hhn/lr_epoch.jpg')


    

    # for x,d in zip(xdim, ddim1):#x、d是xdim、ddim中的数组元素，x代表坐标，d代表类别；而ww、xx代表数组w与x1中的参数；x1数组比数组x多1个1，用于与数组w中b相乘
    #     wb2,c2 = hebbian(wb,x,d,nn)
    #     X2.append(c1)
    #     print(wb2,f'epoch = {c2}')
