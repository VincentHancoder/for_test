import sys,os,math,time
import matplotlib.pyplot as plt
from numpy import *

xdim = [(-0.1,0.3), (0.5,0.7), (-0.5,0.2),(-0.7,0.3),(0.7,0.1),(0,0.5)]#数据集
ddim = [1,-1,1,1,-1,1]#真值标签

def sigmoid(x):#sigmoid激活函数定义
    return 1/(1+exp(-x))

def hebbian(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])#w指的是[b,w1,w2],x指[1,x[0],x[1]],将w与x组成的坐标元组数组中的对象挨个相乘，最后进行求和
    o = sigmoid(net)#使用sigmoid激活函数作为神经元模型的输出，后续相同就省略
    w1 = [ww+o*xx for ww,xx in zip(w,x1)]#每一个更新后的权重是前一个权重加上 学习率（这里默认为1）x 神经元输出值 x 输入向量（公式就是这样定义）
    return w1

def perceptron(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])
    o = 1 if net >= 0 else -1#二值激活函数
    w1 = [ww+(d-o)*xx for ww,xx in zip(w,x1)]
    return w1

def delta(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])
    o = sigmoid(net)
    o1 = o*(1-o)
    w1 = [ww+(d-o)*o1*xx for ww,xx in zip(w,x1)]
    return w1

def widrawhoff(w,x,d):
    x1 = [1,x[0],x[1]]
    net = sum([ww*xx for ww,xx in zip(w, x1)])
    o = sigmoid(net)
    w1 = [ww+(d-o)*xx for ww,xx in zip(w,x1)]
    return w1

def correlation(w,x,d):
    x1 = [1,x[0],x[1]]
    w1 = [ww+d*xx for ww,xx in zip(w,x1)]
    return w1

wb = [0,0,0]                        # [b, w1, w2]
print('hebbian')
for x,d in zip(xdim, ddim):#x、d是xdim、ddim中的数组元素，x代表坐标，d代表类别；而ww、xx代表数组w与x1中的参数；x1数组比数组x多1个1，用于与数组w中b相乘
    wb1 = hebbian(wb,x,d)
    print(wb1)

print('perceptron')
for x,d in zip(xdim, ddim):
    wb2 = perceptron(wb,x,d)
    print(wb2)

print('delta')
for x,d in zip(xdim, ddim):
    wb3 = delta(wb,x,d)
    print(wb3)

print('widrawhoff')
for x,d in zip(xdim, ddim):
    wb4 = widrawhoff(wb,x,d)
    print(wb4)

print('correlation')
for x,d in zip(xdim, ddim):
    wb5 = correlation(wb,x,d)
    print(wb5)
