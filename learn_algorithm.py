import sys,os,math,time
import matplotlib.pyplot as plt
from numpy import *

xdim = [(-0.1,0.3), (0.5,0.7), (-0.5,0.2),(-0.7,0.3),(0.7,0.1),(0,0.5)]#各个样本的数据
ldim = [1,-1,1,1,-1,1]#label，1代表...,-1代表...

print("序列", "X1", "X2", "类别")

count = 0
for x,l in zip(xdim, ldim):

    count += 1
    print("%d %3.1f %3.1f %d"%(count, x[0], x[1], l))

    if l > 0:
        marker = 'o'
        color = 'blue'
    else:
        marker = '+'
        color = 'red'

    plt.scatter(x[0], x[1], marker=marker, c=color)
    plt.text(x[0]+0.05,x[1],'(%3.1f,%3.1f)'%(x[0],x[1]))

plt.axis([-0.8, 0.8,-0.1, 1])
plt.xlabel("X1")
plt.ylabel("X2")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.savefig('/home/hhn/learn_algrothim1.jpg')
