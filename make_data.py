from itertools import count
from re import X
import sys,os,math,time
import matplotlib.pyplot as plt
from numpy import *
import torch,torchvision

C = [0,1,1,1,0,
     1,0,0,0,1,
     1,0,0,0,0,
     1,0,0,0,1,
     0,1,1,1,0]
H = [1,0,0,0,1,
     1,0,0,0,1,
     1,1,1,1,1,
     1,0,0,0,1,
     1,0,0,0,1]
L = [1,0,0,0,0,
     1,0,0,0,0,
     1,0,0,0,0,
     1,0,0,0,0,
     1,1,1,1,1]
for i in range(25):
    if C[i] == 0:
        C[i] = 1
    else:
        C[i] = 0
    print(f'C{i} = ',C)
for i in range(25):
    if H[i] == 0:
        H[i] = 1
    else:
        H[i] = 0
    print(f'H{i} = ',H)
for i in range(25):
    if L[i] == 0:
        L[i] = 1
    else:
        L[i] = 0
    print(f'L{i} = ',L)