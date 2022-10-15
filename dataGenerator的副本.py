import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs

##################月牙形数据集合###################
#两类
#ret：[x1,y1,x2,y2]，注意y不是标签值，标签与下标相同，(x,y)代表该点的二维坐标值
def moon2Data(datanum):
    x1 = np.linspace(-3, 3, datanum)
    noise = np.random.randn(datanum) * 0.15
    y1 = -np.square(x1) / 3 + 4.5 + noise

    x2 = np.linspace(0, 6, datanum)
    noise = np.random.randn(datanum) * 0.15
    y2 = np.square(x2 - 3) / 3 + 0.5 + noise

    return [x1,y1,x2,y2]

    # 绘图代码
# if __name__ == '__main__':
#     [x1,y1,x2,y2] = moon2Data(300)
#     plt.clf()
#     plt.axis([-3.5, 6.5, -.5, 5.5])
#     plt.scatter(x1, y1, s=10)
#     plt.scatter(x2,y2,s = 10)
#     plt.show()




##################方形数据集合###################
#三类
#ret:[x1,y1,x2,y2,x3,y3]
def square2Data(datanum):
    x = np.random.rand(datanum, 2)
    condition1 = x[:, 1] <= x[:, 0]
    condition2 = x[:, 1] <= (1 - x[:, 0])
    index1 = np.where(condition1 & condition2)
    x1 = x[index1]
    x = np.delete(x, index1, axis=0)

    index2 = np.where(x[:, 0] <= 0.5)
    x2 = x[index2]
    x3 = np.delete(x, index2, axis=0)

    return [x1[:,0],x1[:,1],x2[:,0],x2[:,1],x3[:,0],x3[:,1]]
#绘图代码
# if __name__ == '__main__':
#     [x1,y1,x2,y2,x3,y3] = square2Data(1000)
#
#     plt.clf()
#     plt.scatter(x1,y1,s=10)
#     plt.scatter(x2,y2,s=10)
#     plt.scatter(x3,y3,s=10)
#     plt.show()




##################环形数据集合###################
#两类
#ret：[x1,y1,x2,y2]
def circle2Data(datanum):
    x,y = make_circles(n_samples=datanum, noise=0.07, random_state=np.random.randint(0, 1000), factor=0.6)
    x1 = x[y==0][:,0]
    y1 = x[y==0][:,1]
    x2 = x[y==1][:,0]
    y2 = x[y==1][:,1]
    return [x1,y1,x2,y2]
#绘图代码
# if __name__ == '__main__':
#     [x1,y1,x2,y2] = circle2Data(1000)
#
#     plt.clf()
#     plt.scatter(x1,y1,s=10)
#     plt.scatter(x2,y2,s=10)
#     plt.show()




##################点簇形数据集合###################
#类别数=classNum
#ret：[x1,y1,x2,y2,...]
def blobs2Data(datanum):
    #classNum：类别数
    classNum = 4
    x,y = make_blobs(n_samples=datanum, n_features=2, centers=classNum, random_state=np.random.randint(0, 1000))
    x1 = x[y==0][:,0]
    y1 = x[y==0][:,1]
    x2 = x[y==1][:,0]
    y2 = x[y==1][:,1]
    x3 = x[y==2][:,0]
    y3 = x[y==2][:,1]
    x4 = x[y==3][:,0]
    y4 = x[y==3][:,1]
    return [x1,y1,x2,y2,x3,y3,x4,y4]
# if __name__ == '__main__':
#     [x1,y1,x2,y2,x3,y3,x4,y4] = blobs2Data(1200)
#
#     plt.clf()
#     plt.scatter(x1,y1,s=10)
#     plt.scatter(x2,y2,s=10)
#     plt.scatter(x3,y3,s=10)
#     plt.scatter(x4,y4,s=10)
#
#     plt.show()

if __name__ == '__main__':
    [x1,y1,x2,y2] = moon2Data(300)
    plt.clf()
    plt.axis([-3.5, 6.5, -.5, 5.5])
    plt.scatter(x1, y1, s=10)
    plt.scatter(x2,y2,s = 10)
    plt.show()