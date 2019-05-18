# -*- coding: utf-8 -*-
import random
import numpy as np
import matplotlib.pyplot as plt
random.seed(1)
data_0 = np.array([[2 + random.random(), 2 + random.random()] for i in np.arange(6)])
data_1 = np.array([[3 + random.random(), 3 + random.random()] for i in np.arange(6)])
data = np.concatenate((data_0,data_1))
label = np.array([0,0,0,0,0,0,1,1,1,1,1,1])
#边界线方程：a0+a1*x1+a2*x2=0
a0,a1,a2 = 1,1,1 #初始值
alpha = 0.002
threshold = 0.2

def hyp(x1, x2): #假设函数
    return 1/(1+np.exp(-np.matmul(np.array([a0,a1,a2]), np.array([1,x1,x2]))))
def alter(data, label, n_a):
    sum = 0
    if n_a == 0:
        for i in range(len(label)):
            sum += (hyp(data[i][0], data[i][1]) - label[i])
    else:
        for i in range(len(label)):
            sum += (hyp(data[i][0], data[i][1]) - label[i])*data[i][n_a-1]
    return sum*alpha
def loss():
    sum = 0
    for i in range(len(label)):
        sum += label[i]*np.log(hyp(data[i][0], data[i][1])) + (1-label[i])*np.log(1-hyp(data[i][0], data[i][1]))
    return sum / (-len(label))

linex = np.linspace(0,4,5)
for i in range(10000):
    a0 -= alter(data, label, 0)
    a1 -= alter(data, label, 1)
    a2 -= alter(data, label, 2)
    if i % 100 == 0:
        print('第%d次,loss=' % i, loss())
        if i % 1000 == 0:
            liney = (a0 + a1*linex)/(-a2)
            plt.plot(linex, liney, label=i)
        if loss() < threshold:
            break
plt.scatter(data[:,0], data[:,1], c=label)
plt.legend()
plt.show()
