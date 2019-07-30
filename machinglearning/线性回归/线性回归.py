# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#单变量线性回归
path = r'C:/Users/DH/Desktop/ex1data1.txt'
data = pd.read_csv(path,names=['Population','Profit'])
#print(data.describe())
plt.scatter(data['Population'], data['Profit'])
plt.xlabel('Population')
plt.ylabel('Profit')
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(8,5))
#plt.show()

def computeCost(X,Y,theta):
    inner = np.power((X*theta.T)-Y,2)
    return np.sum(inner)/(2*len(X))

data.insert(0,'Ones',1)
#print(data.head())

#set X(training data) and Y(target varible)
cols = data.shape[1] #列数
X = data.iloc[:,0:cols-1] #取前cols-1列，即输入向量,.iloc函数从0开始计数，选取列，第一个参数选取行，第二参数选取列
Y = data.iloc[:,cols-1:cols] #取最后一列，即目标向量
#print(X.head(),Y.head()) #.head()选取前五行

#将X,Y转换为矩阵，初始化theta为一个（1,2）矩阵
X = np.matrix(X.values)
Y = np.matrix(Y.values)
theta =np.matrix([0,0])

#print(computeCost(X,Y,theta)) #初始32.072733877455676
def gradientDescent(X, Y, theta, alpha, epoch):
    temp = np.matrix(np.zeros(theta.shape))  # 初始化一个临时矩阵(1, 2)
    #parameters = int(theta.flatten().shape[1])  # 参数 θ的数量
    cost = np.zeros(epoch)  # 初始化一个ndarray，包含每次epoch的cost
    m = X.shape[0] #shape[0]为行数，shape[1]为列数
    #利用向量化一步求解
    for i in range(epoch):
        temp = theta - (alpha / m) * (X * theta.T - Y).T * X
        theta = temp
        cost[i] = computeCost(X,Y,theta) #得到每次迭代代价函数的值
    return theta,cost

#初始化学习率α和要进行迭代的次数
alpha = 0.01
epoch = 1000
final_theta,cost = gradientDescent(X,Y,theta,alpha,epoch)
#print(computeCost(X,Y,final_theta)) #4.5159555
#print(cost[995:])

#绘制线性模型及数据，直观看出拟合
x = np.linspace(data.Population.min(),data.Population.max(),100)
f = final_theta[0,0] + (final_theta[0,1] * x) #numpy.linspace()等差数列函数

#plt.plot(x,f,lable = 'Prediction')
#plt.plot(kind='scatter', x=data['Population'], y=data['Profit'], figsize=(8,5))
#plt.show()
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data['Population'], data.Profit, label='Traning Data')
ax.legend(loc=2)  # 2表示在左上角
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

fig,ax = plt.subplots(figsize = (8,4))
ax.plot(np.arange(epoch),cost,'r') #np.arange()返回等差数组
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

#正规方程法
def normalEqn(X, y):
    theta = np.linalg.inv(X.T*X)*X.T*y#X.T@X等价于X.T.dot(X)
    return theta.T
final_theta2 = normalEqn(X,Y)

print(final_theta,'\n',final_theta2)


"""
#多变量线性回归
path = r'C:/Users/DH/Desktop/ex1data2.txt'
data = pd.read_csv(path,names = ['sizes','rooms','prices'])
#print(data.head())

#特征缩放
data['sizes'] = (data['sizes'] - data['sizes'].mean())/data['sizes'].std()
data['rooms'] = (data['rooms'] - data['rooms'].mean())/data['rooms'].std()
data['prices'] = (data['prices'] - data['prices'].mean())/data['prices'].std()
print(data.head())

theta = np.matrix(np.zeros(3))
#插入一列，作为x0,值为1
data.insert(0,'ones',1)
#X = np.matrix(data[:2]) 仅选取前两行
#X = np.matrix([data['sizes'],data['rooms'],data['prices']])
cols = data.shape[1]
X = np.matrix(data.iloc[:,0:cols-1])
Y = np.matrix(data.iloc[:,cols-1:cols])
#代价函数
def computeCost(X,Y,theta):
    m = X.shape[0]
    return np.sum(np.power((X*theta.T) - Y,2))/(2*m)
#梯度下降
def gradientDescent(X,Y,theta,alpha,epoch):
    m = X.shape[0]
    cost = []
    for i in range(epoch):
        theta = theta - alpha/m*(X*theta.T - Y).T*X
        cost.append(computeCost(X,Y,theta))       #列表添加元素
    return theta,cost

alpha = 0.01
epoch = 1000
final_theta,cost = gradientDescent(X,Y,theta,alpha,epoch)
final_cost = computeCost(X,Y,final_theta)
#print(final_cost)

#画图查看训练过程
fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(epoch),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()

#正规方程法
def normalEqn(X, y):
    theta = np.linalg.inv(X.T*X)*X.T*y#X.T@X等价于X.T.dot(X)
    return theta.T
final_theta2 = normalEqn(X,Y)

print(final_theta,'\n',final_theta2)
"""

















