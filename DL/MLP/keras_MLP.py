
#-*- coding: utf-8 -*-

import pandas as pd
from random import shuffle
import numpy as np

datafile = 'model.xls'
data = pd.read_excel(datafile)
data = data.as_matrix()
shuffle(data)

p = 0.8 #设置训练数据比例
train = data[:int(len(data)*p),:]
test = data[int(len(data)*p):,:]

#构建LM神经网络模型
from keras.models import Sequential #导入神经网络初始化函数
from keras.layers.core import Dense, Activation,Dropout #导入神经网络层函数、激活函数


net = Sequential() #建立神经网络
net.add(Dense(input_dim = 3, output_dim = 10, activation="relu")) #添加输入层（3节点）到隐藏层（10节点）的连接
net.add(Dropout(0.3)) #隐藏层使用relu激活函数
net.add(Dense(input_dim = 10, output_dim = 1, activation="sigmoid")) #添加隐藏层（10节点）到输出层（1节点）的连接
net.add(Dropout(0.2)) #输出层使用sigmoid激活函数
net.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics=["accuracy"]) #编译模型，使用adam方法求解

result = net.fit(train[:,:3], train[:,3], epochs=int(len(data)/10), batch_size=10, validation_split=0.2) #训练模型，循环1000次

predict_result = net.predict((test[:, :3]))

print(predict_result)

print("test_accuracy:", np.mean(result.history['val_acc']))

