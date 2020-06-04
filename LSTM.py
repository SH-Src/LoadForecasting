
# coding: utf-8

# In[5]:


import keras.backend as K
from keras import optimizers
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Activation, Lambda
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
import keras.layers

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import grid_rnn
import numpy as np
import pandas as pd
import random
import eleceval
import os
from sklearn import preprocessing
from tensorflow.python.ops import array_ops

import keras

import matplotlib.pyplot as plt

import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


predictperiod = '6h'  # 15m:15分钟，6h：6小时，1d：天
modeltype = 'TCN'  # LSTM,GRU,pLSTM,gridLSTM

summary_dir = "/media/dzf/data/data/MSFNN_train_result/0_test/" + modeltype + predictperiod
MODEL_SAVE_PATH = "/media/dzf/data/data/MSFNN_train_result/0_test/" + modeltype + predictperiod
MODEL_NAME = "model.ckpt"

# learning_rate = 0.001
training_iters = 40001
batch_size = 240
regularization_rate = 0.0001

data_size = 5000
train_data = int(data_size * 0.6)
val_data = int(data_size * 0.2)

n_input = 5
# 用前10个数据预测下一个,第batch_size个数据，n_step个为一组，一个n_input个特征
n_steps = 240
n_hidden = 128
n_class = 1
n_layers = 18
num_epochs = 5000
data_path = "GEFC2014guiyi.csv"
period_step = 24
period_day = 7

'''
for z in range(1000):
    summary_dir = summary_dir+'_'+('%d'%z)
    MODEL_SAVE_PATH = MODEL_SAVE_PATH+'_'+('%d'%z)
    if os.path.exists(summary_dir) or os.path.exists(MODEL_SAVE_PATH):
        continue        
    os.makedirs(summary_dir)
    os.makedirs(MODEL_SAVE_PATH)
    break
'''


def dofile(filename, datasize):
    df = pd.read_csv(filename)
    X = [];
    Y = []
    for i in range(datasize - n_steps):
        x = df.loc[i:i + n_steps - 1, ['load']].values.tolist()
        y = df.loc[i + 1:i + n_steps, ['load']].values.tolist()
        X.append(x)
        yy=y*9
        Y.append(yy)
    return X, Y


def pinball_loss(y_true, y_pred):
    '''
    r 邻域范围
    location 记录真实值和预测值相比那个地方大，那个地方小
    re 邻域的张量
    e 真实值与预测值差距的绝对值
    position 真实值与预测值差距的绝对值 与 邻域 的大小比较
    h1 l2规范
    h2 l1规范
    h标准规范化后的数据
    p损失值
    '''
    print('_true;')
    print(y_true.shape)
    print(y_pred.shape)
    p = 0
    q1 = tf.linspace(0.1, 0.9, 9)
    q2 = 1 - q1
    r = 0.01
    location = tf.less(y_true, y_pred)
    e = tf.abs(y_true - y_pred)

    re = tf.ones_like(e) * r
    position = tf.less(e, re)

    h1 = tf.square(e) / (2 * r)
    h2 = e - r / 2
    h = tf.where(position, h1, h2)

    p1 = tf.multiply(q2, h)
    p2 = tf.multiply(q1, h)
    p = tf.reduce_mean(tf.where(location, p1, p2), axis=0)
    p = tf.reduce_mean(p)

    return p





def build_model(max_len, num_feat):
    input_layer = Input(name='input_layer', shape=(max_len, num_feat))
    x = input_layer
    x = LSTM(128, return_sequences=True)(x)
    #x = LSTM(128, return_sequences=True)(x)
    #x = LSTM(128, return_sequences=True)(x)
    print('1.x.shape=', x.shape)
    output_layer = Dense(9, kernel_regularizer=keras.regularizers.l2(0.0001))(x)

    print('2.output_layer.shape=', output_layer.shape)
    model = Model(input_layer, output_layer)
    adam = optimizers.Adam(lr=0.05, clipnorm=1., decay=0.1, amsgrad=True)
    model.compile(adam, loss=pinball_loss)
    return model


# 载入训练数据
x_raw,y_raw = dofile(data_path, data_size)

xtrain = x_raw[0:train_data]
ytrain = y_raw[0:train_data]
x_train=np.array(xtrain)
y_train=np.array(ytrain).reshape(3000,9,240).transpose(0,2,1)

xval = x_raw[train_data:train_data+val_data]
yval = y_raw[train_data:train_data+val_data]
x_val=np.array(xval)
y_val=np.array(yval).reshape(1000,9,240).transpose(0,2,1)

xtest = x_raw[train_data+val_data:data_size-n_steps]
ytest = y_raw[train_data+val_data:data_size-n_steps]
x_test=np.array(xtest)
y_test=np.array(ytest)
print(f'y_test.shape = {y_test.shape}')
y_test = y_test.reshape(760,9,240).transpose(0,2,1)
y_test=y_test[:,239,:]



def pinball_score(true, pred, number_q):
    q=np.linspace(1/(number_q+1),(1-(1/(number_q+1))),number_q)
    loss = np.where(np.less(true,pred), (1-q)*(np.abs(true-pred)), q*(np.abs(true-pred)))
    return np.mean(loss)

class PrintSomeValues(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.pinball_flag = 100.0

    def on_epoch_begin(self, epoch, logs={}):
        # print(f'x_test[0:1] = {x_test[0:1]}.')
        # print(f'y_test[0:1] = {y_test[0:1]}.')
        # print(f'pred = {self.model.predict(x_test[0:1])}.')
        lr = K.get_value(model.optimizer.lr)
        print("current learning rate is {}".format(lr))
        pred = model.predict(x_test)
        pred = pred[:,239,:]

        print('pred.shape:')
        print(pred.shape)
        print('y_test.shape:')
        print(y_test.shape)

        predict_all = pred.flatten()
        truth_all = y_test.flatten()
        '''
        pred_train = model.predict(x_train)
        predict_train = pred_train.flatten()
        truth_train = y_train.flatten()
        mape_train = eleceval.calcMAPE(predict_train,truth_train)
        mae_train = eleceval.calcMAE(predict_train,truth_train) 
        mse_train = eleceval.calcMSE(predict_train,truth_train)
        rmse_train = eleceval.calcRMSE(predict_train,truth_train) 
        r_2_train = eleceval.r2(predict_train,truth_train)        
        print("After %d training step(s),"
              "on test data MAPE_train = %.4f,MAE_train = %.4f,MSE_train = %.4f,RMSE_train = %.4f,R2_train = %.4f"\
              % (i*10, mape_train,mae_train,mse_train,rmse_train,r_2_train))
        '''
        a1=pred
        a2=y_test
        a = pinball_score(a1,a2,9)
        mape = eleceval.calcMAPE(predict_all,truth_all)
        mae = eleceval.calcMAE(predict_all, truth_all)
        mse = eleceval.calcMSE(predict_all, truth_all)
        rmse = eleceval.calcRMSE(predict_all, truth_all)
        r_2 = eleceval.r2(predict_all, truth_all)
        print("After %d training step(s),"
              "on test data MAPE = %.4f,MAE = %.4f,MSE = %.4f,RMSE = %.4f,R2 = %.4f,Pinball loss = %.4f"\
              % (epoch, mape, mae, mse, rmse, r_2, a))
        
        
        if a <= self.pinball_flag:
            self.pinball_flag = a
            # 两个ndarray列合并
            # y_con = np.concatenate((truth_all, predict_all), axis=1)
            truth_all_reshape = np.reshape(y_test[:, 0], [-1, 1])
            predict_all_reshape = pred
            y_con = np.concatenate((truth_all_reshape, predict_all_reshape), axis=1)
            # 输出真实值和预测值0
            y_out = pd.DataFrame(y_con)
            y_out.to_csv('result1h/steps=%d-pinball=%.4f.csv' % (epoch, a))
         

# def run_task():

model = build_model(max_len=x_train.shape[1], num_feat=x_train.shape[2])

print(f'x_train.shape = {x_train.shape}')
print(f'y_train.shape = {y_train.shape}')

#psv = PrintSomeValues()
psv = PrintSomeValues()
# Using sparse softmax.
# http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
model.summary()

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, mode='min')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=2, mode='min')

# for i in range(1000):
# callbacks=[psv]

model.fit(x_train, y_train,
          validation_data=(x_val, y_val),
          epochs=5000,
          batch_size=100,
          initial_epoch=0,
          callbacks=[early_stopping, reduce_lr, psv]
          )
print(result)
# if __name__ == '__main__':
#    run_task()

