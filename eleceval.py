# encoding=utf-8
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from keras.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score


# ¼ÆËãRMSE
def calcRMSE(pred, true):
    return np.sqrt(mean_squared_error(true, pred))


def calcMSE(pred, true):
    return mean_squared_error(true, pred)


# ¼ÆËãMAE
def calcMAE(pred, true):
    # pred = pred[:, 0]
    return mean_absolute_error(true, pred)


# ¼ÆËãMAPE
def calcMAPE(pred, true, epsion=0.0000000):
    # pred = pred[:,0] # ÁÐ×ªÐÐ£¬±ãÓÚ¹ã²¥¼ÆËãÎó²îÖ¸±ê
    # print (true-pred).shape
    # print true.shape
    # print pred.shape
    # true += epsion
    return np.sum(np.abs((true - pred) / true)) * 100 / len(true)
    # return mean_absolute_percentage_error(true, pred)


# ¼ÆËãSMAPE
def calcSMAPE(pred, true):
    delim = (np.abs(true) + np.abs(pred)) / 2.0
    return np.sum(np.abs((true - pred) / delim)) / len(true) * 100


def mape(predicted, test):
    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += abs(predicted[i] - test[i]) * 100.0 / test[i]
        instances += 1

    return temps / instances


def mpe(predicted, test):
    if not len(predicted) == len(test):
        print("Predicted values and output test instances do not match.")

    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += (predicted[i] - test[i]) * 100.0 / test[i]
        instances += 1

    return temps / instances


def mse(predicted, test):
    if not len(predicted) == len(test):
        print("Predicted values and output test instances do not match.")

    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += (predicted[i] - test[i]) ** 2
        instances += 1

    return temps / instances


def rmse(predicted, test):
    return (mse(predicted, test)) ** 0.5


def mae(predicted, test):
    if not len(predicted) != len(test):
        print("Predicted values and output test instances do not match.")

    temps = 0.0
    instances = 0
    for i in range(len(predicted)):
        temps += abs(predicted[i] - test[i])
        instances += 1

    return temps / instances


def r2(predicted, test):
    if not len(predicted) == len(test):
        print("Predicted values and output test instances do not match.")

    return r2_score(test, predicted)

def pinballLoss(y_true, y_pred):
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
    p=0
    q1=tf.cast(tf.linspace(0.1,0.1,1),tf.float64)
    q2=tf.cast(1-q1,tf.float64)
    r=0.01
    location=tf.less(y_true,y_pred)
    e = tf.abs(y_true - y_pred)
    
    re=tf.ones_like(e)*r
    position=tf.less(e,re)
    
    h1=tf.square(e)/(2*r)
    h2=e-r/2
    h=tf.where(position,h1,h2)
    
    p1=tf.multiply(q2,h)
    p2=tf.multiply(q1,h)
    p=tf.reduce_mean(tf.where(location,p1,p2),axis=0)
    p=tf.reduce_mean(p)
    
    return p