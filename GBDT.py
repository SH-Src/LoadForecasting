
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import ensemble
import tensorflow as tf
data = pd.read_csv('GEFC2014guiyi.csv')
data.drop(columns=['ID.1'], inplace=True)
data.set_index('ID',inplace=True)
time=data['time'].copy()
data.drop(columns='time',inplace=True)

train=data[0:3000]
train.info()
y=train['load']
X=train.drop(columns='load')
val=data[3001:4000]
y_val=val['load']
x_val=val.drop(columns='load')
test=data[4000:4760]
y_test=np.array(test['load'])
a=[]
for i in range(0,9):
    a.append(y_test)
y_test=np.array(a).transpose(1,0)
print(y_test.shape)
x_test=test.drop(columns='load')


N_ESTIMATORS = 1000
def gb_quantile(X_train, train_labels, X,q):
    gbf = ensemble.GradientBoostingRegressor(loss='quantile', alpha=q,
                                             n_estimators=N_ESTIMATORS,
                                             max_depth=3,
                                             learning_rate=0.1, min_samples_leaf=9,
                                             min_samples_split=9)
    gbf.fit(X_train, train_labels)
    return gbf.predict(X)


def pinball_score(true, pred, number_q):
    q=np.linspace(1/(number_q+1),(1-(1/(number_q+1))),number_q)
    loss = np.where(np.less(true,pred), (1-q)*(np.abs(true-pred)), q*(np.abs(true-pred)))
    return np.mean(loss)

quantile=np.arange(0.1,1,0.1)
result={}
for i in quantile:
    result[i]=gb_quantile(X,y,x_test,i)
result1=np.array(pd.DataFrame(result))
a=pinball_score(result1, y_test,9)
pd.DataFrame(result).to_csv("pinball_score="+str(a)+"%GBDT_1h.csv")
print("pinball loss: ", a)
#print(result.shape)

