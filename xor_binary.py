# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 11:00:54 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 30 15:46:39 2018

@author: Administrator
"""
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
os.chdir('D:\\Documents\\GitHub\\machine_learning_examples')

 
#
#def get_binary_data():
#    Xtrain,Ytrain=get_data()
#    Xtrain2=Xtrain[Ytrain<=1]
#    Ytrain=Ytrain[Ytrain<=1]
#    return Xtrain2,Ytrain

Xtrain=np.array([[1,0],[0,1],[0,0],[1,1],[0,1],[1,1],[0,0],[1,0]])
Xtest=np.array([[1,0],[1,1],[0,1],[0,1]])
Ytrain=np.array([[1],[1],[0],[0],[1],[0],[0],[1]])
Ytest=np.array([[1],[0],[1],[1]])

M=5      #hidden units
N=len(Xtrain)
Ntest=len(Xtest)
D=Xtrain.shape[1]
Nout=1   #unique elements

np.random.seed(1)
W=np.random.randn(D*M).reshape((D,M))
V=np.random.randn(M*Nout).reshape((M,Nout))
b=np.zeros(M)
c=0

def sigmoid(Xtrain,W,b):
    z=1/(1+np.exp(-(Xtrain.dot(W)+b)))
    return z

def calculate_z(Xtrain,W,b):
    z=sigmoid(Xtrain,W,b)
    return z

def calculate_p(z,V,c):
    p=sigmoid(z,V,c)
    return p


def forward(Xtrain,W,b,V,c):
   N=len(Xtrain)
   z=calculate_z(Xtrain,W,b)
   p=calculate_p(z,V,c)
   return p,z
#find the index of max value along axis 1

def predict_scores(Ytrain,p):
    return np.mean(Ytrain==p)


#comupte error item
def delta_end(Y,p):   #NOTICE: function name cannot be the same with var name
   delta=Y-p
   return delta

def delta_hid(deltaend,V,z):
    delta=deltaend.dot(V.T)*z*(1-z)
    return delta

#gradient desent
alpha=0.01;iteration=10000
coststrain=[]# cannot put into for loop or it will clear itself every loop
coststest=[]

for i in range(iteration):
    p,z=forward(Xtrain,W,b,V,c)
    
    deltaend=delta_end(Ytrain,p)
    V+=alpha*z.T.dot(deltaend)
    c+=alpha*deltaend.sum(axis=0)
    #warning: += instead of =+. Or every value will not change 
    
    deltahid=delta_hid(deltaend,V,z)
    W+=alpha*Xtrain.T.dot(deltahid)
    b+=alpha*deltahid.sum(axis=0)
    
    ptest,ztest=forward(Xtest,W,b,V,c)
    
    costtrain=Ytrain*np.log(p)+(1-Ytrain)*np.log(1-p)
    #notice that it is Y1 which match p instead of Ytrain
    #if use Ytrain and prediction it will appear lots of nan because log function
    #and a lot of zeros in prediction
    costtrain=costtrain.sum()
    #if cost= Ytrain*np.log(p).sum() then will return an array
    costtest=Ytest*np.log(ptest)+(1-Ytest)*np.log(1-ptest)
    costtest=costtest.sum()
    
    
    coststrain.append(costtrain)
    coststest.append(costtest)
    
    scores=predict_scores(Ytrain,p)
    if i%100==0:
        print("scores:",scores)
        print("costfortrain:",costtrain)
        

line1,=plt.plot(coststrain,label='train')
line2,=plt.plot(coststest,label='test')
plt.legend(loc='best')
plt.show()