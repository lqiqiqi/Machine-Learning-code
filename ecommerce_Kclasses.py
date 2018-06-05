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

def get_data():
    df=pd.read_csv('ann_logistic_extra\\ecommerce_data.csv')
    data=df.as_matrix()
    
    Xtrain=data[:-200,:-1]
    Xtest=data[-200:,:-1]
    Ytrain=data[:-200,-1:]
    Ytest=data[-200:,-1:]
    
    
    #normalization
    Xtrain[:,1]=(Xtrain[:,1]-Xtrain[:,1].mean())/Xtrain[:,1].std()
    Xtrain[:,2]=(Xtrain[:,2]-Xtrain[:,2].mean())/Xtrain[:,2].std()
    Xtest[:,1]=(Xtest[:,1]-Xtest[:,1].mean())/Xtest[:,1].std()
    Xtest[:,2]=(Xtest[:,2]-Xtest[:,2].mean())/Xtest[:,2].std()
    
    N,D=Xtrain.shape
    Xtrain2=np.zeros((N,D+3))
    Xtrain2[:,:4]=Xtrain[:,:4]
    Ntest,Dtest=Xtest.shape
    Xtest2=np.zeros((Ntest,Dtest+3))
    Xtest2[:,:4]=Xtest[:,:4]


    #make time of day into four features/ catergories
    for n in range(N):
        t=int(Xtrain[n,4]) #make them into integer so can be used as indexes
        Xtrain2[n,t+4]=1
    for n in range(Ntest):
        t=int(Xtest[n,4]) #make them into integer so can be used as indexes
        Xtest2[n,t+4]=1


    Z=np.zeros((N,4))
    Z[range(N),Xtrain[:,D-1].astype(np.int32)]=1
    #it is not only assignment. it should be understood in two axis
    
    assert(np.abs(Z-Xtrain2[:,-4:]).sum()<10e-10)
    #use another method to caculate and check consistent

    return Xtrain2, Xtest2, Ytrain,Ytest
#
#def get_binary_data():
#    Xtrain,Ytrain=get_data()
#    Xtrain2=Xtrain[Ytrain<=1]
#    Ytrain=Ytrain[Ytrain<=1]
#    return Xtrain2,Ytrain



def sigmoid(Xtrain,W,b):
    z=1/(1+np.exp(-(Xtrain.dot(W)+b)))
    return z

def calculate_z(Xtrain,W,b):
    z=sigmoid(Xtrain,W,b)
    return z

def softmax(a):
    p=np.exp(a)/np.exp(a).sum(axis=1,keepdims=True)
    #keep 2d dimensions. only calculate the sum for each row
    return p

def calculate_p(z,V,c):
    p=z.dot(V)+c
    p=softmax(p)
    return p


def forward(Xtrain,W,b,V,c):
   N=len(Xtrain)
   z=calculate_z(Xtrain,W,b)
   p=calculate_p(z,V,c)
   prediction=np.argmax(p,axis=1).reshape((N,1))
   return p,z,prediction
#find the index of max value along axis 1

def predict_scores(Ytrain,prediction):
    return np.mean(Ytrain==prediction)

#testing
def y2index(Y,N,K):
    Y1=np.zeros((N,K))
    for n in range(N):
        t=int(Y[n])
        Y1[n,t]=1
    return Y1

#comupte error item
def delta_end(Y1,p):   #NOTICE: function name cannot be the same with var name
   delta=Y1-p
   return delta

def delta_hid(deltaend,V,z):
    delta=deltaend.dot(V.T)*z*(1-z)
    return delta

def main(M,alpha,iteration): #hidden units
    Xtrain,Xtest,Ytrain,Ytest =get_data()
    
        
    N=len(Xtrain)
    Ntest=len(Xtest)
    D=Xtrain.shape[1]
    K=len(np.unique(Ytrain))   #unique elements
    Ytrain1=y2index(Ytrain,N,K)
    Ytest1=y2index(Ytest,Ntest,K)
    
    np.random.seed(1)
    W=np.random.randn(D*M).reshape((D,M))
    V=np.random.randn(M*K).reshape((M,K))
    b=np.zeros(M)
    c=np.zeros(K)

    #gradient desent
    coststrain=[]# cannot put into for loop or it will clear itself every loop
    coststest=[]
    
    for i in range(iteration):
        p,z,prediction=forward(Xtrain,W,b,V,c)
        
        deltaend=delta_end(Ytrain1,p)
        V+=alpha*z.T.dot(deltaend)
        c+=alpha*deltaend.sum(axis=0)
        #warning: += instead of =+. Or every value will not change 
        
        deltahid=delta_hid(deltaend,V,z)
        W+=alpha*Xtrain.T.dot(deltahid)
        b+=alpha*deltahid.sum(axis=0)
        
        ptest,ztest,predictiontest=forward(Xtest,W,b,V,c)
        
        costtrain=Ytrain1*np.log(p)
        #notice that it is Y1 which match p instead of Ytrain
        #if use Ytrain and prediction it will appear lots of nan because log function
        #and a lot of zeros in prediction
        costtrain=costtrain.sum()
        #if cost= Ytrain*np.log(p).sum() then will return an array
        costtest=Ytest1*np.log(ptest)
        costtest=costtest.sum()
        
        
        coststrain.append(costtrain)
        coststest.append(costtest)
        
        scores=predict_scores(Ytrain,prediction)
        if i%100==0:
            print("scores:",scores)
            print("costfortrain:",costtrain)
            
    
    line1,=plt.plot(coststrain,label='train')
    line2,=plt.plot(coststest,label='test')
    plt.legend(loc='best')
    plt.show()

if __name__=='__main__':
    main(5,0.001,5000)
