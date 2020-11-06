# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:26:45 2020

@author: HP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris=pd.read_csv('C:/Users/HP/Desktop/python prgrmg/KNN/iris.csv')

#split train and test data
from sklearn.model_selection import train_test_split
train,test=train_test_split(iris,test_size=0.2,random_state=0)

#KNN
from sklearn.neighbors import KNeighborsClassifier as KNC
#model building for k=3
neigh=KNC(n_neighbors=3)
neigh.fit(train.iloc[:,0:4],train.iloc[:,4])
train_predict=neigh.predict(train.iloc[:,0:4])
pd.crosstab(train_predict,train.iloc[:,4])
train_acc=(39+34+41)/(39+34+41+3+3)
train_acc
#or another way for calc accuracy
train_acc=np.mean(neigh.predict(train.iloc[:,0:4])==train.iloc[:,4])
train_acc

test_acc=np.mean(neigh.predict(test.iloc[:,0:4])==test.iloc[:,4])
test_acc

#to find best k value for building best model
acc=[]
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:4],train.iloc[:,4])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:4])==train.iloc[:,4])
    test_acc=np.mean(neigh.predict(test.iloc[:,0:4])==test.iloc[:,4])
    acc.append([train_acc,test_acc])
    
plt.plot(np.arange(3,50,2),[i[0] for i in acc],'bo-')
plt.plot(np.arange(3,50,2),[i[1] for i in acc],'ro-')
plt.legend(['train','test'])


#from plots at k=8 we get best model
#model building at k=8
neigh8=KNC(n_neighbors=8)
neigh8.fit(train.iloc[:,0:4],train.iloc[:,4])

train_acc8=np.mean(neigh8.predict(train.iloc[:,0:4])==train.iloc[:,4])
train_acc8
test_acc8=np.mean(neigh8.predict(test.iloc[:,0:4])==test.iloc[:,4])
test_acc8
