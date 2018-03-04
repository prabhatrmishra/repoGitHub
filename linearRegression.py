# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:37:36 2018

@author: prm
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

def computeCost(X,y,theta):
    m = X.shape[0];
    cost = 0;
    print ("the number of rows --",m)
    h = X * theta.T
    inner = np.power((h - y),2);
    cost = np.sum(inner)/ (2 * m)
    return cost


def gradientDescent(X , y , theta , alpha , iterations):
    m = len(y)
    temp_theta = np.matrix(np.zeros((theta.shape)))
    params = X.shape[1]
    cost = np.zeros(iterations)
    for i in range(iterations):   
        error = (X * theta.T) - y 
        for j in range (params):
            terms = np.multiply(error, X[:,j])
            #print(terms);
            temp_theta[0,j] = theta[0,j] - ((alpha / m) * (np.sum(terms)))
            print(temp_theta)
        theta = temp_theta 
        cost[i] = computeCost(X , y, theta )
        print("Cost after this interation",cost[i]);
    return theta ,cost



   
def predictValue(g ,x):
    return (g[0,0] + (g[0, 1] * x))
       
        
#path = os.getcwd() + 'ex1data1.txt'  

data = pd.read_csv('ex1data1.txt', header=None, names=['Population', 'Profit'])  
data.head()

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8)) 


# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)   
cols = data.shape[1]# shape[0] to get no of rows and shape[1] to get no of cols

# set X (training data) and y (target variable)

X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols];
X = np.matrix(X.values)  
y = np.matrix(y.values) 

theta = np.matrix(np.array([0.2,1.1]));
iterations = 1500
alpha = 0.01

initial_cost = computeCost(X,y,theta)

print("This is the initial cost :- ", initial_cost)

g , cost = gradientDescent(X , y , theta , alpha , iterations)



x = np.linspace(data.Population.min(), data.Population.max(), 100)  
f = g[0, 0] + (g[0, 1] * x)
fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x, f, 'r', label='Prediction')  
ax.scatter(data.Population, data.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size') 



fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')  

print( predictValue(g , 23))