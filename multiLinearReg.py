# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 23:35:24 2018

@author: prm
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def computeCost(X,y,theta):
    m = X.shape[0];
    cost = 0;
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




data = pd.read_csv('ex1data2.txt', header=None, names=['Size', 'Bedrooms','Price'])  

data = (data - data.mean()) / data.std()  
data.head()
data.describe()
data.insert(0, 'Ones', 1)
X = data.iloc[:,0:3]
y = data.iloc[:,3]

X = np.matrix(X.values)
y = np.matrix(y.values)

m = len(y)

theta = np.matrix(np.array([0.001,0.0002,0.0003]))
print("Inintial Cost -------> ",computeCost(X,y,theta))

alpha = 0.02
iterations = 1000
g , cost = gradientDescent(X , y , theta , alpha , iterations)

computeCost(X, y, g) 


fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(np.arange(iterations ), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch') 
