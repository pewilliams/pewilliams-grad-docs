#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 16:32:01 2018
Neural network implementation from scratch - arbitrary number of hidden layers
@author: pewilliams
"""

import pandas as pd
import numpy as np

# setup - arbitrary inputs
n = 1000 #nrecs
X = pd.DataFrame({'x1' : np.random.normal(0,1,n), 'x2' : np.random.choice(6,n)})
y = 1 + 0.5*X['x1'] + 0.25*X['x2'] + np.random.normal(0,0.5,n)

#Normalize y -> [0,1]
y = (y - min(y))/(max(y) - min(y))

#weight matrix and bias initialize
number_layers = 3
regularization_term = 0.001
#weight matrix input layer
W1 = np.matrix(abs(np.random.normal(0,0.25,number_layers*X.shape[1])).reshape(number_layers,X.shape[1]))
#weight matrix activation layer
W2 = np.array(abs(np.random.normal(0,0.25,number_layers)))
#bias terms
bias = np.array(abs(np.random.normal(0,0.25,number_layers + 1)))

def activation_function(z):
    return(1./(1 + np.exp(-z)))

def predict_nn(x):
    activation_layer = np.zeros(number_layers)
    for i in np.arange(number_layers): 
       z = np.dot(W1[i],x) + bias[i]
       activation_layer[i] = activation_function(z)
    return( activation_function( np.dot(W2,activation_layer) + bias[number_layers])  )
    
#cost function
def costFun(X,y, W1, W2, bias, regularization_term):
    return(1./len(y) * sum(np.apply_along_axis(predict_nn, 1, X) - y)**2 ) + (regularization_term/2)*( np.square(W1).sum() + np.square(W2).sum())

costFun(X=X,y=y, W1= W1,W2 = W2,bias = bias, regularization_term = 0.01)

#need adaptive stochastic gradient descent - first without backpropagation?    
def gradientDescentAdaptive(maxiter,X,y,regularization_term, rate, eps, number_layers):
    #weight matrix input layer W_1
    #weight matrix activation layer W_2, and
    #bias terms
    W1 = np.matrix(abs(np.random.normal(0,0.25,number_layers*X.shape[1])).reshape(number_layers,X.shape[1]))
    W2 = np.array(abs(np.random.normal(0,0.25,number_layers)))
    bias = np.zeros(number_layers + 1) #np.array(abs(np.random.normal(0,0.25,number_layers + 1)))
    
    #initial cost
    cost = costFun(X = X,y=y, W1 = W1,W2 = W2, bias = bias, regularization_term = regularization_term)
    converged = False
    niter = 1
    
    #feed forward pass
    z = []
    a = []
    for i in np.arange(number_layers): 
       z_i = np.dot(W1[i],X.transpose()) + bias[i]
       z.append(z_i)
       a_i = activation_function(z_i)
       a.append(a_i)
    
    a_out = []
    for i in np.arange(number_layers): 
        a_out.append(W2[i]*a[i])
    a_out = activation_function(sum(a_out) + bias[number_layers])
    
    delta_out = -1*  np.array((np.array(y) - a_out)) * np.array(activation_function(z[2])) * np.array(1 - activation_function(z[2])) 
    
    
    for l in np.arange(0,number_layers): 
        z_l = np.dot(W1[l],X.transpose()) + bias[l]
        a_l = activation_function(z_l)
        print(a_l)

    #need to fix this
    while converged == False: 
        
        #try with delta rule
        #grad(w_ij) = rate*(actual_j - pred_j)*activation_function(z_j)(1 - activation_function(z_j))*x_i

        W1_update = W1
        #bias_update = bias
        h_total = []
        for l in np.arange(0,number_layers):  
            #weight matrix
            z = np.dot(W1[l],X.transpose()) + bias[l]
            h = activation_function(z)
            W1_update[l] = W1[l] - rate * 1./len(y) * np.dot( np.array((np.array(y) - h)) * np.array(h) * np.array(1-h)  , X)
            #bias update
        W1 = W1_update
        #W2 update
        W2_update = W2 - rate * activation_function() * (1 - activation_function())  #np.dot(X.transpose(), np.apply_along_axis(predict_nn, 1, X) - y  )
        W2 = W2_update        
        error = costFun(X,y,W1,W2, regularization_term)       
        
        #adaptive
        if error < cost: 
            rate = rate * 1.1
        else: 
            rate = rate * 0.5
        #stop criteria
        if abs(cost - error) <= eps:
            converged = True
            
        elif niter == maxiter:
            converged = True
        else:
            cost = error
            niter += 1
            
    return {'W1': W1,'W2':W2 ,'cost_res' : cost, 'iter': niter, 'learning_rate': rate,'regularization_term': regularization_term}



print(gradientDescentAdaptive(maxiter = 100, X = X , y = y, regularization_term = 0.01, rate = 0.01, eps= 0.000001, number_layers =2))





