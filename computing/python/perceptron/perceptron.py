#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 15 21:35:39 2018
NN with backprop from scratch
@pewilliams
"""
import pandas as pd
import numpy as np

#needs:
#add bias term
#add regularization term
#set up epsilon for termination, or max iteration crit

#activation function
def sigmoid(z):
    return 1.0/(1+ np.exp(-z))

def sigmoid_derivative(z):
    return z * (1.0 - z)

#outcome setup
n = 100 #nrecs
X = pd.DataFrame({'x1' : np.random.normal(0,1,n), 'x2' : np.random.choice(6,n), 'x3': np.random.choice(1,n)})
y = 1 + 0.5*X['x1'] + 0.25*X['x2'] + 0.1*X['x3'] + np.random.normal(0,1,n)
y = (y - min(y))/(max(y) - min(y)) #normalize
y = np.array([np.array([x]) for x in y]) #nested array for convenience

#set up - 4 neurons - 1 hidden layers
neurons = 4
w1 = np.random.rand(X.shape[1],neurons) 
w2 = np.random.rand(neurons,1)              
#first pass
output = np.zeros(y.shape)
layer1 = sigmoid(np.dot(X, w1))
output = sigmoid(np.dot(layer1, w2))
cost = sum((y - output)**2)
num_iters = 200
rate = 0.01
for i in range(num_iters): 
    #calc sweep of derivatives - backprop   
    d_w2 = np.dot(layer1.T, ((y - output) * sigmoid_derivative(output)) )
    d_w1 = np.dot(X.T, (np.dot((y - output) * sigmoid_derivative(output), w2.T)*sigmoid_derivative(layer1)))
    # update the weights with the derivative (slope) of the loss function
    w1 += rate * d_w1
    w2 += rate * d_w2         
    layer1 = sigmoid(np.dot(X, w1))
    output = sigmoid(np.dot(layer1, w2))
    error = sum((y - output)**2)
    #adaptive
    if error < cost: 
        rate = rate * 1.05
    else: 
        rate = rate * 0.5
    cost = error
    print('Iter:',i, 'Loss:',sum((y - output)**2))
    

#set up - 4 neurons - 2 hidden layers
neurons = 4
w1 = np.random.rand(X.shape[1],neurons) 
w2 = np.random.rand(neurons,neurons)          
w3 = np.random.rand(neurons,1)       
#first pass
output = np.zeros(y.shape)
layer1 = sigmoid(np.dot(X, w1))
layer2 = sigmoid(np.dot(layer1, w2))
output = sigmoid(np.dot(layer2, w3))

cost = sum((y - output)**2)
num_iters = 200
rate = 0.01
for i in range(num_iters): 
    #calc sweep of derivatives - backprop   
    d_w3 = np.dot(layer2.T, ((y - output) * sigmoid_derivative(output)) )
    d_w2 = np.dot(layer1.T, (np.dot((y - output) * sigmoid_derivative(output), w3.T)*sigmoid_derivative(layer2)))
    d_w1 = np.dot(X.T, (np.dot((y - output) * sigmoid_derivative(output), w3.T)*sigmoid_derivative(layer1)))
    # update the weights with the derivative (slope) of the loss function
    w1 += rate * d_w1
    w2 += rate * d_w2
    w3 += rate * d_w3             
    layer1 = sigmoid(np.dot(X, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    output = sigmoid(np.dot(layer2, w3))
    error = sum((y - output)**2)
    #adaptive
    if error < cost: 
        rate = rate * 1.05
    else: 
        rate = rate * 0.5
    cost = error
    print('Iter:',i, 'Loss:',sum((y - output)**2))
    
 #src: https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6   
    