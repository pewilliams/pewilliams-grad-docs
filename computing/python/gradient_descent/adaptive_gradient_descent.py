#gradient descent
#mv regression case

import pandas as pd
import numpy as np
from scipy import optimize

# setup
n = 1000 #nrecs
X = pd.DataFrame({'x0' : np.repeat(1, n, axis = 0), 'x1' : np.random.normal(0,1,n), 'x2' : np.random.choice(6,n)})
y = 10 + 0.5*X['x1'] + 0.25*X['x2'] + np.random.normal(0,0.5,n)


# syntax sucks
ols = np.dot(np.linalg.inv(np.dot(X.transpose(),X)), np.dot(X.transpose(),y))
yhat = np.dot(X, ols)

mse = sum((y - yhat)**2)

alpha = 0.001
eps = 0.0001
maxiter = 100

#grad descent - basic regression case
def gradientDescentLR(maxiter, X, y, rate, eps):
 beta = np.repeat(0,X.shape[1]) #initialize betas
 cost = 1/len(y) * sum( (y - np.dot(X, beta))**2 )
 converged = False
 niter = 1
 while converged == False: 
 	gradient = 1./len(y) * np.dot(X.transpose(), np.dot(X, beta) - y  )
 	betaUpdate = beta - rate * gradient
 	beta = betaUpdate
 	error = 1/len(y) * sum( (np.dot(X, beta) - y)**2 )
 	
 	if abs(cost - error) <= eps: 
 		converged = True	
 	
 	elif niter == maxiter: 
 		converged = True
 	
 	else: 
 		cost = error
 		niter += 1
 
 return {'beta': beta, 'mse' : cost, 'iter': niter, 'learnRate': alpha}	
 

print('Gradient Descent estimator: ')
print(gradientDescentLR(maxiter = 10000, X = X , y = y, rate = 0.01, eps= 0.0001))
print('OLS estimator: ')
print(ols)

#need to add adaptive component

def gradientDescentAdaptive(maxiter,X,y,rate,eps):
    beta = np.repeat(0,X.shape[1])
    cost = 1./len(y) * sum((y - np.dot(X,beta))**2)
    converged = False
    niter = 1
    while converged == False:
        gradient = 1./len(y) * np.dot(X.transpose(), np.dot(X,beta) - y)
        beta_update = beta - rate * gradient
        beta = beta_update
        error = 1./len(y) * sum((y - np.dot(X,beta))**2)
        
        #adaptive
        if error < cost: 
            rate = rate * 1.05
        else: 
            rate = rate * 0.5
        
        if abs(cost - error) <= eps:
            converged = True
            
        elif niter == maxiter:
            converged = True
        
        else:
            cost = error
            niter += 1
            
    return {'beta': beta, 'mse' : cost, 'iter': niter, 'learnRate': rate}

print(gradientDescentAdaptive(maxiter = 100000, X = X , y = y, rate = 0.01, eps= 0.0001))

#
def gradientDescentAdaptiveRidge(maxiter,X,y,alpha,rate,eps):
    beta = np.repeat(0,X.shape[1])
    cost = 1./len(y) * sum((y - np.dot(X,beta))**2 + alpha*np.dot(beta,beta))
    converged = False
    niter = 1
    while converged == False:
        gradient = 1./len(y) * (np.dot(X.transpose(), np.dot(X,beta) - y) + alpha*beta)
        beta_update = beta - rate * gradient
        beta = beta_update
        error = 1./len(y) * sum((y - np.dot(X,beta))**2 + alpha*np.dot(beta,beta))
        
        #adaptive
        if error < cost: 
            rate = rate * 1.05
        else: 
            rate = rate * 0.5
        
        if abs(cost - error) <= eps:
            converged = True
            
        elif niter == maxiter:
            converged = True
        
        else:
            cost = error
            niter += 1
            
    return {'beta': beta, 'mse' : cost, 'iter': niter, 'learnRate': rate}

print(gradientDescentAdaptiveRidge(maxiter = 10000, X = X , y = y, alpha = 0.5, rate = 0.01,eps= 0.000001))

#bfgs gen purpose

def costFun(beta):
    return(1./len(y) * sum((y - np.dot(X,beta))**2))
    
def gradFun(beta):
    return((1./len(y) * np.dot(X.transpose(), np.dot(X, beta) - y )))

beta = np.zeros(X.shape[1])
optimize.minimize(costFun,beta ,method = 'BFGS', jac = gradFun, options = {'disp':True})


alpha = 0
def costFunRidge(beta):
    return(1./len(y) * sum((y - np.dot(X,beta))**2 + alpha*np.dot(beta,beta)))
    
def gradFunRidge(beta):
    return((1./len(y) * (np.dot(X.transpose(), np.dot(X,beta) - y) + alpha*beta)))

beta = np.zeros(X.shape[1])
optimize.minimize(costFunRidge,beta ,method = 'BFGS', jac = gradFunRidge, options = {'disp':True})

    








            
     
        
 
