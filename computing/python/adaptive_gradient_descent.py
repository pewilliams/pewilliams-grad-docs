#gradient descent
#mv regression case

import pandas as pd
import numpy as np

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
def gradientDescentLR(maxiter, X, y, alpha, eps):
 beta = np.repeat(0,X.shape[1]) #initialize betas
 cost = sum( (y - np.dot(X, beta))**2 )
 converged = False
 niter = 1
 while converged == False: 
 	gradient = 1./len(y) * np.dot(X.transpose(), np.dot(X, beta) - y  )
 	betaUpdate = beta - alpha * gradient
 	beta = betaUpdate
 	error = sum( (np.dot(X, beta) - y)**2 )
 	
 	if abs(cost - error) <= eps: 
 		converged = True	
 	
 	elif niter == maxiter: 
 		converged = True
 	
 	else: 
 		cost = error
 		niter += 1
 
 return {'beta': beta, 'mse' : cost, 'iter': niter, 'learnRate': alpha}	
 

print('Gradient Descent estimator: ')
print(gradientDescentLR(maxiter = 10000, X = X , y = y, alpha = 0.01, eps= 0.0001))
print('OLS estimator: ')
print(ols)

#need to add adaptive component
 
