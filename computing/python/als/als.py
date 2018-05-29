#assumption of explicity ratings...
# utf-8
import pandas as pd
import numpy as np
import os
os.getcwd()

ratingsNames = ["userId", "movieId", "rating", "timestamp"]
ratings = pd.read_table("~/Projects/pewilliams-grad-docs/computing/python/als/movielens/u.data", header=None, sep="\t", names=ratingsNames)

usersNames = ["userId", "gender", "age", "occupation", "zipCode"]
users = pd.read_table("~/Projects/pewilliams-grad-docs/computing/python/als/movielens/u.user", header=None, sep="|", names=usersNames)

#
#moviesNames = ["movieId", "title", "genres"]
#movies = pd.read_table("/Users/pewilliams/Desktop/als/movielens/u.item", header=None, sep="|")

#input parameters
f = 10 #number of latent factors - inner dimension of product
regLamba = 0.1 #regularization parameter
iters = 10 #als iterations
n = len(ratings.movieId.unique()) #number of unique movies
m = len(ratings.userId.unique()) #number of unique users


def normaliseRow(x):
    return x / sum(x)

def initialiseMatrix(n, f):
    A = abs(np.random.randn(n, f))
    return np.apply_along_axis(normaliseRow, 1, A)

# Initialise Y matrix, n x f
Y = initialiseMatrix(n, f) # item decomp matrix
# Initialise X matrix, m x f
X = initialiseMatrix(m, f) #user decomp matrix


# Create a dummy entry for each movie
temp = np.zeros((n, 4))
for i in range(1, n):
    temp[i,] = [m+1, i, 0, 0]
    
ratings = ratings.append(pd.DataFrame(temp, columns =ratingsNames))

#Input matrix
ratingsMatrix = ratings.pivot_table(columns=['movieId'], index =['userId'], values='rating', dropna = False)
ratingsMatrix = ratingsMatrix.fillna(0).as_matrix()

# Drop the dummy movie
ratingsMatrix = ratingsMatrix[0:m,0:n]

def ratingsPred(X, Y):
    return np.dot(X, Y.T)

def MSE(ratingsPred, ratingsMatrix):
    idx = ratingsMatrix > 0
    return sum((ratingsPred[idx] - ratingsMatrix[idx]) ** 2) / np.count_nonzero(ratingsMatrix)
    
print(MSE(ratingsPred(X, Y), ratingsMatrix))

nonZero = ratingsMatrix > 0

#lambda diagonal
reg = regLamba * np.eye(f,f)


#alternating least squares
def als(iters):
    for k in range(1, iters):
        for i in range(1, m):
            idx = nonZero[i,:]
            a = Y[idx,]
            b = np.dot(np.transpose(Y[idx,]), ratingsMatrix[i, idx])
            updateX = np.linalg.solve(np.dot(np.transpose(a), a) + reg, b)
            X[i,] = updateX
        
        for j in range(1, n):
            idx = nonZero[:,j]
            a = X[idx,]
            b = np.dot(np.transpose(X[idx,]), ratingsMatrix[idx, j])
            updateY = np.linalg.solve(np.dot(np.transpose(a), a) + reg, b)
            Y[j,] = updateY
            
        ratingsP = ratingsPred(X, Y)
        mse = MSE(ratingsP, ratingsMatrix)
        print("MSE: " + str(mse))
            
    return print("Done")

#iteration test
als(iters = 10) 


#with multiprocessing
#alternating least squares
from multiprocessing import Pool

#goof around
def sqit(x):
    return print(x**2)

if __name__== '__main__': 
    pool = Pool(processes = 3) #ncores
    pool.map(sqit, range(0,4))

pool.close()


#with als ---
def row_update(i):
    idx = nonZero[i,:]
    a = Y[idx,]
    b = np.dot(np.transpose(Y[idx,]), ratingsMatrix[i, idx])
    updateX = np.linalg.solve(np.dot(np.transpose(a), a) + reg, b)
    return updateX

def col_update(j):
    idx = nonZero[:,j]
    a = X[idx,]
    b = np.dot(np.transpose(X[idx,]), ratingsMatrix[idx, j])
    updateY = np.linalg.solve(np.dot(np.transpose(a), a) + reg, b)
    Y[j,] = updateY
    return Y


# Initialise Y matrix, n x f
Y = initialiseMatrix(n, f) # item decomp matrix
# Initialise X matrix, m x f
X = initialiseMatrix(m, f) #user decomp matrix



if __name__== '__main__': 
    pool = Pool(processes = 2) #ncores
    X = np.vstack(pool.map(row_update, range(1,m))) #row bind the results
    pool.close()
    
if __name__== '__main__': 
    pool = Pool(processes = 2) #ncores
    Y = np.vstack(pool.map(row_update, range(0,n))) #row bind the results
    pool.close()

ratingsP = ratingsPred(X, Y)
mse = MSE(ratingsP, ratingsMatrix)
print("MSE: " + str(mse))



