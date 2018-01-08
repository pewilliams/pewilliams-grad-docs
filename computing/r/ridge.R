require(mlbench)
data('BostonHousing')
dat <- BostonHousing #outcome of interest is medv median value of homes

gelman_scale <- function(x){
  (x - mean(x)) /(2*sd(x))
}

y <- dat$medv
y <- gelman_scale(y)

X <- dat[,c('crim','indus','nox','ptratio','age','dis','tax')] #play with numeric
X <- as.matrix(data.frame(lapply(X, function(a) gelman_scale(a))))


#rr estimator

mle_est <- solve(t(X) %*% X)%*%t(X)%*%y
yhat_mle <- X %*% mle_est
rmse_mle <- sqrt(mean((yhat_mle - y)^2))


alpha = 0.2
rr_est <- solve(t(X) %*% X + alpha*(diag(ncol(X))))%*%t(X)%*%y
yhat_rr <- X %*% rr_est
rmse_rr <- sqrt(mean((yhat_rr - y)^2))


oos_test <- function() {
err <- sapply(seq(0,2,by = 0.1), function(alpha) {
	row_sample <- sample(1:506, 50)
	Xtrain <- X[row_sample,]
	Xtest <- X[-row_sample,]
	rr_est <- solve(t(Xtrain) %*% Xtrain + alpha*(diag(ncol(Xtrain))))%*%t(Xtrain)%*%y[row_sample]
	yhat_rr <- Xtest %*% rr_est
	sqrt(mean((yhat_rr - y[-row_sample])^2))
})

data.frame(alpha = seq(0,2,by=0.1), err = err)
}

a <- do.call('rbind', lapply(1:500, function(x) oos_test()))

ggplot(a, aes(x = factor(alpha), y = err)) + geom_boxplot()





