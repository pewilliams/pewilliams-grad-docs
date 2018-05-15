require(gpuR)

np<-50 #number of predictors
nr<-1e06 #number of observations

#synethic datatset and outcome
X<-cbind(1,matrix(rnorm((np-1)*nr,0,.01),nrow=nr,ncol=(np-1)))
betas <- matrix(c(1,3,runif(np-2,0,0.2)),ncol=1)
y<-X %*% betas+ matrix(rnorm(nr,0,1),nrow=nr,ncol=1)


#Ridge Regression - apply across various settings for alpha

ridge_res <- lapply(seq(0,15, by = 0.5), function(alpha) { 	
	beta_length <- ncol(X)
	alpha_Ip <- alpha * diag(beta_length)
	vcl_X = vclMatrix(X,type="float") 
	vcl_XtX = 
	vcl_alpha_Ip = vclMatrix(alpha_Ip,type="float") 
	vcl_y = vclMatrix(y, type="float")
	rr_res <-gpuR::solve(gpuR::crossprod(vcl_X) + vcl_alpha_Ip) %*% gpuR::crossprod(vcl_X, vcl_y)
	res_out <- data.frame(t(as.matrix(rr_res)), alpha) 
	res_out
	
	})

#combine results into one data frame
rr_path <- do.call('rbind',ridge_res)

#visualize the impact of penalization terms on effect estimates
with(rr_path, {
	paths <- rr_path[,!grepl('alpha', names(rr_path))]
	plot(NULL, ylim = c(.5, -.5), xlim = c(min(alpha), max(alpha)), bty = 'n', xlab = 'alpha')
	lapply(2:ncol(paths), function(c){ #ignore first two co-vars with static beta effects
		lines(alpha, paths[,c], col = c, lwd = 2)
	})
	})

#garbage cleanup
gc()





