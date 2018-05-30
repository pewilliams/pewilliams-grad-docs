## normal and half-normal quantile plots
normalplot <- function(y, label=F,  n=length(y), fac.names=NULL, xlim=c(-2.2, 2.2), main="Normal Plot", ...)
{ # label the most singificant n effects
  m <- length(y)
  x <- seq(0.5/m, 1.0-0.5/m, by=1/m)
  x <-  qnorm(x)
  y <-  sort(y)
  qqplot(x, y, xlab="normal quantiles", ylab="effects",  xlim=xlim, main=main, ...)
  if(is.null(fac.names)) fac.names <-  names(y)
  else fac.names <-  rev( c(fac.names, rep("", length(y)-length(fac.names)) ) )
  
  ord=order(abs(y))
  if(label) for(i in ord[(m-n+1):m]) text(x[i]+.35,y[i], fac.names[i])  
}

# we modify the halfnormal plot function to show label names
halfnormal <-halfnormalplot <- function(y, label=F, n=length(y), fac.names=NULL, xlim=c(-.1, 2.5), main="Half-Normal Plot",  ...)
{ # label the most singificant n effects 
  m <- length(y)
  x <- seq(0.5+0.25/m, 1.0-0.25/m, by=0.5/m)
  x <-  qnorm(x)
  y <-  sort(abs(y))
  qqplot(x, y, xlab="half-normal quantiles", ylab="absolute effects",  xlim=xlim, main=main, ...)
  if(is.null(fac.names)) fac.names <-  names(y)
  else fac.names <-  rev( c(fac.names, rep("", length(y)-length(fac.names)) ) )
  if(label) for(i in (m-n+1):m) text(x[i]+.2,y[i], fac.names[i])  
}
