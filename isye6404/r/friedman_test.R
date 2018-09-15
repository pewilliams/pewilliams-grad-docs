# 2018-09-15
# pewilliams
# rework of http://users.stat.umn.edu/~helwig/notes/np2loc-Notes.pdf

#Model: X_{ij} = \theta + \beta_i + \tau_j + \varepsilon_{ij}
#beta to control blocking effect
#H_O: \tau_1 = \tau_2 = ... = tau_k
#H_{\alpha}: \tau_i \neq \tau_j \text{for some} i \neq j 

friedman_test <- function(X){ #with rank ties
  #matrix dimension
  n <- nrow(X)
  k <- ncol(X)
  rt_rank <- apply(X, 1, rank)
  #tie helper - table groups number of tied values for cubing
  cube_row_rank <- function(x){
    return(sum(table(x)^3) - k)
  }
  #helper stats
  R_k <- apply(rt_rank,1,sum)
  numerator <- 12 * (sum((R_k - n*(k+1)/2)^2))
  denominator <- n*k*(k + 1) - 1/(k-1)*sum(apply(rt_rank, 2, cube_row_rank))
  S_prime <- numerator/denominator #test statistic
  pval <- 1 - pchisq(q = S_prime, df = k - 1) #pvalue
  return(list(Test_stat = S_prime, pvalue = pval ))
}

#base running strategies data
rtimes = matrix(c(5.40, 5.50, 5.55,
                  5.85, 5.70, 5.75,
                  5.20, 5.60, 5.50,
                  5.55, 5.50, 5.40,
                  5.90, 5.85, 5.70,
                  5.45, 5.55, 5.60,
                  5.40, 5.40, 5.35,
                  5.45, 5.50, 5.35,
                  5.25, 5.15, 5.00,
                  5.85, 5.80, 5.70,
                  5.25, 5.20, 5.10,
                  5.65, 5.55, 5.45,
                  5.60, 5.35, 5.45,
                  5.05, 5.00, 4.95,
                  5.50, 5.50, 5.40,
                  5.45, 5.55, 5.50,
                  5.55, 5.55, 5.35,
                  5.45, 5.50, 5.55,
                  5.50, 5.45, 5.25,
                  5.65, 5.60, 5.40,
                  5.70, 5.65, 5.55,
                  6.30, 6.30, 6.25),ncol=3,byrow=TRUE)


friedman_test(rtimes)





