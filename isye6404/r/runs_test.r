#!/usr/bin/env Rscript
#pewilliams 2018-09-03
#runs test - exact distribution
runs_prob <- function(n1,n2,r){
	
	#must be at least two runs in a runs test
	if(r < 2){
		print('there must be at least two runs in a runs test')
		}
	
	#case even probability
	else if(r %% 2 == 0){
		k <- r/2
		return(2*(choose(n1-1,k-1) * choose(n2-1,k-1))/choose(n1 + n2, n2))
	
	#case odd probability
	}else if(r %%2 != 0){
		k <- (r - 1)/2
		numerator <- choose(n1-1,k)*choose(n2-1,k-1) + choose(n1-1,k-1)*choose(n2-1,k)
		denominator <- choose(n1+n2,n2)
		return(numerator/denominator)
	}

}

#test critical region <= number_runs
cume_runs_prob <- function(n1,n2,number_runs){
	return(sum(sapply(2:number_runs, function(x) {
		runs_prob(n1=n1,n2=n2,x)
	})))
}

#see p257 Hogg | Tanis "Probability & Statistical Inference"
#test case 8.6-1
#runs_prob(n1=10,n2=10,r=2) #2/184756 2/choose(20,10) = 1.082509e-05
#test up to r <= 7 -> 0.051
#cume_runs_prob(n1=10,n2=10,number_runs=7)
#test up to r <= 7 -> 0.051

#test case 8.6-5 $C = \{r: r \leq 11\}$
#cume_runs_prob(n1=15,n2=15,number_runs=11)