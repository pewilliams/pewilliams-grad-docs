#brute force HPD implementation
#example implementation below based on randomly generated normal variates
N <- 1e04
x <- rnorm(N)
x <- sort(x, decreasing = F)
interval <- 0.9

#brute force
find_hpd <- function(x, interval){
  n <- length(x)
  x <- sort(x, decreasing = F)
  point_int <- round(n * interval, digits = 0)
  start_point <- 1:(n-point_int + 1)
  end_point <- start_points + 8999
  
  hpd_res <- rbindlist(mclapply(1:(n-point_int+1), 
    function(a){
      data.frame(
        max_x = x[start_point[a]],
        min_x = x[end_point[a]],
        range_x = x[end_point[a]] - x[start_point[a]],
        index_x_range = paste0('[',start_point[a],',',end_point[a],']'),
        stringsAsFactors = F
      )
    }, mc.cores = 10))

    return(hpd_res[which.min(hpd_res$range_x)])
}

find_hpd(x = x, interval = 0.9)


