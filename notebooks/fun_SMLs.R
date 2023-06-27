### Functions to read NOAA csv files 

split_strings <- function(x, obs_name, n_obs = NULL, split = ',') {
  # function to split observations with string comprising of multiple values
  # into matrix. Function is designed for meteostation csv files obtianed from
  # NOAA for stations operated by Czech meteo institute
  #@ Arguments:
  #@ x - data.frame with all observations
  #@ obs_name - name of column of observations to be split
  #@ n_obs - (maximum) number of observations in a single
  #@ split - splting character
  #@ Value:
  #@ Matrix where each row represent one observatoin and each column a parameter
  
  
  
  obsList <- strsplit(x[[obs_name]], ',')
  len <- unlist(lapply(obsList, length))
  
  if (is.null(n_obs)) {
    n_obs <- max(len)
  }
  
  obsMtx <- matrix(NA, n_obs, length(obsList)) # observation matrix
  
  # fill in into matrix only complete observations
  idCompleteObs <- which(len == n_obs)  
  if (length(idCompleteObs) == 0){stop('No avialable observations!')}
  obsList <- strsplit(as.character(x[[obs_name]][idCompleteObs]), ',')
  obsMtx[, idCompleteObs] <- unlist(obsList)
  
  #obsDf <- as.data.frame(t(obsMtx))
  
  return(t(obsMtx))
  
}

zoo_precip_AA1 <- function(x) {
  # create zoo series with precip. data
  require(zoo)
  tim <- strptime(x$DATE, "%Y-%m-%dT%H:%M:%S")
  iddupl <- which(duplicated(tim))
  
  if (length(iddupl) > 0){
    x <- x[-iddupl, ]
    tim <- tim[-iddupl]
    warning('observations with duplicated timestamps were removed')
  }
  
  precip <- split_strings(x, obs_name = 'AA2')
  zoo_precip <- zoo(as.numeric(precip[, 2]) / 10, tim)
  
  # treat NAs
  zoo_precip[zoo_precip > 999] <- NA
  
  return(zoo_precip)
}