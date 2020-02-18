# Do some analysis to see if the CV/ER is correlated with the Chronoquestionnaire scores.
# Or for the impulsivity questionnaire - it's the same process.

# install.packages("ggplot2")
library("ggplot2")

# Main data: pax, chronoQ -------------------------------------------------


# NIPs of the pax
pax     <- c('hm070076', 'fr190151', 'at140305', 'cc150418', 'eb180237', 
             'ld190260', 'ms180425', 'ch180036', 'cg190026', 'ih190084', 
             'cr170417', 'll180197', 'tr180110', 'lr190095', 'ep190335', 
             'gl180335', 'ad190335', 'ag170045', 'pl170230', 'ma190185')
paxnum  <-c(1:20)
# Obtained score of the chrono questionnaire
chronoq <- c(59, 54, 53, 58, 70, 
             35, 42, 25, 61, 47, 
             38, 67, 48, 63, 45, 
             56, 49, 66, 47, 46)
# Cathegory of the chrono questionnaire according to the obtained score of the answers
chwakeup <- c(4, 3, 3, 3, 5, 
              2, 3, 1, 4, 3, 
              2, 4, 3, 4, 3, 
              3, 3, 4, 3, 3)

chrono <- data.frame(pax, paxnum, chronoq, chwakeup, stringsAsFactors=FALSE)

# Setting the directory --------------------------------------

# I need to read the int dur columns from the .txt files and store them in
# specific variables such as B1_145
# Then before moving to the next pax, store the columns, as a list in 
# an element in the df 
dir1 <- 'D:/ScaledTime/Matlab data/Final_folders/'
dir2 <- 'D:/ScaledTime/Matlab data/'
intervals <- list('1.45', '2.9', '5.8')

# Find the outliers and remove them  --------------------------------------

for (i in 1:length(pax)) {
  dir <- paste(dir1, 'ScaledTime', i, '/', sep = "")
  
  for (j in (1:6)) { # iterates through 6 blocks
    
    for (k in 1:length(intervals)) { # iterates through 3 intervals
      
      filename <- paste(dir, 'ScaledTime_Play_subj_', i, '_bl_', j, '_int_', intervals[k], '.txt', sep = "")
      if (file.exists(filename)) {
        d <- read.table(paste(filename, sep=''), header=T) 
        
        # Column names
        ime <- paste("int_out_", d$shuffledInterval[1],  sep = "")
        
        ###chrono[i, ime][[1]] <- list(c(unlist(chrono[i, ime]), d$interval))
        
        #####
        # In this part, I need to exclude the 5*int outliers
        woout1 <- list(c())
        for (m in 1:length(d$interval)) {
          inte <- d$interval[m] # inte is a single interval they are producing store in
          # the column d$interval 
          if (inte < 5 * as.numeric(unlist(d$shuffledInterval[1]))) {
              woout1 <- c(unlist(woout1), inte)
          }
        }
        # Adds the list in a specific cell but iteratively -- after the previous values
        chrono[i, ime][[1]] <- list(c(unlist(chrono[i, ime]), woout1))
      }
    }
  }
}

for (i in 1:length(pax)) {
  # iterates through 3 intervals
  for (k in 1:length(intervals)) { 
    
    ime <- paste("int_out_", intervals[k],  sep = "")
    # ime2 <- paste("int_out_ch_", intervals[k],  sep = "")
    
    # Calculate the mean
    imemean <- paste("int_out_", intervals[k], '_mean', sep = "")
    chrono[i, imemean] <- lapply(chrono[i,ime], mean)
    
    # Calculate the standard deviation SD
    imeSD <- paste("int_out_", intervals[k], '_sd', sep = "")
    chrono[i, imeSD] <- lapply(chrono[i,ime], sd)
    
    # Set the limits for the exclusion of outliers
    hlimitname <- paste("high_limit_", intervals[k], sep = "")
    llimitname <- paste("low_limit_",  intervals[k], sep = "")
    
    chrono[i, hlimitname]  <- chrono[i, imemean] + 3 * chrono[i, imeSD]
    chrono[i, llimitname]  <- chrono[i, imemean] - 3 * chrono[i, imeSD]

  }
}
    

# Data reading w/o outliers: blocks separated, pax separated --------------


for (i in 1:length(pax)) {
  dir <- paste(dir1, 'ScaledTime', i, '/', sep = "")

  for (j in (1:6)) { # iterates through 6 blocks
    # ints <- c()
    
    for (k in 1:length(intervals)) { # iterates through 3 intervals
      filename <- paste(dir, 'ScaledTime_Play_subj_', i, '_bl_', j, '_int_', intervals[k], '.txt', sep = "")
      if (file.exists(filename)) {
        d <- read.table(paste(filename, sep=''), header=T)

        # Column names
        # It adds all the data in a separate column per block
        ime <- paste("B", j, "_", d$shuffledInterval[1],  sep = "")
        
        # In this part, I need to exclude the outliers that I found in the previous section,
        # before importing the data per block into the dataframe. 
        woout2 <- list(c())
        for (m in 1:length(d$interval)) {
          hlimitname <- paste("high_limit_", d$shuffledInterval[1], sep = "")
          llimitname <- paste("low_limit_",  d$shuffledInterval[1], sep = "")
          inte <- d$interval[m] # inte is a single interval they are producing store in
                                # the column d$interval 
            if (inte > chrono[i, llimitname] && inte < chrono[i, hlimitname]){
              woout2 <- c(unlist(woout2), inte)
            }
          }
        # Adds the list in a specific cell
        chrono[i, ime][[1]] <- list(c(unlist(chrono[i, ime]), woout2))
        
        # Normalize the interval productions
        imenorm <- paste("B", j, "_", d$shuffledInterval[1], '_norm', sep = "")
        # chrono[i, imenorm] = c()
        normzd <- c()
        for (p in chrono[i, ime]) {
          normzd <- list(c(normzd, p/as.numeric(intervals[k]))) # intervals[k] == d$shuffledInterval[1]
        }
        chrono[i, imenorm][[1]] <- normzd

        # Calculate the mean
        imemean <- paste("B", j, "_", d$shuffledInterval[1], '_mean', sep = "")
        chrono[i, imemean] <- lapply(chrono[i,ime], mean)

        # Calculate the standard deviation SD
        imeSD <- paste("B", j, "_", d$shuffledInterval[1], '_sd', sep = "")
        chrono[i, imeSD] <- lapply(chrono[i,ime], sd)

        # Calculate the CVs
        imeCV <- paste("B", j, "_", d$shuffledInterval[1], '_CV', sep = "")
        chrono[i, imeCV] <- chrono[i, imeSD]/chrono[i, imemean]

        # Calculate the ERs
        imeER <- paste("B", j, "_", d$shuffledInterval[1], '_ER', sep = "")
        chrono[i, imeER] <- chrono[i, imemean]/as.numeric(intervals[k])

      }
    }
  }
}



# Data processing: ints - all blocks together, pax separated --------------

# Put all int productions of all blocks, per interval, of a pax in one columnn
  
for (i in 1:length(pax)) {
  # All blocks together
  chrono[i,"int_1.45"][[1]] <- list(c(unlist(chrono[i, "B1_1.45"]), unlist(chrono[i, "B2_1.45"]),
                                      unlist(chrono[i, "B3_1.45"]), unlist(chrono[i, "B4_1.45"]), 
                                      unlist(chrono[i, "B5_1.45"]), unlist(chrono[i, "B6_1.45"])))
  chrono[i,"int_2.9"][[1]]  <- list(c(unlist(chrono[i, "B1_2.9"]), unlist(chrono[i, "B2_2.9"]),
                                      unlist(chrono[i, "B3_2.9"]), unlist(chrono[i, "B4_2.9"]), 
                                      unlist(chrono[i, "B5_2.9"]), unlist(chrono[i, "B6_2.9"])))
  chrono[i,"int_5.8"][[1]]  <- list(c(unlist(chrono[i, "B1_5.8"]), unlist(chrono[i, "B2_5.8"]),
                                      unlist(chrono[i, "B3_5.8"]), unlist(chrono[i, "B4_5.8"]), 
                                      unlist(chrono[i, "B5_5.8"]), unlist(chrono[i, "B6_5.8"])))
  
  # All blocks together, normalized
  # 1.45
  normzed1 <- c()
  for (p in chrono[i, "int_1.45"]) {
    normzed1 <- list(c(normzed1, p/1.45))
  }
  chrono[i, "int_1.45_norm"][[1]] <- normzed1
  # 2.9
  normzed2 <- c()
  for (p in chrono[i, "int_2.9"]) {
    normzed2 <- list(c(normzed2, p/2.9))
  }
  chrono[i, "int_2.9_norm"][[1]] <- normzed2
  # 5.8
  normzed3 <- c()
  for (p in chrono[i, "int_5.8"]) {
    normzed3 <- list(c(normzed3, p/5.8))
  }
  chrono[i, "int_5.8_norm"][[1]] <- normzed3
  
}


# All blocks together: Means of the all int productions (6 blocks together) per pax
chrono$int_1.45_mean <- lapply(chrono$int_1.45, mean)
chrono$int_2.9_mean  <- lapply(chrono$int_2.9, mean)
chrono$int_5.8_mean  <- lapply(chrono$int_5.8, mean)

# All blocks together: SDs of the all int productions (6 blocks together) per pax
chrono$int_1.45_sd <- lapply(chrono$int_1.45, sd)
chrono$int_2.9_sd  <- lapply(chrono$int_2.9, sd)
chrono$int_5.8_sd  <- lapply(chrono$int_5.8, sd)

# All blocks together: CV
chrono$int_1.45_CV <- unlist(chrono$int_1.45_sd) / unlist(chrono$int_1.45_mean)
chrono$int_2.9_CV  <- unlist(chrono$int_2.9_sd)  / unlist(chrono$int_2.9_mean)
chrono$int_5.8_CV  <- unlist(chrono$int_5.8_sd)  / unlist(chrono$int_5.8_mean)

# All blocks together: ER
chrono[i, imeER] <- chrono[i, imemean]/as.numeric(intervals[k])
chrono$int_1.45_ER <- unlist(chrono$int_1.45_mean) / 1.45
chrono$int_2.9_ER  <- unlist(chrono$int_2.9_mean)  / 2.9
chrono$int_5.8_ER  <- unlist(chrono$int_5.8_mean)  / 5.8

################################
################################ Final plots
################################

# Plotting x-->chronoq; y-->ints ER/CV
# ER
ggplot(data=chrono, aes(x=chronoq, y=int_1.45_ER)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chronoq, y=int_2.9_ER)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chronoq, y=int_5.8_ER)) + geom_point() + geom_smooth(method="lm")
# CV
ggplot(data=chrono, aes(x=chronoq, y=int_1.45_CV)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chronoq, y=int_2.9_CV)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chronoq, y=int_5.8_CV)) + geom_point() + geom_smooth(method="lm")

# Plotting x-->chronoq; y-->ints ER/CV
# ER
ggplot(data=chrono, aes(x=chwakeup, y=int_1.45_ER)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chwakeup, y=int_2.9_ER)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chwakeup, y=int_5.8_ER)) + geom_point() + geom_smooth(method="lm")
# CV
ggplot(data=chrono, aes(x=chwakeup, y=int_1.45_CV)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chwakeup, y=int_2.9_CV)) + geom_point() + geom_smooth(method="lm")
ggplot(data=chrono, aes(x=chwakeup, y=int_5.8_CV)) + geom_point() + geom_smooth(method="lm")







################################
################################ Draft plots
################################

# Create a new df with all blocks together  -------------------------

# chronoall <- data.frame(chrono$pax,           chrono$chronoq,      chrono$chwakeup,
#                         chrono$int_1.45,      chrono$int_2.9,      chrono$int_5.8,
#                         chrono$int_1.45_norm, chrono$int_2.9_norm, chrono$int_5.8_norm,
#                         chrono$int_1.45_mean, chrono$int_2.9_mean, chrono$int_5.8_mean,
#                         chrono$int_1.45_sd,   chrono$int_2.9_sd,   chrono$int_5.8_sd,
#                         chrono$int_1.45_CV,   chrono$int_2.9_CV,   chrono$int_5.8_CV,
#                         chrono$int_1.45_ER,   chrono$int_2.9_ER,   chrono$int_5.8_ER)

# Data processing: pax together, blocks separated -------------------------

### Create a new data frame with all the blocks 1 of all pax together
#$B1_1.45 <- c(B1_1.45)




# Save the chrono dataframe -----------------------------------------------
setwd("D:/ScaledTime/Data analyses/R data and plots/")
save(chrono,file="chrono_fixed.Rda")
load("chrono.Rda")


# Plotting ----------------------------------------------------------------

# Density plots per interval
d <- density(unlist(chrono$int_1.45)) # returns the density data
plot(d) # plots the results
lines (density(unlist(chrono$int_2.9)))
lines (density(unlist(chrono$int_5.8)))

# Plotting the density w/ ggplot
library('reshape2')
d2 <- melt(chrono$int_1.45)

library('ggplot2')
ggplot(d2, aes(x = value)) + geom_density(alpha = .25)


# plotting ----------------------
# To show different cathegories of intervals, turn the chrono$int_1.45 and chrono$int_2.9 in a 
# longformat, where one column will denote the diff cathegories
library(tidyr)
chrono2 <- data.frame(chrono$pax, chrono$chronoq, chrono$chwakeup,
                      chrono$int_1.45_CV, chrono$int_2.9_CV, chrono$int_5.8_CV,
                      chrono$int_1.45_ER, chrono$int_2.9_ER, chrono$int_5.8_ER)
chrono_data_long <- gather(chrono2, condition, measurement, chrono.int_1.45_CV:chrono.int_5.8_ER, factor_key=TRUE)

# Creating a new df
chrono3 <- data.frame(1:20)
chrono3$pax <- chrono$pax
chrono3$int_1.45 <- chrono$int_1.45
chrono3$int_2.9 <- chrono$int_2.9
chrono3$int_5.8 <- chrono$int_5.8

data_long2 <- gather(chrono3, condition, measurement, int_1.45:int_5.8, factor_key = TRUE)
as.data.frame(lapply(data_long2, unlist))
data_long2 %>%
  transform(measurement = split(y, ",")) %>%
  df <- unnest(measurement)



### tuka nekade sum, probuvam da gi izvadam od lista elementite



# ggplot(data=dat, aes(x=estim1, y=sequenceDur1, by=n_items, color=n_items)) + 
#   geom_point() +
#   facet_wrap(~ subj)

# ggplot(data=chrono, aes(x=pax, y=int_1.45_CV)) + geom_point()
ggplot(data=chrono, aes(x=paxnum, y=int_2.9_CV)) + geom_point()
ggplot(data=chrono, aes(x=int_1.45_CV, y=int_1.45_ER)) + geom_point()





chrono2 <- data.frame(chrono$pax, chrono$chronoq, chrono$chwakeup,
                      chrono$int_1.45_CV, chrono$int_2.9_CV, chrono$int_5.8_CV,
                      chrono$int_1.45_ER, chrono$int_2.9_ER, chrono$int_5.8_ER)

x <- cbind(chrono$int_1.45_CV, chrono$int_2.9_CV, chrono$int_5.8_CV)
y <- cbind(chrono$int_1.45_ER, chrono$int_2.9_ER, chrono$int_5.8_ER)



s <- 
  "A       B        C       G       Xax
0.451   0.333   0.034   0.173   0.22        
0.491   0.270   0.033   0.207   0.34    
0.389   0.249   0.084   0.271   0.54    
0.425   0.819   0.077   0.281   0.34
0.457   0.429   0.053   0.386   0.53    
0.436   0.524   0.049   0.249   0.12    
0.423   0.270   0.093   0.279   0.61    
0.463   0.315   0.019   0.204   0.23
"


d <- read.delim(textConnection(s), sep="")

library(ggplot2)
library(reshape2)
d <- melt(d, id.vars="Xax")

# Everything on the same plot
ggplot(d, aes(Xax,value, col=variable)) + 
  geom_point() + 
  stat_smooth() 

# Separate plots
ggplot(d, aes(Xax,value)) + 
  geom_point() + 
  stat_smooth() +
  facet_wrap(~variable)
