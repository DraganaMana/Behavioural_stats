# Initial checks for Analyze timeInWM
# TW Kononowicz

library(RcppCNPy)
library(R.utils)
library(scales)
library(LambertW)
library(ggplot2)
library(ggExtra)
library(mgcv)
library(lme4)
library(lmerTest)
library(dplyr)
library(itsadug)
library(plyr)
library(reshape2)

#=======================================================================================
options(digits=8)
if (Sys.info()['login']=='tadeusz') {
  dir <- '/Users/tadeusz/Dropbox/experiments/timeInWM/v001/Data/' 
} else if (Sys.info()['login']=='Izem') {
  dir <- '/Users/Izem/Dropbox/timeInWM/v001/Data/'
} else if (Sys.info()['login']=='sophie') {
  dir <- '/Users/sophie/Nextcloud/PROJECT_WMtime/myTimeinWM_DB/v001/Data/'
}	

ls <- list.files(dir)
loopCount <- 1

for (i in c(1:length(ls))) {
  if (grepl('timeinWMv1', ls[i]) & grepl('.txt', ls[i]) & grepl('bl_', ls[i])) {
    d <- read.table(paste(dir, ls[i], sep=''), header=T)   
    if (loopCount==1) { dat <- d 
    } else 	{
      dat <- rbind(dat, d)}
    loopCount <- loopCount+1 } }
# write.table(dat, '/Users/tadeusz/Data/ .csv') 

dat <- subset(dat, errorTrial!=1)

# TODO Data cleaning: to be extended....
#=======================================================================================

dat$estimNorm1 <- dat$estim1/dat$sequenceDur1
dat$estimNorm2 <- dat$estim2/dat$sequenceDur2
dat$estimNorm3 <- dat$estim3/dat$sequenceDur3

dat$last_item <- dat$n_items
dat$last_item <- ifelse(dat$n_items==1, dat$estim1,
                        ifelse(dat$n_items==2, dat$estim2,
                               ifelse(dat$n_items==3, dat$estim3, 0)))
#=======================================================================================

ggplot(data=dat, aes(x=estim1, y=sequenceDur1, by=n_items, color=n_items)) + 
  geom_point() +
  facet_wrap(~ subj)

ggplot(data=dat, aes(x=estimNorm1, y=n_items)) + 
  geom_point() +
  facet_wrap(~ subj)

ggplot(data=dat, aes(x=estimNorm1)) + 
  geom_density() +
  facet_wrap(~ subj + n_items) +
  theme_minimal()+
  geom_vline(xintercept=1., color='dodgerblue', linetype='dashed')
# geom_hline(yintercept=2.25, color='firebrick', linetype='dashed') 



# geom_violin(trim=F) + labs(x='Target duration', y = 'Produced interval (s)' ) + scale_fill_manual(values=c('dodgerblue', 'firebrick'), labels=c('1.5 s', '2.25s')) 
# theme_bw(base_size=15) + theme(panel.border=element_blank(), panel.grid.major=element_blank(), axis.line=element_line(color='black'), plot.title = element_text(hjust = 0.5)) +
# labs(fill='Target durationt', labels=c('short','correct','long'), title='FOJ by target duration') +
# theme(legend.position='right') +
# geom_boxplot(width=0.04, fill='white', outlier.size=NA)+
# scale_x_discrete(labels=c( '1'='1.5 s', '2'='2.25s'))

# make long table where each item is a row
library(tidyr)
tmp <- dat %>% dplyr::select('subj', 'block', 'trial','n_items', 'estimNorm1', 'estimNorm2', 'estimNorm3','sequenceDur1','sequenceDur2','sequenceDur3')
dat_long <- gather(tmp, position, ProdNorm, estimNorm1:estimNorm3, factor_key=TRUE)
dat_long

# rename the positions
dat_long$position <- ifelse(dat_long$position=='estimNorm1', 1,
                            ifelse(dat_long$position=='estimNorm2', 2,
                                   ifelse(dat_long$position=='estimNorm3', 3, 0)))
# add the duration per item
dat_long$item_dur <- dat_long$n_items
dat_long$item_dur <- ifelse(dat_long$position==1, dat_long$sequenceDur1,
                            ifelse(dat_long$position==2, dat_long$sequenceDur2,
                                   ifelse(dat_long$position==3, dat_long$sequenceDur3, 0)))
# clean up
dat_long$sequenceDur1 <- NULL
dat_long$sequenceDur2 <- NULL
dat_long$sequenceDur3 <- NULL

# now plot position x n_items per SB
sbs = sort(unique(dat$subj))
for (sb in sbs){
  plot = ggplot(data=dat_long[dat_long$subj==sb,], aes(x=ProdNorm)) +  
    geom_density() +
    facet_wrap(~ n_items + position) +
    theme_minimal()+
    geom_vline(xintercept=1., color='dodgerblue', linetype='dashed') 
  print(plot)
}

# now plot position x n_items 
q <- ggplot(data=dat_long, aes(x=ProdNorm)) +  
  geom_density(color="darkblue", size=1) +
  facet_wrap(~ n_items + position) +
  theme_minimal()+
  geom_vline(xintercept=1., color='dodgerblue', linetype='dashed') 

q + xlim(0,4)

# plot precision for single items per duration
dat_long_tmp = dat_long[dat_long$n_items==1,]
p <- ggplot(data=dat_long_tmp[dat_long_tmp$item_dur!=0,], aes(x=ProdNorm)) +  
  geom_density(color="darkblue", size = 1) +
  facet_wrap(~ item_dur, nrow=3) +
  geom_vline(xintercept=1., color='dodgerblue', linetype='dashed') +
  geom_hline(yintercept=1, color='dodgerblue', linetype='dashed') +
  theme_minimal()
p + xlim(0,3) + ylim(0,2)


