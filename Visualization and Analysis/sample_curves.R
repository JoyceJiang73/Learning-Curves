
#
# Sample learning curves as shown in the paper (Fig. 2); run fuse_data.R first
#

load('metricsAll.Rd')

datasetName = 'NLP'
res = fuseSimulations('data/allEpoch_nlp/',pattern='*KNN*', nepochs = 50, nsims = 20)
labels = c('anger','fear','joy','love','sadness','surprise') # emotion

res$class = gsub('processing curve sim_.*?_(.*?).csv','\\1',res$fl)
res$class = gsub('-','/',res$class)
res[1:5,]

for (i in 0:5) {
  res$class = gsub(paste('Class',i,sep=''),labels[i+1],res$class)
}
res[1:5,]

classes = unique(res$class)
compares = unique(res$class)

png("Sim1_Epochs_Combined.png", width=1800, height=1800, res=300)

#par(mfrow=c(2,2),mar=c(5,5,5,1))
par(mfrow=c(2,2))

tmp_dat = res[res$sim==1&res$epoch==1&res$cl=='anger/fear',]
plot(tmp_dat$t,tmp_dat$dv,type='o',col='black',ylim=c(0.5,1),pch=16,ylab='KNN classification performance',
     xlab='Time slice in trial',main='Sim. 1, Epoch 1 (Sentence, anger/fear)')
metrics = metricsAll[metricsAll$sim==1&metricsAll$epoch==1&metricsAll$cl=='anger/fear',]
metrics$end_m_start = metrics$vend-metrics$v0
text(10,0.55,paste(round(metrics[,c(4,9,6,7)],2),collapse='    '),pos=4)

tmp_dat = res[res$sim==1&res$epoch==50&res$cl=='anger/fear',]
plot(tmp_dat$t,tmp_dat$dv,type='o',col='black',ylim=c(0.5,1),pch=16,ylab='KNN classification performance',
     xlab='Time slice in trial',main='Sim. 1, Epoch 20 (NLP, anger/fear)')
metrics = metricsAll[metricsAll$sim==1&metricsAll$epoch==50&metricsAll$cl=='anger/fear',]
metrics$end_m_start = metrics$vend-metrics$v0
text(10,0.55,paste(round(metrics[,c(4,9,6,7)],2),collapse='    '),pos=4)

#### GESTURE

datasetName = 'gesture'
res = fuseSimulations('data/allEpoch_gesture',pattern='*KNN*', nepochs = 50, nsims = 20)
labels = c('no gestures','body','head','hand','body-head','head-hand movements') # gesture

res$class = gsub('processing curve sim_.*?_(.*?).csv','\\1',res$fl)
res$class = gsub('-','/',res$class)
res[1:5,]

for (i in 0:5) {
  res$class = gsub(paste('Class',i,sep=''),labels[i+1],res$class)
}
res[1:5,]

classes = unique(res$class)
compares = unique(res$class)

# GESTURE
tmp_dat = res[res$sim==1&res$epoch==1&res$cl=='no gestures/hand',]
plot(tmp_dat$t,tmp_dat$dv,type='o',col='black',ylim=c(0.5,1),pch=16,ylab='KNN classification performance',
     xlab='Time slice in trial',main='Sim. 1, Epoch 1 (gesture, no gestures/hand)')
metrics = metricsAll[metricsAll$sim==1&metricsAll$epoch==1&metricsAll$cl=='no gestures/hand',]
metrics$end_m_start = metrics$vend-metrics$v0
text(10,0.55,paste(round(metrics[,c(4,9,6,7)],2),collapse='    '),pos=4)

tmp_dat = res[res$sim==1&res$epoch==50&res$cl=='no gestures/hand',]
plot(tmp_dat$t,tmp_dat$dv,type='o',col='black',ylim=c(0.5,1),pch=16,ylab='KNN classification performance',
     xlab='Time slice in trial',main='Sim. 1, Epoch 20 (gesture, no gestures/hand)')
metrics = metricsAll[metricsAll$sim==1&metricsAll$epoch==20&metricsAll$cl=='no gestures/hand',]
metrics$end_m_start = metrics$vend-metrics$v0
text(10,0.55,paste(round(metrics[,c(4,9,6,7)],2),collapse='    '),pos=4)

# Close the PNG device
dev.off()



