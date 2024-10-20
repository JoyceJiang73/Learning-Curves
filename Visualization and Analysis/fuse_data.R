library(RColorBrewer)

#
# run each of these NLP / gesture separately them stack them into metricsAll
# see bottom of this code for more description, where saving takes place
#

fuseSimulations = function(folder, pattern='*KNN*', nepochs, nsims) {
  simulations = c()
  for (sim in 1:nsims) {
    path = paste(folder,'/sim',sim,'/',sep='')
    print(paste("Looking in folder:", path))
    fls = list.files(path = path, pattern=pattern)
    for (i in 1:length(fls)) {
      thisSimulation = read.csv(paste(path,fls[i],sep=''))
      wndws = nrow(thisSimulation)
      dv = as.numeric(unlist(t(thisSimulation[,2:ncol(thisSimulation)]))) # first col is epoch #
      epoch = rep(1:nepochs,wndws)
      tix = c(); for(j in 1:wndws){tix=c(tix,rep(j,nepochs))}
      colnames(thisSimulation) = c('t',paste('epoch',1:nepochs,sep=''))
      simulations = rbind(simulations, data.frame(fl=fls[i],t=tix,sim=sim,epoch=epoch,dv=dv))
    }
  }
  return(simulations)
}

nepochs=100
nsims=10

#datasetName = 'nlp' # run this first
#res = fuseSimulations('data/sentence_50epochs',pattern='*KNN*', nepochs=nepochs, nsims=nsims)
#labels = c('anger','fear','joy','love','sadness','surprise') # emotion

datasetName = 'gesture' # run this second, then see line 135
res = fuseSimulations('data/geature_50epochs',pattern='*KNN*', nepochs=nepochs, nsims=nsims)
labels = c('no gestures','body','head','hand','body-head','head-hand') # gesture

#datasetName = 'gesture_null' # run this first
#res = fuseSimulations('data/allepoch_gesture_reshuffle',pattern='*KNN*', nepochs=nepochs, nsims=nsims)
#labels = c('null','null','null','null','null','null') 

#####

res$class = gsub('processing curve sim_.*?_(.*?).csv','\\1',res$fl)
res$class = gsub('-','/',res$class)
res[1:5,]

for (i in 0:5) {
  res$class = gsub(paste('Class',i,sep=''),labels[i+1],res$class)
}
res[1:5,]

classes = unique(res$class)

metrics = c()

for (epoch in 1:nepochs) {
  print(epoch)
  for (cl in classes) {
    for (sim in 1:nsims) {
      tmp = res[res$class==cl&res$epoch==epoch&res$sim==sim,]
      v0 = smooth(tmp$dv)[1]
      vend = smooth(tmp$dv)[nrow(tmp)]
      vmax = max(smooth(tmp$dv))
      maxt = which.max(smooth(tmp$dv))/nrow(tmp)
      # linmod = lm(dv~poly(t,3),data=tmp)
      # theCoefs = coef(linmod)
      metrics = rbind(metrics, 
                         data.frame(epoch=epoch,sim=sim,cl=cl,
                                    v0=v0,vend=vend,vmax=vmax,maxt = maxt
                                    #t(theCoefs),poly.rsq=summary(linmod)$r.sq
                                    ))
    }
  }
}

#########################################
# Information processing and developmental trajectories
#########################################
base_palette <- brewer.pal(12, "Paired")
colors <- colorRampPalette(base_palette)(length(classes)) 

png(file = paste('plots/', datasetName, 'trajectories.png'), height = 6, width = 12, units = "in", res = 500)
par(mfrow=c(1,2), mar=c(4,4,2,1))

# Plot1 : information processing trajectory
frm1a = maxt~epoch+cl;frm1 = maxt~epoch 
frm2a = vend-v0~epoch+cl;frm2 = vend-v0~epoch

dv1a = aggregate(frm1a,data=metrics,mean) 
dv2a = aggregate(frm2a,data=metrics,mean)

for (clix in 1:length(classes)) {
  cl = classes[clix]
  tmp = metrics[metrics$cl==cl,]
  
  dv1 = aggregate(frm1,data=tmp,mean) 
  dv2 = aggregate(frm2,data=tmp,mean)
  
  if (clix==1) {
    plot(dv1[,2],dv2[,2],type='o',pch=16,col=colors[clix],main=paste(datasetName, '(information processing trajectory)'),
         #xlim=range(dv1a[,3]),ylim=range(dv2a[,3]),
         xlab='Time slice of performance maximum',ylab='Performance end - start', # information
         # xlab='Start performance',ylab='Maximum performance', # learning
         cex=seq(.2,2,length=max(tmp$epoch)),xlim=c(0,1),ylim=c(-0.1,1))
  } else {
    points(dv1[,2],dv2[,2],type='o',pch=16,col=colors[clix],
           cex=seq(.5,2,length=max(tmp$epoch)))
  }
  
}

#legend("topright",col=colors,lwd=2,classes,cex=0.75)

# Plot 2: developmental trajectory
frm1a = v0~epoch+cl;frm1 = v0~epoch 
frm2a = vmax~epoch+cl;frm2 = vmax~epoch

dv1a = aggregate(frm1a,data=metrics,mean) # all
dv2a = aggregate(frm2a,data=metrics,mean)

for (clix in 1:length(classes)) {
  cl = classes[clix]
  tmp = metrics[metrics$cl==cl,]
  
  dv1 = aggregate(frm1,data=tmp,mean)
  dv2 = aggregate(frm2,data=tmp,mean)
  
  if (clix==1) {
    plot(dv1[,2],dv2[,2],type='o',pch=16,col=colors[clix],main=paste(datasetName, '(developmental trajectory)'),
         #xlim=range(dv1a[,3]),ylim=range(dv2a[,3]),
         # xlab='Time slice of performance maximum',ylab='Performance end - start', # information
         xlab='Start performance',ylab='Maximum performance', # learning
         cex=seq(.2,2,length=max(tmp$epoch)),xlim=c(0.5,1),ylim=c(0.5,1))
  } else {
    points(dv1[,2],dv2[,2],type='o',pch=16,col=colors[clix],
           cex=seq(.5,2,length=max(tmp$epoch)))
  }
  
}

legend("bottomright",col=colors,lwd=2,classes,cex=.75 )
dev.off()

metrics$training = datasetName

metricsAll = metrics

#### run NLP first; then gesture... then this:
metricsAll = rbind(metricsAll, metrics)

save(file='metricsAll_gesture_epoch100.Rd',metricsAll)
