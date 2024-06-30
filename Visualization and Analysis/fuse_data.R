source('assorted_functions.R')

#
# run each of these NLP / gesture separately them stack them into metricsAll
# see bottom of this code for more description, where saving takes place
#

datasetName = 'NLP' # run this first
res = fuseSimulations('data/allepochs_nlp/',pattern='*KNN*')
labels = c('anger','fear','joy','love','sadness','surprise') # emotion

datasetName = 'gesture' # run this second, then see line 135
res = fuseSimulations('data/allepochs_gesture',pattern='*KNN*')
labels = c('no gestures','body','head','hand','body-head','head-hand movements') # gesture

#####

res$class = gsub('processing curve sim_.*?_(.*?).csv','\\1',res$fl)
res$class = gsub('-','/',res$class)
res[1:5,]

for (i in 0:5) {
  res$class = gsub(paste('Class',i,sep=''),labels[i+1],res$class)
}
res[1:5,]

classes = unique(res$class)
compares = unique(res$class)

metrics = c()

for (epoch in 1:20) {
  print(epoch)
  for (cl in classes) {
    for (sim in 1:25) {
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
#
# information flow
#
#########################################

pdf(file=paste(datasetName,'information.pdf'),height=6,width=6)
par(mar=c(4,4,2,1))

frm1a = maxt~epoch+cl;frm1 = maxt~epoch # information flow
frm2a = vend-v0~epoch+cl;frm2 = vend-v0~epoch

dv1a = aggregate(frm1a,data=metrics,mean) # all
dv2a = aggregate(frm2a,data=metrics,mean)

for (clix in 1:length(classes)) {
  cl = classes[clix]
  tmp = metrics[metrics$cl==cl,]
  
  dv1 = aggregate(frm1,data=tmp,mean) # information flow
  dv2 = aggregate(frm2,data=tmp,mean)
  
  if (clix==1) {
    plot(dv1[,2],dv2[,2],type='o',pch=16,col=rainbow(length(classes))[clix],main=datasetName,
         #xlim=range(dv1a[,3]),ylim=range(dv2a[,3]),
         xlab='Time slice of performance maximum',ylab='Performance end - start', # information
         # xlab='Start performance',ylab='Maximum performance', # learning
         cex=seq(.2,2,length=max(tmp$epoch)),xlim=c(0,1),ylim=c(-0.1,1))
  } else {
    points(dv1[,2],dv2[,2],type='o',pch=16,col=rainbow(length(classes))[clix],
           cex=seq(.5,2,length=max(tmp$epoch)))
  }
  
}

# legend("bottomright",col=rainbow(length(classes)),lwd=2,classes,cex=.5)

dev.off()

#########################################
#
# learning flow
#
#########################################

pdf(file=paste(datasetName,'learning.pdf'),height=6,width=6)
par(mar=c(4,4,2,1))

frm1a = v0~epoch+cl;frm1 = v0~epoch # learning flow
frm2a = vmax~epoch+cl;frm2 = vmax~epoch

dv1a = aggregate(frm1a,data=metrics,mean) # all
dv2a = aggregate(frm2a,data=metrics,mean)

for (clix in 1:length(classes)) {
  cl = classes[clix]
  tmp = metrics[metrics$cl==cl,]
  
  dv1 = aggregate(frm1,data=tmp,mean) # information flow
  dv2 = aggregate(frm2,data=tmp,mean)
  
  if (clix==1) {
    plot(dv1[,2],dv2[,2],type='o',pch=16,col=rainbow(length(classes))[clix],main=datasetName,
         #xlim=range(dv1a[,3]),ylim=range(dv2a[,3]),
         # xlab='Time slice of performance maximum',ylab='Performance end - start', # information
         xlab='Start performance',ylab='Maximum performance', # learning
         cex=seq(.2,2,length=max(tmp$epoch)),xlim=c(0,1),ylim=c(-0.1,1))
  } else {
    points(dv1[,2],dv2[,2],type='o',pch=16,col=rainbow(length(classes))[clix],
           cex=seq(.5,2,length=max(tmp$epoch)))
  }
  
}

legend("bottomright",col=rainbow(length(classes)),lwd=2,classes,cex=.5)

dev.off()

metrics$training = datasetName

metricsAll = metrics

#### run NLP first; then gesture... then this:
metricsAll = rbind(metricsAll, metrics)

# save(file='metricsAll.Rd',metricsAll)
