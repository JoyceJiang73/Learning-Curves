library(car)
library(ggplot2)
library(dplyr)

#
# For our initial analysis of regression (reported qualitatively in paper)
#
getPoly = function(df, degree = 3) {
  for (i in 1:ncol(df)) {
    dat = df[,i]
    betas = as.numeric(coef(lm(dat~poly(1:length(dat),degree=degree))))
  }
  return(betas)
}

#
# Our chosen ribbon plot method shown in paper; NB: replaced with `doRibbon` in results.R
#
makeRibbonPlot <- function(simulations,p=NULL,addToPlot=FALSE,lineLabel='') {
  numSimulations = length(simulations[1,]) - 1
  numEpochs = length(simulations[,1])
  epochs = simulations[,1]
  results = simulations[,-1]
  means = apply(results,1,mean)
  ses = 2*apply(results,1,sd)/sqrt(numSimulations)
  data = data.frame(epoch=epochs,mean=means,se=ses)
  if (!addToPlot) {
    p = ggplot(data,aes(x=epoch,y=mean)) + geom_ribbon(aes(ymin=mean-se,ymax=mean+se),alpha=0.5) + ylim(0.5,1) + ylab('Performance') + xlab('Time slice')
  } else {
    p = p + geom_ribbon(mapping=aes(ymin=mean-se,ymax=mean+se),data=data,alpha=0.5)
  }
  if (nchar(lineLabel)>0) {
    p = p + geom_text(data=data.frame(epoch=epochs[1],mean=means[1],se=ses[1]),mapping=aes(x=epoch,y=mean,label=lineLabel),hjust=0,vjust=1)        
  } 
  return(p)
}

#
# No longer used; see sample_curves.R for replacement code
#
makeLinePlot = function(simulations,addToPlot=FALSE,lineLabel='',col='black',ylim=c(0,1), lwd=1, main='',ylab='y',xlab='x') {
  x = simulations[,1]
  y = rowMeans(simulations[,2:ncol(simulations)])
  if (!addToPlot) {
    plot(x,y,type='l',col=col,ylim=ylim,lwd=lwd,main=main,xlab=xlab,ylab=ylab)  
  } else {
    points(x,y,type='l',col=col,lwd=lwd)  
  }
}

#
# Old analysis that used only stimulus bin; deprecated, backing up here
#
runAnalysis_stimulus_bin = function(folder, pattern='*KNN*') {
  fls = list.files(path = folder, pattern=pattern)
  fls = paste(folder,fls,sep='')
  
  par(mfrow=c(2,2))
  
  betas = c()
  sdDf = c()
  cols = rainbow(length(fls))
  for (i in 1:length(fls)) {
    simulations = read.csv(fls[i],header=T)
    makeLinePlot(simulations, addToPlot = !(i==1),ylim=c(0.5,1), 
                 col=cols[i],lwd=3,main='Parwise Discrim. Time',xlab='Time slice',ylab='Performance')
    betas = rbind(betas, getPoly(simulations, 3))
    simulations$i = i
    sdDf = rbind(sdDf,simulations)
  }
  
  plot(betas[,2],betas[,1],col=cols,pch=16,xlab='Linear effect',ylab='Intercept',main='Polynomial Fits')
  # text(betas[,2],betas[,1],substr(fls,20,36),col=cols,font=2,cex=.75)
  
  dim(sdDf)
  tStats = c()
  for (t in 1:max(sdDf$X)) {
    tmp = sdDf[sdDf$X == t,]
    vals = rowMeans(tmp[,2:(ncol(tmp)-1)])
    tStats = rbind(tStats,data.frame(t=t,m=mean(vals),sd=sd(vals)))
  }
  plot(tStats$m+tStats$sd/sqrt(nrow(tStats)),main='Mean +/- SE',type='l',
       xlab='Time slice',ylab='Mean performance')
  points(tStats$m-tStats$sd/sqrt(nrow(tStats)),type='l')
  plot(tStats$sd/tStats$m,main='Coefficient of Variation',type='l',xlab='Time slice',ylab='SD_t/m_t')
  mtext(pattern, side = 3, line = -20, outer = TRUE, cex=.7)
  mtext(folder, side = 3, line = -21, outer = TRUE, cex=.5)
  return(tStats)
}

#
# Critical function that fuses the data for combination into metrics.Rd
#
fuseSimulations = function(folder, pattern='*KNN*', nepochs=20, nsims=25) {
  simulations = c()
  for (sim in 1:nsims) {
    path = paste(folder,'/sim',sim,'/',sep='')
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

#
# Bird's-eye view with heat map; not used in paper; backing up
#
plotHeatMap = function(x,y) {
  x = round(x*20)
  y = round(y*20)
  grid = matrix(0,20,20)
  for (i in 1:length(x)) {
    grid[x[i],y[i]] = grid[x[i],y[i]] + 1
  }
  print(grid)
  image(grid,col=heat.colors(20))
}

#
# For initial visualization of our 4 metrics; no longer used; can be used to look at start/start-end spaces; 
# chose ribbon plot for visualizing results instead (see doRibbon in results.R)
#
plotFilledEllipse = function(x, y, theCol='red', add=FALSE, xlim=c(-1,1),ylim=c(-1,1),xlab='x',ylab='y',main='') {
  dataEllipse(x, y, levels = c(0.1), fill=TRUE, fill.alpha=0.2, add=add,
              plot.points=FALSE, col=theCol, xlim=xlim,ylim=ylim,
              xlab=xlab, ylab=ylab, main=main)
}
