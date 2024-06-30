library(ggplot2)
library(dplyr)
library(gridExtra)
library(viridis)
library(RColorBrewer)

#
# Run fuse_data.R first
#

#
# Ribbon plots used in paper
#
doRibbon = function(data, ylim=c(0,1), lab='', palette='') {
  data_summary <- data %>%
    group_by(epoch, cl) %>%
    summarize(mean_dv = mean(dv),
              se_dv = sd(dv)/sqrt(n()))
  
  print(data_summary[data_summary$epoch==1|data_summary$epoch==20,],n=30)
  
  p <- ggplot(data_summary, aes(x = epoch, y = mean_dv, group = cl, fill = cl, color = cl)) +
    geom_ribbon(aes(ymin = mean_dv - se_dv, ymax = mean_dv + se_dv), alpha = 0.2) +
    geom_line() + coord_cartesian(ylim=ylim) +
    labs(x = "Epoch", y = "", title = lab, fill = 'Classification', color = 'Classification') +
    theme_minimal() +
    coord_cartesian(ylim=ylim)
  
  if (palette == 'viridis') {
    p <- p + scale_fill_viridis(discrete = TRUE) + scale_color_viridis(discrete = TRUE)
  } else if (palette == 'brewer') {
    p <- p + scale_fill_brewer(palette = "Set1") + scale_color_brewer(palette = "Set1")
  } else if (nchar(palette)>0) {
    p <- p + scale_fill_manual(values = palette) + scale_color_manual(values = palette)
  }
  
  return(p)
}

library(lme4)

load('metricsAll.Rd') # see fuse_data.R

nlp = metricsAll[metricsAll$training=='NLP',]
gesture = metricsAll[metricsAll$training=='gesture',]

metricsAll[1,]
table(metricsAll$training)

###
###
###
### max
###
###
###

nlp$dv = nlp$vmax
p_n = doRibbon(nlp,ylim=c(0,1),lab='Max performance (NLP)')

gesture$dv = gesture$vmax
p_g = doRibbon(gesture,ylim=c(0,1),lab='Max performance (gesture)',palette='viridis')

grid.arrange(p_n, p_g, ncol = 2)

summary(lm(dv~as.factor(training),data=rbind(nlp,gesture)))

summary(lm(dv~epoch*as.factor(cl),data=nlp))
summary(lm(dv~epoch*as.factor(cl),data=gesture))

summary(lm(dv~as.factor(cl),data=nlp))
summary(lm(dv~as.factor(cl),data=gesture))

anova(lm(dv~epoch,data=nlp),lm(dv~epoch*as.factor(cl),data=nlp))
anova(lm(dv~epoch,data=gesture),lm(dv~epoch*as.factor(cl),data=gesture))

summary(lm(dv~epoch*as.factor(cl),data=nlp))$r.squared-summary(lm(dv~epoch,data=nlp))$r.squared
summary(lm(dv~epoch*as.factor(cl),data=gesture))$r.squared-summary(lm(dv~epoch,data=gesture))$r.squared

###
###
###
### start
###
###
###

nlp$dv = nlp$v0
p_n = doRibbon(nlp,ylim=c(0,1),lab='Perf. at start (NLP)')

gesture$dv = gesture$v0
p_g = doRibbon(gesture,ylim=c(0,1),lab='Perf. at start (gesture)',palette='viridis')

grid.arrange(p_n, p_g, ncol = 2)

summary(lm(dv~as.factor(training),data=rbind(nlp,gesture)))
summary(lm(dv~epoch:as.factor(training),data=rbind(nlp,gesture)))

anova(lm(dv~as.factor(training),data=rbind(nlp,gesture)),
      lm(dv~as.factor(training)*epoch,data=rbind(nlp,gesture)))

summary(lm(dv~as.factor(training),data=rbind(nlp,gesture)))$r.squared - summary(lm(dv~as.factor(training)*epoch,data=rbind(nlp,gesture)))$r.squared

summary(lm(dv~as.factor(cl),data=nlp))
summary(lm(dv~as.factor(cl),data=gesture))

anova(lm(dv~epoch,data=nlp),lm(dv~epoch*as.factor(cl),data=nlp))
anova(lm(dv~epoch,data=gesture),lm(dv~epoch*as.factor(cl),data=gesture))

summary(lm(dv~epoch*as.factor(cl),data=nlp))$r.squared-summary(lm(dv~epoch,data=nlp))$r.squared
summary(lm(dv~epoch*as.factor(cl),data=gesture))$r.squared-summary(lm(dv~epoch,data=gesture))$r.squared

###
###
###
### end-start
###
###
###

nlp$dv = nlp$vend - nlp$v0
p_n = doRibbon(nlp,ylim=c(-0.1,0.5),lab='End - start (NLP)')

gesture$dv = gesture$vend - gesture$v0
p_g = doRibbon(gesture,ylim=c(-.1,0.5),lab='End - start (gesture)',palette='viridis')

grid.arrange(p_n, p_g, ncol = 2)

###
###
###
### tmax
###
###
###

nlp$dv = nlp$maxt
p_n = doRibbon(nlp,ylim=c(0,1),lab='Time at max (NLP)')

gesture$dv = gesture$maxt
p_g = doRibbon(gesture,ylim=c(0,1),lab='Time at max (gesture)',palette='viridis')

grid.arrange(p_n, p_g, ncol = 2)

summary(lm(dv~as.factor(cl),data=nlp))
summary(lm(dv~as.factor(cl),data=gesture))
summary(lm(dv~as.factor(training),data=rbind(nlp,gesture)))

###
###
###
### end - start
###
###
###

nlp$dv = nlp$vend-nlp$v0
p_n = doRibbon(nlp,ylim=c(-0.3,.2),'End - start (NLP)')

gesture$dv = gesture$vend-gesture$v0
p_g = doRibbon(gesture,ylim=c(-0.3,.2),'End - start (gesture)')

grid.arrange(p_n, p_g, ncol = 2)

summary(lm(dv~as.factor(cl),data=nlp))
summary(lm(dv~as.factor(cl),data=gesture))

summary(lm(dv~as.factor(training),data=rbind(nlp,gesture)))


summary(lm(dv~epoch*as.factor(cl),data=nlp))
summary(lm(dv~epoch*as.factor(cl),data=gesture))

anova(lm(dv~epoch,data=nlp),lm(dv~epoch*as.factor(cl),data=nlp))
anova(lm(dv~epoch,data=gesture),lm(dv~epoch*as.factor(cl),data=gesture))

summary(lm(dv~epoch*as.factor(cl),data=nlp))$r.squared-summary(lm(dv~epoch,data=nlp))$r.squared
summary(lm(dv~epoch*as.factor(cl),data=gesture))$r.squared-summary(lm(dv~epoch,data=gesture))$r.squared






