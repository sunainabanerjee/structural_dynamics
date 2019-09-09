library(ggplot2)
library(ggsci)
x <- read.csv("/home/sumanta/Project/structural_dynamics/coarsegrained/martini/scripts/run2/wild_concat/2dproj.csv", header=T)
g<-ggplot(x) + 
    geom_point(aes(x,y, color=factor(tag), alpha=sample), shape=20, size= 3) + 
    theme_classic() + scale_color_npg() + theme(legend.position="none") + 
    xlim(-20,25) + ylim(-20,20) + 
    xlab('PC-1')+ ylab('PC-2') + 
    theme(axis.title = element_text(size = 18)) + 
    theme(axis.text = element_text(size=15))

#y <- read.csv("/home/sumanta/Project/structural_dynamics/coarsegrained/martini/scripts/state_space_explore/A204V/2d_proj.csv", header=T)
#ggplot(y) + geom_point(aes(pc1,pc2, color=factor(run)), pch=20, alpha=0.6, scale_color_npg()) + theme_classic()
