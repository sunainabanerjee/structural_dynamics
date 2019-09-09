library(ggplot2)
x <- read.csv('tyr_sasa.csv', header=T)
x <- x[1:15000,]
g <- ggplot(x) + 
     geom_point(aes(actual, predicted), color = "darkorange2", pch = ".") + 
     theme_bw() + 
     ggtitle('TYROSINE') +
     xlim(c(0,220))+ 
     ylim(c(0,220)) + 
     xlab(expression('FreeSASA estimate \uc5'^'2')) + 
     ylab(expression('CG SASA estimate \uc5'^'2')) + 
     geom_smooth(aes(actual, predicted), method='lm', size = 0.5, color= "dark magenta") + 
     theme(title=element_text(hjust=0.5)) 

print(g)