library(ggplot2)
library(RColorBrewer)
x <- read.csv('/home/sumanta/PycharmProjects/structural_dynamics/scripts/md_features/md_results.csv', header=T)


for(r in unique(x$resid)){ mean(abs(x$actual[x$resid == r] - x$predicted[x$resid == r])) }
mad <- c()
for(r in unique(x$resid)){ mad <- append( mad, mean(abs(x$actual[x$resid == r] - x$predicted[x$resid == r]))) }
resname <- c()
for(r in unique(x$resid)){ resname <- append(resname, unique(as.character(x$resname[x$resid == r])) ) }
#data <- data.frame(mad=mad, resname=resname)
#for(r in unique(x$resid)){ resname <- append(resname, unique(as.character(x$resname[x$resid == r])) }
data <- data.frame(mad=mad, resname=resname, resid=unique(x$resid))


#sasa = data.frame(resname=c("ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his", "Ile", "Leu", "Lys", "Met", "Phe", "Pro", "Ser", "Thr", "Trp", "Tyr", "Val"), sasa=c(209, 335, 259, 258, 240, 286, 285, 185, 290, 273, 278, 303, 291, 311, 235, 223, 243, 350, 328, 250))

avg.sasa <- c()
for(r in unique(x$resid)){ avg.sasa <- append(avg.sasa, mean(x$actual[x$resid == r])) }
data$sasa <- avg.sasa
data$resname <- toupper(data$resname)

number_ticks <- function(n) {function(limits) pretty(limits, n)}
g <- ggplot(data, aes(x=resid,y=sasa)) + 
        geom_point(aes(color=factor(resname))) + 
        geom_errorbar(aes(ymin=sasa-mad/2, ymax=sasa+mad/2), 
        size=0.1,width=1) + 
        theme_bw() +  
        theme_classic() + 
        theme(legend.position = 'bottom', legend.title = element_blank()) +
        xlab('Residue position in HCV NS3 helicase (1A1V)') +
        ylab(expression('CG SASA estimate \uc5'^'2')) +
        scale_x_continuous(breaks=number_ticks(10)) +
        scale_y_continuous(breaks=number_ticks(10)) 
print (g)
  