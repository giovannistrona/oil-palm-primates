####R code for graphs



###plot legend for figure 2
a<-read.csv('./results/comparison_aoc_low_int_high.csv',header=T)
a<-a[2:10,]
a[,3:5]<-a[,3:5]/10000
pdf('./results/legend_figure_2.pdf',height=6,width=6)
par(mgp=c(1.20,0.0,0))
plot(0,0,col='white',xlim=c(0,11),ylim=c(0,11),xaxt='n',yaxt='n',xlab='primate vulnerability',ylab='oil palm suitability',cex.lab=3)
cols=c('#006d2c','#31a354','#bae4b3','#2b8cbe','#a6bddb','#ece7f2','#a50f15','#de2d26','#fcae91')
names=c('HL','ML','LL','HM','MM','LM','HH','MH','LH')

sc<-1
for (row in c(0,4,8)){
	for (col in c(8,4,0)){
		rect(row,col,row+3,col+3,col=cols[sc],border=cols[sc])
		q=round(100*a[as.character(a$category)==names[sc],4][1]/sum(a[,4]),1)
		text(row+1.45,col+1.45,q,cex=3.5,col='white')
		text(row+1.5,col+1.5,q,cex=3.5)
		sc<-sc+1}}
		

dev.off()

###FIG 3

pdf('./results/fig3.pdf',width=8,height=6)

par(mgp=c(2.00,0.75,0))
a<-read.csv('./results/comparison_aoc_low_int_high.csv',header=T)
a<-a[2:10,]
a[,3:5]<-a[,3:5]/10000
a <- a[order(-a[,3]),] 
#mean_vals<-rowMeans(a[,3:5])
max_vals<-apply(a[,3:5],1,max)


b<-barplot(rbind(a$km2_low,a$km2_intermediate,a$km2_high),col=rbind(as.character(a$color),as.character(a$color),as.character(a$color)),beside=T,names=a$category, xlab='oil palm suitability - primate vulnerability',ylab="extent (Mha)",cex.axis=1.5,cex.lab=1.5,cex=1.5)
text(b[2,],max_vals+2,round(a$km2_intermediate,1))

dev.off()







####
###Figure 4
res_acc<-read.csv('./results/res_accessibility.csv',header=F)
res_suit<-read.csv('./results/res_suitability.csv',header=F)
res_carbon<-read.csv('./results/res_carbon.csv',header=F)
res_best<-read.csv('./results/res_vulnerability.csv',header=F)
res_rand<-read.csv('./results/res_random.csv',header=F)

res_acc<-res_acc[res_acc[,1]<=550100,]
res_suit<-res_suit[res_suit[,1]<=550100,]
res_carbon<-res_carbon[res_carbon[,1]<=550100,]
res_best<-res_best[res_best[,1]<=550100,]
res_rand<-res_rand[res_rand[,1]<=550100,]



res_acc[,1]<-res_acc[,1]/10000
res_carbon[,1]<-res_carbon[,1]/10000
res_rand[,1]<-res_rand[,1]/10000
res_suit[,1]<-res_suit[,1]/10000
res_best[,1]<-res_best[,1]/10000


#XXXXX
res_acc[,2]<-res_acc[,2]/10000
res_carbon[,2]<-res_carbon[,2]/10000
res_rand[,2]<-res_rand[,2]/10000
res_suit[,2]<-res_suit[,2]/10000
res_best[,2]<-res_best[,2]/10000


res_acc_<-aggregate(res_acc[,2]~res_acc[,1],FUN='mean')
res_carbon_<-aggregate(res_carbon[,2]~res_carbon[,1],FUN='mean')
res_rand_<-aggregate(res_rand[,2]~res_rand[,1],FUN='mean')
res_suit_<-aggregate(res_suit[,2]~res_suit[,1],FUN='mean')
res_best_<-aggregate(res_best[,2]~res_best[,1],FUN='mean')


res_acc_min<-aggregate(res_acc[,2]~res_acc[,1],FUN='min')
res_carbon_min<-aggregate(res_carbon[,2]~res_carbon[,1],FUN='min')
res_rand_min<-aggregate(res_rand[,2]~res_rand[,1],FUN='min')
res_suit_min<-aggregate(res_suit[,2]~res_suit[,1],FUN='min')
res_best_min<-aggregate(res_best[,2]~res_best[,1],FUN='min')


res_acc_max<-aggregate(res_acc[,2]~res_acc[,1],FUN='max')
res_carbon_max<-aggregate(res_carbon[,2]~res_carbon[,1],FUN='max')
res_rand_max<-aggregate(res_rand[,2]~res_rand[,1],FUN='max')
res_suit_max<-aggregate(res_suit[,2]~res_suit[,1],FUN='max')
res_best_max<-aggregate(res_best[,2]~res_best[,1],FUN='max')


res_acc_2<-aggregate(res_acc[,4]~res_acc[,1],FUN='mean')
res_carbon_2<-aggregate(res_carbon[,4]~res_carbon[,1],FUN='mean')
res_rand_2<-aggregate(res_rand[,4]~res_rand[,1],FUN='mean')
res_suit_2<-aggregate(res_suit[,4]~res_suit[,1],FUN='mean')
res_best_2<-aggregate(res_best[,4]~res_best[,1],FUN='mean')


res_acc_min2<-aggregate(res_acc[,4]~res_acc[,1],FUN='min')
res_carbon_min2<-aggregate(res_carbon[,4]~res_carbon[,1],FUN='min')
res_rand_min2<-aggregate(res_rand[,4]~res_rand[,1],FUN='min')
res_suit_min2<-aggregate(res_suit[,4]~res_suit[,1],FUN='min')
res_best_min2<-aggregate(res_best[,4]~res_best[,1],FUN='min')


res_acc_max2<-aggregate(res_acc[,4]~res_acc[,1],FUN='max')
res_carbon_max2<-aggregate(res_carbon[,4]~res_carbon[,1],FUN='max')
res_rand_max2<-aggregate(res_rand[,4]~res_rand[,1],FUN='max')
res_suit_max2<-aggregate(res_suit[,4]~res_suit[,1],FUN='max')
res_best_max2<-aggregate(res_best[,4]~res_best[,1],FUN='max')



my<-max(c(res_suit_max[,2],res_acc_max[,2],res_carbon_max[,2],res_best_max[,2],res_rand_max[,2]))

cols=c('#d95f02','#7570b3','#e7298a','#66a61e')		#http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=5
pdf('fig4.pdf',width=10,height=5)

par(mfrow=c(1,2))
par(mgp=c(2.00,0.75,0))
plot(res_acc_[,1],res_acc_[,2],ylim=c(0,my),type='l',col='white',lwd=2,xlab="oil palm expansion (Mha)",ylab="cumulative primate range loss (Mha)",las=1,cex.axis=1.2,cex.lab=1.2,main='a')

polygon(c(rev(res_suit_[,1]), res_suit_[,1]), c(rev(res_suit_min[,2]), (res_suit_max[,2])), col = adjustcolor(cols[1],alpha.f=0.5), border = NA)
polygon(c(rev(res_acc_[,1]), res_acc_[,1]), c(rev(res_acc_min[,2]), (res_acc_max[,2])), col = adjustcolor(cols[2],alpha.f=0.5), border = NA)
polygon(c(rev(res_carbon_[,1]), res_carbon_[,1]), c(rev(res_carbon_min[,2]), (res_carbon_max[,2])), col = adjustcolor(cols[3],alpha.f=0.5), border = NA)
polygon(c(rev(res_best_[,1]), res_best_[,1]), c(rev(res_best_min[,2]), (res_best_max[,2])), col = adjustcolor(cols[4],alpha.f=0.5), border = NA)
polygon(c(rev(res_rand_[,1]), res_rand_[,1]), c(rev(res_rand_min[,2]), (res_rand_max[,2])), col = adjustcolor("darkgrey",alpha.f=0.5), border = NA)

lines(res_suit_[,1],res_suit_[,2],col=cols[1],lwd=1)
lines(res_acc_[,1],res_acc_[,2],col=cols[2],lwd=1)
lines(res_carbon_[,1],res_carbon_[,2],col=cols[3],lwd=1)
lines(res_best_[,1],res_best_[,2],col=cols[4],lwd=1)
lines(res_rand_[,1],res_rand_[,2],col='black',lwd=1,lty=4)

demand<-c(22,44,26.5,53) #c(22,44,37,74)
cols_2<-c('grey17','grey17','grey17','grey17')
line_type<-c(5,5,5,5)

for (i in 1:4){
	mmm<-max(min(res_suit_[res_suit_[,1]>demand[i],2]),min(res_acc_[res_acc_[,1]>demand[i],2]),min(res_carbon_[res_carbon_[,1]>demand[i],2]),min(res_best_[res_best_[,1]>demand[i],2]),min(res_rand_[res_rand_[,1]>demand[i],2]))
	lines(x=c(demand[i],demand[i]),y=c(-100,mmm),lwd=0.8,lty=line_type[i],col=cols_2[i])
}

legend(0,800,c('suitability','accessibility','carbon','vulnerability','random'),col=c(cols[1],cols[2],cols[3],cols[4],'black'),lty=c(1,1,1,1,4),lwd=c(2,2,2,2,2),cex=0.6,bty = 'n')

my<-max(c(res_suit_max2[,2],res_acc_max2[,2],res_carbon_max2[,2],res_best_max2[,2],res_rand_max2[,2]))
plot(res_acc_2[,1],res_acc_2[,2],ylim=c(0,my),type='l',col='white',lwd=2,xlab="oil palm expansion (Mha)",ylab='affected primate species',las=1,cex.axis=1.2,cex.lab=1.2,main='b')


polygon(c(rev(res_suit_2[,1]), res_suit_2[,1]), c(rev(res_suit_min2[,2]), (res_suit_max2[,2])), col = adjustcolor(cols[1],alpha.f=0.5), border = NA)
polygon(c(rev(res_acc_2[,1]), res_acc_2[,1]), c(rev(res_acc_min2[,2]), (res_acc_max2[,2])), col = adjustcolor(cols[2],alpha.f=0.5), border = NA)
polygon(c(rev(res_carbon_2[,1]), res_carbon_2[,1]), c(rev(res_carbon_min2[,2]), (res_carbon_max2[,2])), col = adjustcolor(cols[3],alpha.f=0.5), border = NA)
polygon(c(rev(res_best_2[,1]), res_best_2[,1]), c(rev(res_best_min2[,2]), (res_best_max2[,2])), col = adjustcolor(cols[4],alpha.f=0.5), border = NA)
polygon(c(rev(res_rand_2[,1]), res_rand_2[,1]), c(rev(res_rand_min2[,2]), (res_rand_max2[,2])), col = adjustcolor("darkgrey",alpha.f=0.5), border = NA)

lines(res_suit_2[,1],res_suit_2[,2],col=cols[1],lwd=1)
lines(res_acc_2[,1],res_acc_2[,2],col=cols[2],lwd=1)
lines(res_carbon_2[,1],res_carbon_2[,2],col=cols[3],lwd=1)
lines(res_best_2[,1],res_best_2[,2],col=cols[4],lwd=1)
lines(res_rand_2[,1],res_rand_2[,2],col='black',lwd=1,lty=4)



for (i in 1:4){
	mmm<-max(min(res_suit_2[res_suit_2[,1]>demand[i],2]),min(res_acc_2[res_acc_2[,1]>demand[i],2]),min(res_carbon_2[res_carbon_2[,1]>demand[i],2]),min(res_best_2[res_best_2[,1]>demand[i],2]),min(res_rand_2[res_rand_2[,1]>demand[i],2]))
	lines(x=c(demand[i],demand[i]),y=c(-10,mmm),lwd=0.8,lty=line_type[i],col=cols_2[i])
}


legend(0,80,c('suitability','accessibility','carbon','vulnerability','random'),col=c(cols[1],cols[2],cols[3],cols[4],'black'),lty=c(1,1,1,1,4),lwd=c(2,2,2,2,2),cex=0.6,bty = 'n')

dev.off()




#####SUPPLEMENTARY FIGURES
a<-read.csv('./results/suitability_vs_vulnerability.csv',header=F)
pdf('figS1.pdf')
d<-a[a[,1]>0,]
boxplot(log(d[,2]+1)~round(d[,1]),outline=F,las=1,csx.axis=1.5,cex.lab=1.5,xlab='oil-palm suitability',ylab='primate vulnerability')
vmean<-aggregate(log(d[,2]+1)~round(d[,1]),FUN='mean')
points(vmean,pch=16,cex=2)
lines(vmean,pch=16,cex=2)
dev.off()







library(akima)
library(fields)
library(viridis)


pdf('./results/figS2.pdf')

x<-runif(1000,1,20)
y<-round(runif(1000,1,30))
z<-round(x*y)

image.plot(interp(x,y,log(z+1),duplicate='mean',linear=T,extrap=F,nx =100,ny=100),xlab='mean vulnerability',ylab='species richness',col=plasma(10),cex.lab=1.5,cex.axis=1.5,las=1)
dev.off()

####FIG S3


demand<-c(22,44,26.5,53) #c(22,44,37,74)
cols_2<-c('grey17','grey17','grey17','grey17')
line_type<-c(5,5,5,5)

res_vuln<-read.csv('./results/res_vulnerability.csv',header=F)
res_rand<-read.csv('./results/res_random.csv',header=F)


res_vuln<-res_vuln[res_vuln[,1]<=550100,]
res_rand<-res_rand[res_rand[,1]<=550100,]






res_vuln[,1]<-res_vuln[,1]/10000
res_rand[,1]<-res_rand[,1]/10000
res_vuln[,2]<-res_vuln[,2]/10000
res_rand[,2]<-res_rand[,2]/10000

res_rand_<-aggregate(res_rand[,2]~res_rand[,1],FUN='mean')
res_rand_min<-aggregate(res_rand[,2]~res_rand[,1],FUN='min')
res_rand_max<-aggregate(res_rand[,2]~res_rand[,1],FUN='max')

res_vuln_<-aggregate(res_vuln[,2]~res_vuln[,1],FUN='mean')
res_vuln_min<-aggregate(res_vuln[,2]~res_vuln[,1],FUN='min')
res_vuln_max<-aggregate(res_vuln[,2]~res_vuln[,1],FUN='max')




res_rand_2<-aggregate(res_rand[,4]~res_rand[,1],FUN='mean')
res_rand_min2<-aggregate(res_rand[,4]~res_rand[,1],FUN='min')
res_rand_max2<-aggregate(res_rand[,4]~res_rand[,1],FUN='max')

res_vuln_2<-aggregate(res_vuln[,4]~res_vuln[,1],FUN='mean')
res_vuln_min2<-aggregate(res_vuln[,4]~res_vuln[,1],FUN='min')
res_vuln_max2<-aggregate(res_vuln[,4]~res_vuln[,1],FUN='max')




models=c('suitability_accessibility','accessibility_suitability','carbon_accessibility','suitability_carbon','accessibility_carbon','carbon_suitability',
'suitability_accessibility_carbon','accessibility_suitability_carbon','carbon_accessibility_suitability','suitability_carbon_accessibility','accessibility_carbon_suitability','carbon_suitability_accessibility')

pdf('all_scenarios_2.pdf',width=12,height=16)
par(mfrow=c(4,3))
for (mod in models){

	res_mod<-read.csv(paste0('./results/res_',mod,'.csv'),header=F)
	res_mod<-res_mod[res_mod[,1]<=550100,]
	res_mod[,1]<-res_mod[,1]/10000
	res_mod[,2]<-res_mod[,2]/10000

	res_mod_<-aggregate(res_mod[,2]~res_mod[,1],FUN='mean')
	res_mod_min<-aggregate(res_mod[,2]~res_mod[,1],FUN='min')
	res_mod_max<-aggregate(res_mod[,2]~res_mod[,1],FUN='max')

	my<-max(c(res_mod_max[,2],res_rand_max[,2],res_vuln_max[,2]))

	par(mgp=c(2.00,0.75,0))
	
	plot(res_mod_[,1],res_mod_[,2],ylim=c(0,my),type='l',col='white',lwd=2,xlab="",ylab="",las=1,cex.axis=1.8,cex.lab=1.8,cex.main=2,main=gsub('_',' + ',mod))

polygon(c(rev(res_mod_[,1]), res_mod_[,1]), c(rev(res_mod_min[,2]), (res_mod_max[,2])), col = adjustcolor("steelblue",alpha.f=0.5), border = NA)
polygon(c(rev(res_rand_[,1]), res_rand_[,1]), c(rev(res_rand_min[,2]), (res_rand_max[,2])), col = adjustcolor("darkgrey",alpha.f=0.5), border = NA)
polygon(c(rev(res_vuln_[,1]), res_vuln_[,1]), c(rev(res_vuln_min[,2]), (res_vuln_max[,2])), col = adjustcolor("darkred",alpha.f=0.5), border = NA)

lines(res_mod_[,1],res_mod_[,2],col='steelblue',lwd=1)
lines(res_rand_[,1],res_rand_[,2],col='black',lwd=1,lty=4)
lines(res_vuln_[,1],res_vuln_[,2],col='red',lwd=1)




for (i in 1:4){
	mmm<-max(min(res_mod_[res_mod_[,1]>demand[i],2]),min(res_rand_[res_rand_[,1]>demand[i],2]),min(res_vuln_[res_vuln_[,1]>demand[i],2]))
	lines(x=c(demand[i],demand[i]),y=c(-10,mmm),lwd=0.8,lty=line_type[i],col=cols_2[i])
}





}



par(mfrow=c(4,3))
for (mod in models){
	res_mod<-read.csv(paste0('./results/res_',mod,'.csv'),header=F)
	res_mod<-res_mod[res_mod[,1]<=550100,]
	res_mod[,1]<-res_mod[,1]/10000
	res_mod_2<-aggregate(res_mod[,4]~res_mod[,1],FUN='mean')
	res_mod_min2<-aggregate(res_mod[,4]~res_mod[,1],FUN='min')
	res_mod_max2<-aggregate(res_mod[,4]~res_mod[,1],FUN='max')

	par(mgp=c(2.00,0.75,0))

	my<-max(c(res_mod_max2[,2],res_rand_max2[,2],res_vuln_max2[,2]))

plot(res_mod_2[,1],res_mod_2[,2],ylim=c(0,my),type='l',col='white',lwd=2,xlab='',ylab='',las=1,cex.axis=1.8,cex.lab=1.8,cex.main=2,main=gsub('_',' + ',mod))

polygon(c(rev(res_mod_2[,1]), res_mod_2[,1]), c(rev(res_mod_min2[,2]), (res_mod_max2[,2])), col = adjustcolor("steelblue",alpha.f=0.5), border = NA)
polygon(c(rev(res_rand_2[,1]), res_rand_2[,1]), c(rev(res_rand_min2[,2]), (res_rand_max2[,2])), col = adjustcolor("darkgrey",alpha.f=0.5), border = NA)
polygon(c(rev(res_vuln_2[,1]), res_vuln_2[,1]), c(rev(res_vuln_min2[,2]), (res_vuln_max2[,2])), col = adjustcolor("darkred",alpha.f=0.5), border = NA)

lines(res_mod_2[,1],res_mod_2[,2],col='steelblue',lwd=1)
lines(res_rand_2[,1],res_rand_2[,2],col='black',lwd=1,lty=4)
lines(res_vuln_2[,1],res_vuln_2[,2],col='red',lwd=1)

for (i in 1:4){
	mmm<-max(min(res_mod_2[res_mod_2[,1]>demand[i],2]),min(res_rand_2[res_rand_2[,1]>demand[i],2]),min(res_vuln_2[res_vuln_2[,1]>demand[i],2]))
	lines(x=c(demand[i],demand[i]),y=c(-10,mmm),lwd=0.8,lty=line_type[i],col=cols_2[i])
}






}


dev.off()




#############Fig S5

demand<-c(22,44,26.5,53) 
#demand<-c(22,44,37,74)
cols_2<-c('grey17','grey17','grey17','grey17')
line_type<-c(5,5,5,5)



res_suit<-read.csv('./results/res_suitability.csv',header=F)
res_best<-read.csv('./results/res_vulnerability.csv',header=F)
res_opt<-read.csv('./results/res_optimization.csv',header=F)


res_suit<-res_suit[res_suit[,1]<=550100,]
res_best<-res_best[res_best[,1]<=550100,]
res_opt<-res_opt[res_opt[,1]<=550100,]



res_suit[,1]<-res_suit[,1]/10000
res_best[,1]<-res_best[,1]/10000
res_opt[,1]<-res_opt[,1]/10000



res_suit[,2]<-res_suit[,2]/10000
res_best[,2]<-res_best[,2]/10000
res_opt[,2]<-res_opt[,2]/10000


res_suit_<-aggregate(res_suit[,2]~res_suit[,1],FUN='mean')
res_best_<-aggregate(res_best[,2]~res_best[,1],FUN='mean')
res_opt_<-aggregate(res_opt[,2]~res_opt[,1],FUN='mean')


res_suit_min<-aggregate(res_suit[,2]~res_suit[,1],FUN='min')
res_best_min<-aggregate(res_best[,2]~res_best[,1],FUN='min')
res_opt_min<-aggregate(res_opt[,2]~res_opt[,1],FUN='min')


res_suit_max<-aggregate(res_suit[,2]~res_suit[,1],FUN='max')
res_best_max<-aggregate(res_best[,2]~res_best[,1],FUN='max')
res_opt_max<-aggregate(res_opt[,2]~res_opt[,1],FUN='max')


res_suit_2<-aggregate(res_suit[,4]~res_suit[,1],FUN='mean')
res_best_2<-aggregate(res_best[,4]~res_best[,1],FUN='mean')
res_opt_2<-aggregate(res_opt[,4]~res_opt[,1],FUN='mean')


res_suit_min2<-aggregate(res_suit[,4]~res_suit[,1],FUN='min')
res_best_min2<-aggregate(res_best[,4]~res_best[,1],FUN='min')
res_opt_min2<-aggregate(res_opt[,4]~res_opt[,1],FUN='min')


res_suit_max2<-aggregate(res_suit[,4]~res_suit[,1],FUN='max')
res_best_max2<-aggregate(res_best[,4]~res_best[,1],FUN='max')
res_opt_max2<-aggregate(res_opt[,4]~res_opt[,1],FUN='max')




cols=c('#d95f02','#7570b3','#e7298a','#66a61e','darkblue')
pdf('figS5.pdf',width=10,height=5)

par(mfrow=c(1,2))
par(mgp=c(2.00,0.75,0))
plot(res_suit_[,1],res_suit_[,2],type='l',col='white',lwd=2,xlab="oil palm expansion (Mha)",ylab="average primate range loss (Mha)",las=1,cex.axis=1.2,cex.lab=1.2,main='a')

polygon(c(rev(res_suit_[,1]), res_suit_[,1]), c(rev(res_suit_min[,2]), (res_suit_max[,2])), col = adjustcolor(cols[1],alpha.f=0.5), border = NA)
polygon(c(rev(res_best_[,1]), res_best_[,1]), c(rev(res_best_min[,2]), (res_best_max[,2])), col = adjustcolor(cols[4],alpha.f=0.5), border = NA)
polygon(c(rev(res_opt_[,1]), res_opt_[,1]), c(rev(res_opt_min[,2]), (res_opt_max[,2])), col = adjustcolor(cols[5],alpha.f=0.5), border = NA)

lines(res_suit_[,1],res_suit_[,2],col=cols[1],lwd=1)
lines(res_best_[,1],res_best_[,2],col=cols[4],lwd=1)
lines(res_opt_[,1],res_opt_[,2],col=cols[5],lwd=1)


for (i in 1:4){
	mmm<-max(min(res_suit_[res_suit_[,1]>demand[i],2]),min(res_best_[res_best_[,1]>demand[i],2]),min(res_opt_[res_opt_[,1]>demand[i],2]))
	lines(x=c(demand[i],demand[i]),y=c(-10,mmm),lwd=0.8,lty=line_type[i],col=cols_2[i])
}




legend(0,400,c('suitability','vulnerability','optimization'),col=c(cols[1],cols[4],cols[5]),lty=c(1,1,1),lwd=2,cex=0.8,bty = 'n')

plot(res_opt_2[,1],res_opt_2[,2],type='l',col='white',lwd=2,xlab="oil palm expansion (Mha)",ylab='affected primate species',las=1,cex.axis=1.2,cex.lab=1.2,main='b')


polygon(c(rev(res_suit_2[,1]), res_suit_2[,1]), c(rev(res_suit_min2[,2]), (res_suit_max2[,2])), col = adjustcolor(cols[1],alpha.f=0.5), border = NA)
polygon(c(rev(res_best_2[,1]), res_best_2[,1]), c(rev(res_best_min2[,2]), (res_best_max2[,2])), col = adjustcolor(cols[4],alpha.f=0.5), border = NA)
polygon(c(rev(res_opt_2[,1]), res_opt_2[,1]), c(rev(res_opt_min2[,2]), (res_opt_max2[,2])), col = adjustcolor(cols[5],alpha.f=0.5), border = NA)

lines(res_suit_2[,1],res_suit_2[,2],col=cols[1],lwd=1)
lines(res_best_2[,1],res_best_2[,2],col=cols[4],lwd=1)
lines(res_opt_2[,1],res_opt_2[,2],col=cols[5],lwd=1)


for (i in 1:4){
	mmm<-max(min(res_suit_2[res_suit_2[,1]>demand[i],2]),min(res_best_2[res_best_2[,1]>demand[i],2]),min(res_opt_2[res_opt_2[,1]>demand[i],2]))
	lines(x=c(demand[i],demand[i]),y=c(-10,mmm),lwd=0.8,lty=line_type[i],col=cols_2[i])
}



legend(0,30,c('suitability','vulnerability','optimization'),col=c(cols[1],cols[4],cols[5]),lty=c(1,1,1),lwd=2,cex=0.8,bty = 'n')

dev.off()


####FIGURE S7
a<-read.csv('./results/sensitivity_vuln_index.csv',header=F)
pdf('figS7.pdf',width=5,height=5)
plot(a[,1]*100,a[,2]*100,type='l',xlab = 'random reduction in IUCN ranges (%)',ylab='cells changing cumulative vulnerability class (%)',las=1,cex.axis=1.2,cex.lab=1.2)
dev.off()


