Cox_regression_survival_analysis
Author: Huang,shu-jing
Date:2023/11/24

## Environments
```shell
conda activate survival
cd /home/emily2835/stage_project_git3/SurvivalAnalysis_CoxRegression
```

## import package
```r
library(tidyverse)
library(tictoc)
library(survminer)
library(survival)
library(dplyr)
```

## Function `cox_regression` : cox_regression function
```r
cox_regression = function(first_index){
    tmp_formula = as.formula(paste0("Surv(OS.time, OS)~" , colnames(test_res[[first_index]][[2]])[3]))
    tryCatch({
        res.cox <- coxph(tmp_formula, data = data.frame(test_res[[first_index]][[2]]))
        return(summary(res.cox)$coefficients)
    }, error = function(e) {
        return(NA)
    })
}
```



## [Early stage] Call the function `cox_regression`
- 8 cancer type: COAD/ESCA/KIRC/KIRP/LIHC/LUAD/STAD/LUSC
COAD
```r
#import data
test_res<-readRDS("CoxRegression/Data/COADearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/COADearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/COADearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/COADearlystageCoxProggeneHR15and067P005.csv",sep=",")
```
ESCA
```r
#import data
test_res<-readRDS("CoxRegression/Data/ESCAearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/ESCAearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/ESCAearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/ESCAearlystageCoxProggeneHR15and067P005.csv",sep=",")
```
KIRC
```r
#import data
test_res<-readRDS("CoxRegression/Data/KIRCearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/KIRCearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRCearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRCearlystageCoxProggeneHR15and067P005.csv",sep=",")
```
KIRP
```r
#import data
test_res<-readRDS("CoxRegression/Data/KIRPearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/KIRPearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRPearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRPearlystageCoxProggeneHR15and067P005.csv",sep=",")
```
LIHC
```r
#import data
test_res<-readRDS("CoxRegression/Data/LIHCearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/LIHCearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LIHCearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LIHCearlystageCoxProggeneHR15and067P005.csv",sep=",")
```
LUAD
```r
#import data
test_res<-readRDS("CoxRegression/Data/LUADearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/LUADearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUADearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUADearlystageCoxProggeneHR15and067P005.csv",sep=",")
```
STAD
```r
#import data
test_res<-readRDS("CoxRegression/Data/STADearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/STADearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/STADearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/STADearlystageCoxProggeneHR15and067P005.csv",sep=",")
```
LUSC
```r
#import data
test_res<-readRDS("CoxRegression/Data/LUSCearlystageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/LUSCearlystageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox %>% filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUSCearlystageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox %>% filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUSCearlystageCoxProggeneHR15and067P005.csv",sep=",")
```

## [Late stage] Call the function `cox_regression`
- 8 cancer type: COAD/ESCA/KIRC/KIRP/LIHC/LUAD/STAD/LUSC
```r
COAD
#import data
test_res<-readRDS("CoxRegression/Data/COADlatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind,cox))
write.table(bindcox,"CoxRegression/Output/COADlatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/COADlatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/COADlatestageCoxProggeneHR15and067P005.csv",sep=",")
```
ESCA
```r
#import data
test_res<-readRDS("CoxRegression/Data/ESCAlatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox<-as.data.frame(do.call(rbind, cox))
write.table(bindcox,"CoxRegression/Output/ESCAlatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub<-bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/ESCAlatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub<-bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/ESCAlatestageCoxProggeneHR15and067P005.csv",sep=",")
```
KIRC
```r
#import data
test_res<-readRDS("CoxRegression/Data/KIRClatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox <- as.data.frame(do.call(rbind, cox))
write.table(bindcox,"CoxRegression/Output/KIRClatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub <- bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRClatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub <- bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRClatestageCoxProggeneHR15and067P005.csv",sep=",")
```
KIRP
```r
#import data
test_res<-readRDS("CoxRegression/Data/KIRPlatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox <- as.data.frame(do.call(rbind, cox))
write.table(bindcox,"CoxRegression/Output/KIRPlatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub <- bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRPlatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub <- bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/KIRPlatestageCoxProggeneHR15and067P005.csv",sep=",")
```
LIHC
```r
#import data
test_res<-readRDS("CoxRegression/Data/LIHClatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox <- as.data.frame(do.call(rbind, cox))
write.table(bindcox,"CoxRegression/Output/LIHClatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub <- bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LIHClatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub <- bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LIHClatestageCoxProggeneHR15and067P005.csv",sep=",")
```
LUAD
```r
#import data
test_res<-readRDS("CoxRegression/Data/LUADlatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox <- as.data.frame(do.call(rbind, cox))
write.table(bindcox,"CoxRegression/Output/LUADlatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub <- bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUADlatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub <- bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUADlatestageCoxProggeneHR15and067P005.csv",sep=",")
```
STAD
```r
#import data
test_res<-readRDS("CoxRegression/Data/STADlatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox <- as.data.frame(do.call(rbind, cox))
write.table(bindcox,"CoxRegression/Output/STADlatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub <- bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/STADlatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub <- bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/STADlatestageCoxProggeneHR15and067P005.csv",sep=",")
```
LUSC
```r
#import data
test_res<-readRDS("CoxRegression/Data/LUSClatestageCoxgroup.rds")
#execute the function
cox <- lapply(1:length(test_res), function(idx){
return(cox_regression(idx))})
bindcox <- as.data.frame(do.call(rbind, cox))
write.table(bindcox,"CoxRegression/Output/LUSClatestageCoxAllgene.csv",sep=",")
#remove the end of gene name "low"
row.names(bindcox)<-gsub("low","",row.names(bindcox))
#extract p-value < 0.05
bindcox_sub <- bindcox%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUSClatestageCoxProggeneP005.csv",sep=",")
#extract Hazard Ratio>1.5 or < 0.67 and p-value < 0.05
bindcox_sub <- bindcox%>%filter(`exp(coef)`>=1.5|`exp(coef)`<=0.67)%>%filter(`Pr(>|z|)`<0.05)
write.table(bindcox_sub,"CoxRegression/Output/LUSClatestageCoxProggeneHR15and067P005.csv",sep=",")
```


