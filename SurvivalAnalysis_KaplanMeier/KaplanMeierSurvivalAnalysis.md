wAuthor : Huang,Shu-jing
Date : 2023/10/05

- [0.1. Environment](#01-environment)
- [0.2. Import package](#02-import-package)
- [0.3. import script](#03-import-script)
- [Kaplan-Meier survival analysis](#kaplan-meier-survival-analysis)
  - [\[Early stage\] kaplan-meier survival analysis](#early-stage-kaplan-meier-survival-analysis)
  - [\[Late stage\] kaplan-meier survival analysis](#late-stage-kaplan-meier-survival-analysis)

## 0.1. Environment 
```shell
conda activate survival
cd /home/emily2835/stage_project_git3/SurvivalAnalysis_KaplanMeier
```

## 0.2. Import package 
```r
library(tidyverse)
library(magrittr)
library(survival)
library(survminer)
```

## 0.3. import script
**:point_right:import function**
- survival_FUN1
- survival_FUN3
- surv_pval_save
- surv_pval_save2

```r
source("survival_analysis_function_script.r",encoding = "utf-8")
```

## Kaplan-Meier survival analysis
### [Early stage] kaplan-meier survival analysis
 - Gene-minprop 0.3
 - 8 cancer type: LIHC/LUAD/STAD/COAD/ESCA/KIRC/KIRP/LUSC

LIHC
```
datafile<-read.csv("Data/LIHCearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/LIHCearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
LUAD
```
datafile<-read.csv("Data/LUADearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/LUADearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
STAD
```
datafile<-read.csv("Data/STADearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/STADearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
COAD
```
datafile<-read.csv("Data/COADearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/COADearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
ESCA
```
datafile<-read.csv("Data/ESCAearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/ESCAearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
KIRC
```
datafile<-read.csv("Data/KIRCearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/KIRCearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
KIRP
```
datafile<-read.csv("Data/KIRPearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/KIRPearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
LUSC
```
datafile<-read.csv("Data/LUSCearlystageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/LUSCearlystageExprandClin_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```


### [Late stage] kaplan-meier survival analysis
 - Gene-minprop 0.3
 - 8 cancer type: LIHC/LUAD/STAD/COAD/ESCA/KIRC/KIRP/LUSC

LIHC
```
datafile<-read.csv("Data/LIHClatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/LIHClatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
LUAD
```
datafile<-read.csv("Data/LUADlatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/LUADlatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
STAD
```
datafile<-read.csv("Data/STADlatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/STADlatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
COAD
```
datafile<-read.csv("Data/COADlatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/COADlatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
ESCA
```
datafile<-read.csv("Data/ESCAlatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/ESCAlatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
KIRC
```
datafile<-read.csv("Data/KIRClatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/KIRClatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```
KIRP
```
datafile<-read.csv("Data/KIRPlatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/KIRPlatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```

LUSC
```
datafile<-read.csv("Data/LUSClatestageExprandClin.csv",sep=",",row.names = 1)
savefile<-"Output/LUSClatestage_data"
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx)
{return(survival_FUN3(input_gene[idx]))
})

surv_pval_save2(test_res,savefile)
```


**[END]**
