# Cox__grouping_preprocessing-Survival analysis optimal cutoff
Author: Huang,shu-jing
Date:2023/11/23

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
```


## Fuction `Categorize_variables` :Cox model Categorize variables function
```r
Categorize_variables = function(gene){

    res.cut <- try(surv_cutpoint(
            datafile,
            time = "OS.time",
            event = "OS",
            variables = gene,
            minprop = 0.3,
            progressbar=TRUE),
            silent = TRUE
            )
            if(all(class(res.cut) == "try-error")){
            return(res.cat=NA)}
            res.cat <- surv_categorize(res.cut)
            res.all <- list(res.cut$ cutpoint,res.cat)
            return(res.all)
        }
```
## [Early stage] Call the function `Categorize_variables`
COAD
```r
#read data
datafile<-read.csv("SampleGrouping/Data/COADearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/COADearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))

#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/COADearlystageCoxgroup.rds")
```
ESCA
```r
#read data
datafile<-read.csv("SampleGrouping/Data/ESCAearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/ESCAearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/ESCAearlystageCoxgroup.rds")
```
KIRC
```r
#read data
datafile<-read.csv("SampleGrouping/Data/KIRCearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/KIRCearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/KIRCearlystageCoxgroup.rds")
```
KIRP
```r
#read data
datafile<-read.csv("SampleGrouping/Data/KIRPearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/KIRPearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/KIRPearlystageCoxgroup.rds")
```
LIHC
```r
#read data
datafile<-read.csv("SampleGrouping/Data/LIHCearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/LIHCearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/LIHCearlystageCoxgroup.rds")
```
LUSC
```r
#read data
datafile<-read.csv("SampleGrouping/Data/LUSCearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/LUSCearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/LUSCearlystageCoxgroup.rds")
```
LUAD
```r
#read data
datafile<-read.csv("SampleGrouping/Data/LUADearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/LUADearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/LUADearlystageCoxgroup.rds")
```
STAD
```r
#read data
datafile<-read.csv("SampleGrouping/Data/STADearlystageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/STADearlystage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/STADearlystageCoxgroup.rds")
```


## [Late stage] Call the function `Categorize_variables`
COAD
```r
#read data
datafile<-read.csv("SampleGrouping/Data/COADlatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/COADlatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
(input_gene = names(datafile)[grep("ENSG",colnames(datafile))])
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/COADlatestageCoxgroup.rds")
```
ESCA
```r
#read data
datafile<-read.csv("SampleGrouping/Data/ESCAlatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/ESCAlatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG",colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
input_gene = names(datafile)[grep("ENSG",colnames(datafile))]
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/ESCAlatestageCoxgroup.rds")
```
KIRC
```r
#read data
datafile<-read.csv("SampleGrouping/Data/KIRClatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/KIRClatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG", colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
input_gene = names(datafile)[grep("ENSG",colnames(datafile))]
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/KIRClatestageCoxgroup.rds")
```
KIRP
```r
#read data
datafile<-read.csv("SampleGrouping/Data/KIRPlatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/KIRPlatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG", colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
input_gene = names(datafile)[grep("ENSG",colnames(datafile))]
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/KIRPlatestageCoxgroup.rds")
```
LIHC
```r
#read data
datafile<-read.csv("SampleGrouping/Data/LIHClatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/LIHClatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG", colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
input_gene = names(datafile)[grep("ENSG",colnames(datafile))]
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/LIHClatestageCoxgroup.rds")
```
LUSC
```r
#read data
datafile<-read.csv("SampleGrouping/Data/LUSClatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/LUSClatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG", colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
input_gene = names(datafile)[grep("ENSG",colnames(datafile))]
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res, file = "SampleGrouping/Output/LUSClatestageCoxgroup.rds")
```
LUAD
```r
#read data
datafile<-read.csv("SampleGrouping/Data/LUADlatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/LUADlatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG", colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
input_gene = names(datafile)[grep("ENSG",colnames(datafile))]
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res,file = "SampleGrouping/Output/LUADlatestageCoxgroup.rds")
```
STAD
```r
#read data
datafile<-read.csv("SampleGrouping/Data/STADlatestageExprandClin.csv",sep=",",row.names=1)
comparefile<-read.csv("SampleGrouping/Data/STADlatestage_surv_diffgene.csv",sep=",",row.names=1)
#datafilter
clin_col= names(datafile)[!grepl("ENSG", colnames(datafile))]
datafile<-datafile %>%  select(c(all_of(clin_col),rownames(comparefile)))
#execute the function
input_gene = names(datafile)[grep("ENSG",colnames(datafile))]
test_res = lapply(1:length(input_gene), function(idx){
return(Categorize_variables(input_gene[idx]))
})
# save file
saveRDS(test_res,file = "SampleGrouping/Output/STADlatestageCoxgroup.rds")
```
