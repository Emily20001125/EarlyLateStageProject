# Combine normalized expression data and clinical data
Author : Huang,Shu-jing.
Date : 2023/11/23

## Normalize and Batch Effect Remove
```r
conda activate basic_work
cd /home/emily2835/stage_project_git3/NormalizeandBatchEffectRemove
```
## Import package 
```r
library(tidyverse)
library(magrittr)
```
## Import data
```r
expr_data<-readRDS("TCGA_Early_LateStageSelected_UQnormalizedExpectedCount_Nov2023.rda")
clin_data<-read.csv("TCGAPanCancerExprandClin8cancertype.csv",stringsAsFactors = F)
```
```
> expr_data
   cancerType stage       sample_select data_UQ
   <chr>      <chr>       <list>        <list>
 1 COAD       early_stage <chr [150]>   <dbl [17,611 × 150]>
 2 ESCA       early_stage <chr [94]>    <dbl [18,201 × 94]>
 3 KIRC       early_stage <chr [321]>   <dbl [17,799 × 321]>
 4 KIRP       early_stage <chr [189]>   <dbl [17,530 × 189]>
 5 LIHC       early_stage <chr [252]>   <dbl [17,389 × 252]>
 6 LUAD       early_stage <chr [388]>   <dbl [17,824 × 388]>
 7 LUSC       early_stage <chr [397]>   <dbl [17,970 × 397]>
 8 STAD       early_stage <chr [172]>   <dbl [18,230 × 172]>
 9 COAD       late_stage  <chr [131]>   <dbl [17,611 × 131]>
10 ESCA       late_stage  <chr [87]>    <dbl [18,201 × 87]>
11 KIRC       late_stage  <chr [207]>   <dbl [17,799 × 207]>
12 KIRP       late_stage  <chr [96]>    <dbl [17,530 × 96]>
13 LIHC       late_stage  <chr [111]>   <dbl [17,389 × 111]>
14 LUAD       late_stage  <chr [112]>   <dbl [17,824 × 112]>
15 LUSC       late_stage  <chr [94]>    <dbl [17,970 × 94]>
16 STAD       late_stage  <chr [215]>   <dbl [18,230 × 215]>
```

## Combine expression data and clinical data
```r
# Transpose data_UQ 
expr_data$data_UQ<-lapply(expr_data$data_UQ,t)
# extract clinical data with not start with ENSG
clin_datasub<-clin_data %>% select(-starts_with("ENSG"))
# early stage 
for (i in 1:8){
    expr_data_sub<-expr_data$data_UQ[[i]]
    expr_data_sub <- as.data.frame(expr_data_sub)
    expr_data_sub <- expr_data_sub %>% mutate("sample" = rownames(expr_data_sub))
    # merge clinical data=sample expr_data_sub=rowname
    comtable <- merge(clin_datasub, expr_data_sub, by = "sample",all.y = TRUE)
    filename<-paste(expr_data$cancerType[i],"earlystageExprandClin.csv",sep="")
    write.csv(comtable,filename)
}
# late stage
for (i in 9:16){
    expr_data_sub<-expr_data$data_UQ[[i]]
    expr_data_sub <- as.data.frame(expr_data_sub)
    expr_data_sub <- expr_data_sub %>% mutate("sample" = rownames(expr_data_sub))
    # merge clinical data=sample expr_data_sub=rowname
    comtable <- merge(clin_datasub, expr_data_sub, by = "sample",all.y = TRUE)
    filename<-paste(expr_data$cancerType[i],"latestageExprandClin.csv",sep="")
    write.csv(comtable,filename)
}
```
