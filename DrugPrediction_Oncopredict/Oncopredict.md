# Drug prediction
Author : Huang,Shu-jing
Date : 2023/11/22

- [Drug prediction](#drug-prediction)
  - [1.1. Environment](#11-environment)
  - [1.2. Import package](#12-import-package)
  - [1.3. Import data](#13-import-data)
  - [1.4. Drug prediction](#14-drug-prediction)
- [import drug data](#import-drug-data)
- [Input expression data](#input-expression-data)
- [drug prediction](#drug-prediction-1)


## 1.1. Environment 
```shell
conda activate oncoPredict2
cd stage_project_git2/Drug_prediction
```

## 1.2. Import package 
```r
library(oncoPredict)
library(tidyverse)
```
## 1.3. Import data
```r
# import data
cancerlist =c("BRCA","BLCA","KIRP","KIRC","HNSC","ESCA","LUAD","LUSC","LIHC","COAD","STAD")
for (i in cancerlist){
  assign(paste("Prog",i,sep=""),read.csv(paste("Data/",i,".csv",sep="")))
}
ProgBRCA = read.csv("Data/BRCA.csv",sep=",")
# annotation
annotation = read.csv("Data/probeMap_gencode.v23.annotation.gene.probemap",sep="\t")
# extract gene expression data which start at ENSG
ProgBRCA = ProgBRCA[,c(1,which(grepl("ENSG",colnames(ProgBRCA))))]
# transform ProgBRCA colname ensembl id to gene symbol
colnames(ProgBRCA)[2:ncol(ProgBRCA)] = annotation$gene[match(colnames(ProgBRCA)[2:ncol(ProgBRCA)],annotation$id)]
```
## 1.4. Drug prediction
```r
# import drug data
dir='./Data/Oncopredict_data/Training Data/'
GDSC2_Expr = readRDS(file=file.path(dir,'GDSC2_Expr (RMA Normalized and Log Transformed).rds'))
GDSC2_Res = readRDS(file = file.path(dir,"GDSC2_Res.rds"))
GDSC2_Res <- exp(GDSC2_Res) 
# Input expression data

# drug prediction
calcPhenotype(
    trainingExprData=GDSC2_Expr,
    trainingPtype=GDSC2_Res,
    testExprData,
    batchCorrect,
    powerTransformPhenotype = TRUE,
    removeLowVaryingGenes = 0.2,
    minNumSamples,
    selection = 1,
    printOutput,
    pcr = FALSE,
    removeLowVaringGenesFrom,
    report_pc = FALSE,
    cc = FALSE,
    percent = 80,
    rsq = FALSE
)
