# Drug prediction 
Author : Huang,Shu-jing
Date : 2024/01/19

# Drug prediction - TPM Expression and GDSC2 drug AUC prediction (data from DepMap)
## Environment
```shell
conda activate oncoPredict2
cd /home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict/DrugPrediction_DepMapTPMandDepMapGDSC2AUC
```
## Import package
```r
library(oncoPredict)
library(tidyverse)
library(magrittr)
```
## Import data
```r
CCLE <- read_csv("OmicsExpressionProteinCodingGenesTPMLogp1.csv")
GDSC2 <- read_csv("Drug_sensitivity_AUC_GDSC2.csv")
```
## Data preprocessing
```r
# rename the ...1 column in the CCLE data
colnames(CCLE)[1] <- "Sample"
# rename the ...1 column in the GDSC2 data
colnames(GDSC2)[1] <- "Sample"
# remove the rownames with () in the CCLE data e.g. TSPAN6 (7105)` `TNMD (64102)` 
names(CCLE) <- gsub("\\s*\\(.*\\)", "", names(CCLE))
# remove - in the CCLE ...1 column
CCLE$Sample <- gsub("-", "_", CCLE$Sample) 
# remove - in the GDSC2 ...1 column
GDSC2$Sample <- gsub("-", "_", GDSC2$Sample)
```
## Prepare the data for train
```r
trainExpr <- t(CCLE[,-1])
colnames(trainExpr) <- CCLE$Sample
trainLabel <- GDSC2[,-1]
trainLabel <- as.data.frame(trainLabel)
rownames(trainLabel) <- GDSC2$Sample
# Check dimansion
dim(trainExpr)
dim(trainLabel)
# Select the intersection of the samples in the CCLE and GDSC2 data
common_samples <- intersect(colnames(trainExpr), rownames(trainLabel))
trainExpr <- trainExpr[,common_samples] # Train x
trainLabel <- trainLabel[common_samples,] # Train y
```
## Prepare the data for valid
```r
# rename trainExpr sample names as valid sample names
validData <- trainExpr
colnames(validData) <- paste0(colnames(trainExpr), "_valid")
dim(validData)
```
## CalcPhenotype
```r
calcPhenotype(
    trainingExprData = as.matrix(trainExpr),
    trainingPtype = as.matrix(trainLabel),
    testExprData = as.matrix(validData),
    batchCorrect = "eb", 
    removeLowVaryingGenes = 0.2,
    minNumSamples = 10,
    percent = 80, 
    selection = -1,
    removeLowVaringGenesFrom = "rawData",
    powerTransformPhenotype = TRUE,
    pcr = FALSE, # dimension reduction false
    printOutput = TRUE, 
    report_pc = FALSE, # report the principal components false
    cc = TRUE,
    rsq=TRUE
)
```

---------------------- screen -r 1599561.oncoPredict2 --------------------------

# Drug prediction - RSEM Expression and GDSC2 drug AUC prediction
## Environment
```shell
conda activate oncoPredict2
cd /home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict/DrugPrediction_OncopredictOriginalandGDSC2IC50
```
## Import package
```r
library(oncoPredict)
library(tidyverse)
library(magrittr)
```
## Import data
```r
GDSC2_Expr = readRDS('GDSC2_Expr (RMA Normalized and Log Transformed).rds')
GDSC2_Res = readRDS("GDSC2_Res.rds")
```
## Data preprocessing
```r
GDSC2_Res <- exp(GDSC2_Res) 
```
## Prepare the data for train
```r
trainExpr <- as.matrix(GDSC2_Expr)
trainLabel <- as.matrix(GDSC2_Res)
dim(trainExpr)
dim(trainLabel)
```
## Prepare the data for valid
```r
validData <- GDSC2_Expr
colnames(validData) <- paste0(colnames(GDSC2_Expr), "_valid")
validData <- as.matrix(validData)
dim(validData)
```
# drug prediction
```r
calcPhenotype(
    trainingExprData = trainExpr,
    trainingPtype = trainLabel,
    testExprData = validData,
    batchCorrect = "eb", 
    removeLowVaryingGenes = 0.2,
    minNumSamples = 10,
    percent = 80, 
    selection = -1,
    removeLowVaringGenesFrom = "rawData",
    powerTransformPhenotype = TRUE,
    pcr = FALSE, # dimension reduction false
    printOutput = TRUE, 
    report_pc = FALSE, # report the principal components false
    cc = TRUE,
    rsq=TRUE
)
```
