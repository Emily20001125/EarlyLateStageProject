# Drug prediction 
Author : Huang,Shu-jing
Date : 2024/01/19


## Training data preparation
Load the packages
```r
library(oncoPredict)
library(tidyverse)
library(magrittr)
```

Import the data
```r
# read the CCLE data
CCLE <- read_csv("/home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict/Data/Training_data/OmicsExpressionProteinCodingGenesTPMLogp1.csv")
# read the GDSC1 data
GDSC1 <- read_csv("/home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict/Data/Training_data/Drug_sensitivity_AUC_GDSC1.csv")
# read the GDSC2 data
GDSC2 <- read_csv("/home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict/Data/Training_data/Drug_sensitivity_AUC_GDSC2.csv")
```
Data preprocessing
```r
# remove the rownames with () in the CCLE data e.g. TSPAN6 (7105)` `TNMD (64102)` 
names(CCLE) <- gsub("\\s*\\(.*\\)", "", names(CCLE))
# remove - in the CCLE ...1 column
CCLE$...1 <- gsub("-", "", CCLE$...1) 
# remove - in the GDSC2 ...1 column
GDSC2$...1 <- gsub("-", "", GDSC2$...1)
# Prepare the data for training
trainingExpr <- t(CCLE[,-1])
colnames(trainingExpr) <- CCLE$...1
trainingLabel <- GDSC2[,-1]
trainingLabel <- as.data.frame(trainingLabel)
rownames(trainingLabel) <- GDSC2$...1
# Select the intersection of the samples in the CCLE and GDSC2 data
common_samples <- intersect(colnames(trainingExpr), rownames(trainingLabel))
trainingExpr <- trainingExpr[,common_samples]
trainingLabel <- trainingLabel[common_samples,]
# Valid data
ValidData <- trainingExpr[1:50,]
# rename validData column names with original names + test
colnames(ValidData) <- paste0(colnames(ValidData), "_test")
# Check duplicated rownames
any(duplicated(rownames(trainingExpr)))
any(duplicated(rownames(trainingLabel)))
any(duplicated(colnames(ValidData)))
# as matrix
trainingExpr <- as.matrix(trainingExpr)
trainingLabel <- exp(trainingLabel)
trainingLabel <- as.matrix(trainingLabel)
ValidData <- as.matrix(ValidData)
```
CalcPhenotype
```r
calcPhenotype(
    trainingExprData = trainingExpr,
    trainingPtype = trainingLabel,
    testExprData = ValidData,
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
Evaluate the model
```r
# import the data
result <- read_csv("calcPhenotype_Output/DrugPredictions.csv")
result <- result %>% select(-c(1))
# calculate the correlation of ValidData and result
trainingLabel2 <- as.data.frame(trainingLabel)
# select the front 50 samples
trainingLabel2 <- trainingLabel2[1:96,1]
# remove NA
trainingLabel2[is.na(trainingLabel2)] <- 0

cor(trainingLabel2["(+)-CAMPTOTHECIN (GDSC2:1003)"],result["Camptothecin_1003"],method = "spearman")
cor(trainingLabel2["VELBAN (GDSC2:1004)"],result["Vinblastine_1004"],method = "spearman")
cor(trainingLabel2[,3],result[,4])
cor(trainingLabel2[,4],result[,5])
```

## Original data test
```r
# import drug data
dir='./Data/OncopredictData/'
GDSC2_Expr = readRDS(file=file.path(dir,'GDSC2_Expr (RMA Normalized and Log Transformed).rds'))
GDSC2_Res = readRDS(file = file.path(dir,"GDSC2_Res.rds"))
GDSC2_Res <- exp(GDSC2_Res) 
testExprData <- as.matrix(GDSC2_Expr[,1:50])
# set the colnames
colnames(testExprData) <- paste0(colnames(testExprData), "_test")

# drug prediction
calcPhenotype(
    trainingExprData = GDSC2_Expr,
    trainingPtype = GDSC2_Res,
    testExprData,
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
### Evaluate the model
```r
# import the data
result<- read_csv("calcPhenotype_Output_rawdata/DrugPredictions.csv")
result <- result %>% select(-c(1))
colnames(result) <- paste0(colnames(result), "_test")
# calculate the correlation of ValidData and result
GDSC2_Res2 <- as.tibble(GDSC2_Res)
# select the front 50 samples
GDSC2_Res2 <- GDSC2_Res2[1:96,]
# add label
# Combine the result and GDSC2_Res2
result2 <- cbind(GDSC2_Res2,result)
# Use 0 to replace NA
result2[is.na(result2)] <- 0
# 
# Plot the scatter plot
ggplot(result2, aes(x = Camptothecin_1003, y = Camptothecin_1003_test)) +
    geom_point(color = "#69b3a2") +
    geom_abline(intercept = 0, slope = 1) +
    theme_bw() +
    theme(legend.position = "none")
ggsave("Camptothecin_1003.png",width = 5,height = 5)

# calculate the correlation
result[is.na(result)] <- 0
GDSC2_Res2[is.na(GDSC2_Res2)] <- 0
cor(GDSC2_Res2[,1],result[,1],method = "spearman")
cor(GDSC2_Res2[,2],result[,2],method = "spearman")
cor(GDSC2_Res2[,3],result[,3],method = "spearman")
cor(GDSC2_Res2[,4],result[,4],method = "spearman")
cor(GDSC2_Res2[,5],result[,5],method = "spearman")
```














## Testing data preparation
Environmect
```r
conda activate basic_work
cd /home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict
```
Import package
```r
library(UCSCXenaTools)
library(tidyverse)
```
Download the data
```r
# set the VROOM_CONNECTION_SIZE
Sys.setenv("VROOM_CONNECTION_SIZE"=500072)
# Download clinical data from UCSC Xena
datasetID = 'tcga_rsem_isoform_tpm'

file_op3 = 
    XenaGenerate() %>% 
    XenaFilter(filterDatasets = datasetID)
file_op3@datasets

data = 
    file_op3 %>%
    XenaQuery() %>%
    XenaDownload() %>%
    XenaPrepare()

head(data)
# Save expression file as csv file
write.csv(data,file="TCGAPanCancer_tcgarsemisoformtpm.csv")
```

## Oncopredict Drug prediction