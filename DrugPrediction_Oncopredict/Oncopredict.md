# Drug prediction 
Author : Huang,Shu-jing
Date : 2024/01/17

- [Drug prediction](#drug-prediction)
- [Risk Grouping](#risk-grouping)
  - [Environment](#environment)
  - [Import package](#import-package)
  - [Import data](#import-data)
  - [Cox Proportional-Hazards Model Survival Risk Grouping](#cox-proportional-hazards-model-survival-risk-grouping)
  - [Plot the survival KM plot for risk group](#plot-the-survival-km-plot-for-risk-group)
  - [Extract the whole genome gene expression data](#extract-the-whole-genome-gene-expression-data)
- [Drug prediction](#drug-prediction-1)
  - [Environment](#environment-1)
  - [Import package](#import-package-1)
  - [Import data](#import-data-1)
  - [Data preprocessing](#data-preprocessing)
  - [Drug prediction](#drug-prediction-2)
  - [Patient impute drug biomarker](#patient-impute-drug-biomarker)


# Risk Grouping 
## Environment 
```shell
cd /home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict
conda activate survival
```

## Import package 
```r
library(survival)
library(survminer)
library(tidyverse)
```

## Import data
```r
KIRP_ELgene <- read_csv('Data/PatientData/KIRPearlylatelabelCoxProggene005.csv')
KIRP_Eexpr <- read_csv('Data/PatientData/KIRPearlystageExprandClin.csv')
KIRP_Lexpr <- read_csv('Data/PatientData/KIRPlatestageExprandClin.csv')
GENEcompare <- read_delim('Data/PatientData/probeMap_gencode.v23.annotation.gene.probemap', delim = '\t')
KIRPXAI5ModelSHAPRank <- read_csv('Data/PatientData/KIRPXAI5ModelSHAPRank.csv')
GENEcompare$id <- sub("\\..*", "", GENEcompare$id)
KIRP_Eexpr$E_L_Stage <- 'early'
KIRP_Lexpr$E_L_Stage <- 'late'
KIRP_Expr <- bind_rows(KIRP_Eexpr, KIRP_Lexpr)
table(KIRP_Expr$E_L_Stage)
KIRP_GeneExpr <- KIRP_Expr %>% select(KIRPXAI5ModelSHAPRank$`Gene ensemble id`)
names(KIRP_GeneExpr) <- GENEcompare$gene[GENEcompare$id %in% colnames(KIRP_GeneExpr)]
KIRP_GeneExpr$E_L_Stage <- KIRP_Expr$E_L_Stage
KIRP_GeneExpr$OS.time <- KIRP_Expr$OS.time
KIRP_GeneExpr$OS <- KIRP_Expr$OS
KIRP_GeneExpr$sample <- KIRP_Expr$sample
```

## Cox Proportional-Hazards Model Survival Risk Grouping 
Survival analysis
[Gene list]
```
[1] "MYCL"      "SGIP1"     "C1orf112"  "CLDND1"    "TMEM39A"   "KPNA4"    
[7] "DNAJC21"   "ARL14EPL"  "SOX30"     "TCF19"     "KLC4"      "RAET1E"
[13] "PI15"      "ZCCHC7"   "E2F8"      "NABP2"     "ARL1"      "YY1"
[19] "NPAP1"     "KDM8"     "PITPNM3"   "RTBDN"     "RAB8A"     "KCNG1"
[25] "CABLES2"   "FMR1" 
```

Early and Late stage
```r
# Survival analysis
res.cox <- coxph(Surv(OS.time, OS) ~ MYCL + SGIP1 + C1orf112 + CLDND1 + TMEM39A + KPNA4 + DNAJC21 + ARL14EPL + SOX30 + TCF19 + KLC4 + RAET1E + PI15 + ZCCHC7 + E2F8 + NABP2 + ARL1 + YY1 + NPAP1 + KDM8 + PITPNM3 + RTBDN + RAB8A + KCNG1 + CABLES2 + FMR1 , data = KIRP_GeneExpr)
summary(res.cox)
# Calculate the risk score
KIRP_GeneExpr$RiskScore <- res.cox$coefficients[1] * KIRP_GeneExpr$MYCL + res.cox$coefficients[2] * KIRP_GeneExpr$SGIP1 + res.cox$coefficients[3] * KIRP_GeneExpr$C1orf112 + res.cox$coefficients[4] * KIRP_GeneExpr$CLDND1 + res.cox$coefficients[5] * KIRP_GeneExpr$TMEM39A + res.cox$coefficients[6] * KIRP_GeneExpr$KPNA4 + res.cox$coefficients[7] * KIRP_GeneExpr$DNAJC21 + res.cox$coefficients[8] * KIRP_GeneExpr$ARL14EPL + res.cox$coefficients[9] * KIRP_GeneExpr$SOX30 + res.cox$coefficients[10] * KIRP_GeneExpr$TCF19 + res.cox$coefficients[11] * KIRP_GeneExpr$KLC4 + res.cox$coefficients[12] * KIRP_GeneExpr$RAET1E + res.cox$coefficients[13] * KIRP_GeneExpr$PI15 + res.cox$coefficients[14] * KIRP_GeneExpr$ZCCHC7 + res.cox$coefficients[15] * KIRP_GeneExpr$E2F8 + res.cox$coefficients[16] * KIRP_GeneExpr$NABP2 + res.cox$coefficients[17] * KIRP_GeneExpr$ARL1 + res.cox$coefficients[18] * KIRP_GeneExpr$YY1 + res.cox$coefficients[19] * KIRP_GeneExpr$NPAP1 + res.cox$coefficients[20] * KIRP_GeneExpr$KDM8 + res.cox$coefficients[21] * KIRP_GeneExpr$PITPNM3 + res.cox$coefficients[22] * KIRP_GeneExpr$RTBDN + res.cox$coefficients[23] * KIRP_GeneExpr$RTBDN + res.cox$coefficients[24] * KIRP_GeneExpr$KCNG1 + res.cox$coefficients[25] * KIRP_GeneExpr$CABLES2 + res.cox$coefficients[26] * KIRP_GeneExpr$FMR1
# Risk group
medianRisk <- median(KIRP_GeneExpr$RiskScore)
KIRP_GeneExpr$Riskgroup <- ifelse(KIRP_GeneExpr$RiskScore > medianRisk, "HighRisk", "LowRisk")
write_csv(KIRP_GeneExpr,'Output/KIRPGeneExprRiskGroup.csv')
```


Early stage
```r
# Select the early stage data
KIRP_GeneExprEarly <- KIRP_GeneExpr[KIRP_GeneExpr$E_L_Stage == 'early',]
res.cox <- coxph(Surv(OS.time, OS) ~ MYCL + SGIP1 + C1orf112 + CLDND1 + TMEM39A + KPNA4 + DNAJC21 + ARL14EPL + SOX30 + TCF19 + KLC4 + RAET1E + PI15 + ZCCHC7 + E2F8 + NABP2 + ARL1 + YY1 + NPAP1 + KDM8 + PITPNM3 + RTBDN + RAB8A + KCNG1 + CABLES2 + FMR1 , data = KIRP_GeneExprEarly)
summary(res.cox)
# Calculate the risk score
KIRP_GeneExprEarly$RiskScore <- res.cox$coefficients[1] * KIRP_GeneExprEarly$MYCL + res.cox$coefficients[2] * KIRP_GeneExprEarly$SGIP1 + res.cox$coefficients[3] * KIRP_GeneExprEarly$C1orf112 + res.cox$coefficients[4] * KIRP_GeneExprEarly$CLDND1 + res.cox$coefficients[5] * KIRP_GeneExprEarly$TMEM39A + res.cox$coefficients[6] * KIRP_GeneExprEarly$KPNA4 + res.cox$coefficients[7] * KIRP_GeneExprEarly$DNAJC21 + res.cox$coefficients[8] * KIRP_GeneExprEarly$ARL14EPL + res.cox$coefficients[9] * KIRP_GeneExprEarly$SOX30 + res.cox$coefficients[10] * KIRP_GeneExprEarly$TCF19 + res.cox$coefficients[11] * KIRP_GeneExprEarly$KLC4 + res.cox$coefficients[12] * KIRP_GeneExprEarly$RAET1E + res.cox$coefficients[13] * KIRP_GeneExprEarly$PI15 + res.cox$coefficients[14] * KIRP_GeneExprEarly$ZCCHC7 + res.cox$coefficients[15] * KIRP_GeneExprEarly$E2F8 + res.cox$coefficients[16] * KIRP_GeneExprEarly$NABP2 + res.cox$coefficients[17] * KIRP_GeneExprEarly$ARL1 + res.cox$coefficients[18] * KIRP_GeneExprEarly$YY1 + res.cox$coefficients[19] * KIRP_GeneExprEarly$NPAP1 + res.cox$coefficients[20] * KIRP_GeneExprEarly$KDM8 + res.cox$coefficients[21] * KIRP_GeneExprEarly$PITPNM3 + res.cox$coefficients[22] * KIRP_GeneExprEarly$RTBDN + res.cox$coefficients[23] * KIRP_GeneExprEarly$RAB8A + res.cox$coefficients[24] * KIRP_GeneExprEarly$KCNG1 + res.cox$coefficients[25] * KIRP_GeneExprEarly$CABLES2 + res.cox$coefficients[26] * KIRP_GeneExprEarly$FMR1
# Risk group
medianRisk <- median(KIRP_GeneExprEarly$RiskScore)
KIRP_GeneExprEarly$Riskgroup <- ifelse(KIRP_GeneExprEarly$RiskScore > medianRisk, "HighRisk", "LowRisk")
write_csv(KIRP_GeneExprEarly,'Output/KIRPGeneExprEarlyRiskGroup.csv')
```

Late stage
```r
# Select the late stage data
KIRP_GeneExprLate <- KIRP_GeneExpr[KIRP_GeneExpr$E_L_Stage == 'late',]
res.cox <- coxph(Surv(OS.time, OS) ~ MYCL + SGIP1 + C1orf112 + CLDND1 + TMEM39A + KPNA4 + DNAJC21 + ARL14EPL + SOX30 + TCF19 + KLC4 + RAET1E + PI15 + ZCCHC7 + E2F8 + NABP2 + ARL1 + YY1 + NPAP1 + KDM8 + PITPNM3 + RTBDN + RAB8A + KCNG1 + CABLES2 + FMR1 , data = KIRP_GeneExprLate)
summary(res.cox)
# Calculate the risk score
KIRP_GeneExprLate$RiskScore <- res.cox$coefficients[1] * KIRP_GeneExprLate$MYCL + res.cox$coefficients[2] * KIRP_GeneExprLate$SGIP1 + res.cox$coefficients[3] * KIRP_GeneExprLate$C1orf112 + res.cox$coefficients[4] * KIRP_GeneExprLate$CLDND1 + res.cox$coefficients[5] * KIRP_GeneExprLate$TMEM39A + res.cox$coefficients[6] * KIRP_GeneExprLate$KPNA4 + res.cox$coefficients[7] * KIRP_GeneExprLate$DNAJC21 + res.cox$coefficients[8] * KIRP_GeneExprLate$ARL14EPL + res.cox$coefficients[9] * KIRP_GeneExprLate$SOX30 + res.cox$coefficients[10] * KIRP_GeneExprLate$TCF19 + res.cox$coefficients[11] * KIRP_GeneExprLate$KLC4 + res.cox$coefficients[12] * KIRP_GeneExprLate$RAET1E + res.cox$coefficients[13] * KIRP_GeneExprLate$PI15 + res.cox$coefficients[14] * KIRP_GeneExprLate$ZCCHC7 + res.cox$coefficients[15] * KIRP_GeneExprLate$E2F8 + res.cox$coefficients[16] * KIRP_GeneExprLate$NABP2 + res.cox$coefficients[17] * KIRP_GeneExprLate$ARL1 + res.cox$coefficients[18] * KIRP_GeneExprLate$YY1 + res.cox$coefficients[19] * KIRP_GeneExprLate$NPAP1 + res.cox$coefficients[20] * KIRP_GeneExprLate$KDM8 + res.cox$coefficients[21] * KIRP_GeneExprLate$PITPNM3 + res.cox$coefficients[22] * KIRP_GeneExprLate$RTBDN + res.cox$coefficients[23] * KIRP_GeneExprLate$RAB8A + res.cox$coefficients[24] * KIRP_GeneExprLate$KCNG1 + res.cox$coefficients[25] * KIRP_GeneExprLate$CABLES2 + res.cox$coefficients[26] * KIRP_GeneExprLate$FMR1
# Risk group
medianRisk <- median(KIRP_GeneExprLate$RiskScore)
KIRP_GeneExprLate$Riskgroup <- ifelse(KIRP_GeneExprLate$RiskScore > medianRisk, "HighRisk", "LowRisk")
write_csv(KIRP_GeneExprLate,'Output/KIRPGeneExprLateRiskGroup.csv')
```

## Plot the survival KM plot for risk group
Environment
```shell
cd /home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict
conda activate python3_10
```
Import packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D
```
Import data
```python
EarlyRiskGroup = pd.read_csv('Output/KIRPGeneExprEarlyRiskGroup.csv')
LateRiskGroup = pd.read_csv('Output/KIRPGeneExprLateRiskGroup.csv')
EarlyLateRiskGroup = pd.read_csv('Output/KIRPGeneExprRiskGroup.csv')
```
Early Stage Survival analysis
```python
kmf = KaplanMeierFitter()
HighRisk = EarlyRiskGroup[EarlyRiskGroup['Riskgroup'] == 'HighRisk']
LowRisk = EarlyRiskGroup[EarlyRiskGroup['Riskgroup'] == 'LowRisk']
HighRisk['OS.time'] = HighRisk['OS.time']/30
LowRisk['OS.time'] = LowRisk['OS.time']/30
results = logrank_test(HighRisk['OS.time'], LowRisk['OS.time'], event_observed_A=HighRisk['OS'], event_observed_B=LowRisk['OS'])
p_value = results.p_value
p_value = "{:.2e}".format(p_value)
print(f'p-value for early stage: {p_value}')
kmf.fit(HighRisk['OS.time'], HighRisk['OS'], label=f'HighRisk (n={len(HighRisk)})')
ax = kmf.plot_survival_function(ci_show=False, color='#236192')
kmf.fit(LowRisk['OS.time'], LowRisk['OS'], label=f'LowRisk (n={len(LowRisk)})')
kmf.plot_survival_function(ax=ax, ci_show=False, color='#B33D26')
legend_elements = [Line2D([0], [0], color='#236192', lw=2, label=f'HighRisk (n={len(HighRisk)})'),
                   Line2D([0], [0], color='#B33D26', lw=2, label=f'LowRisk (n={len(LowRisk)})'),
                   Line2D([0], [0], color='white', label=f'p-value: {p_value}')]
plt.legend(handles=legend_elements)
plt.title(f'Survival Analysis of KIRP Early Stage Across Different Risk Groups')
plt.xlabel('Time in months')
plt.ylabel('Survival probability')
plt.savefig(f'Figure/KIRPearlyKMplot.png')
plt.close()
```
Late Stage Survival analysis
```python
kmf = KaplanMeierFitter()
HighRisk = LateRiskGroup[LateRiskGroup['Riskgroup'] == 'HighRisk']
LowRisk = LateRiskGroup[LateRiskGroup['Riskgroup'] == 'LowRisk']
HighRisk['OS.time'] = HighRisk['OS.time']/30
LowRisk['OS.time'] = LowRisk['OS.time']/30
results = logrank_test(HighRisk['OS.time'], LowRisk['OS.time'], event_observed_A=HighRisk['OS'], event_observed_B=LowRisk['OS'])
p_value = results.p_value
p_value = "{:.2e}".format(p_value)
print(f'p-value for late stage: {p_value}')
kmf.fit(HighRisk['OS.time'], HighRisk['OS'], label=f'HighRisk (n={len(HighRisk)})')
ax = kmf.plot_survival_function(ci_show=False, color='#236192')
kmf.fit(LowRisk['OS.time'], LowRisk['OS'], label=f'LowRisk (n={len(LowRisk)})')
kmf.plot_survival_function(ax=ax, ci_show=False, color='#B33D26')
legend_elements = [Line2D([0], [0], color='#236192', lw=2, label=f'HighRisk (n={len(HighRisk)})'),
                   Line2D([0], [0], color='#B33D26', lw=2, label=f'LowRisk (n={len(LowRisk)})'),
                   Line2D([0], [0], color='white', label=f'p-value: {p_value}')]
plt.legend(handles=legend_elements)
plt.title(f'Survival Analysis of KIRP Late Stage Across Different Risk Groups')
plt.xlabel('Time in months')
plt.ylabel('Survival probability')
plt.savefig(f'Figure/KIRPlateKMplot.png')
plt.close()
```
Early and Late Stage Survival analysis
```python
kmf = KaplanMeierFitter()
HighRisk = EarlyLateRiskGroup[EarlyLateRiskGroup['Riskgroup'] == 'HighRisk']
LowRisk = EarlyLateRiskGroup[EarlyLateRiskGroup['Riskgroup'] == 'LowRisk']
HighRisk['OS.time'] = HighRisk['OS.time']/30
LowRisk['OS.time'] = LowRisk['OS.time']/30
results = logrank_test(HighRisk['OS.time'], LowRisk['OS.time'], event_observed_A=HighRisk['OS'], event_observed_B=LowRisk['OS'])
p_value = results.p_value
p_value = "{:.2e}".format(p_value)
print(f'p-value for early and late stage: {p_value}')
kmf.fit(HighRisk['OS.time'], HighRisk['OS'], label=f'HighRisk (n={len(HighRisk)})')
ax = kmf.plot_survival_function(ci_show=False, color='#236192')
kmf.fit(LowRisk['OS.time'], LowRisk['OS'], label=f'LowRisk (n={len(LowRisk)})')
kmf.plot_survival_function(ax=ax, ci_show=False, color='#B33D26')
legend_elements = [Line2D([0], [0], color='#236192', lw=2, label=f'HighRisk (n={len(HighRisk)})'),
                   Line2D([0], [0], color='#B33D26', lw=2, label=f'LowRisk (n={len(LowRisk)})'),
                   Line2D([0], [0], color='white', label=f'p-value: {p_value}')]
plt.legend(handles=legend_elements)
plt.title(f'Survival Analysis of KIRP Across Different Risk Groups')
plt.xlabel('Time in months')
plt.ylabel('Survival probability')
plt.savefig(f'Figure/KIRPearlyandlateKMplot.png')
plt.close()
```


## Extract the whole genome gene expression data
```r
# import data
KIRPLateRiskGroup <- read_csv('Output/KIRPGeneExprLateRiskGroup.csv')
KIRPEarlyRiskGroup <- read_csv('Output/KIRPGeneExprEarlyRiskGroup.csv')
# KIRP_Eexpr add risk group label
KIRP_Eexpr$Riskgroup <- KIRPEarlyRiskGroup$Riskgroup
KIRP_Lexpr$Riskgroup <- KIRPLateRiskGroup$Riskgroup
# change ensembl id to gene symbol
# select start with ENSG id in KIRP_Eexpr
KIRP_EexprENSG <- KIRP_Eexpr %>% select(starts_with('ENSG'))
KIRP_LexprENSG <- KIRP_Lexpr %>% select(starts_with('ENSG'))
# change the row name to gene symbol
colnames(KIRP_EexprENSG) <- GENEcompare$gene[GENEcompare$id %in% colnames(KIRP_EexprENSG)]
colnames(KIRP_LexprENSG) <- GENEcompare$gene[GENEcompare$id %in% colnames(KIRP_LexprENSG)]
# add clinical data
KIRP_EexprENSG$E_L_Stage <- KIRP_Eexpr$E_L_Stage
KIRP_EexprENSG$Riskgroup <- KIRP_Eexpr$Riskgroup
KIRP_EexprENSG$OS.time <- KIRP_Eexpr$OS.time
KIRP_EexprENSG$OS <- KIRP_Eexpr$OS
KIRP_EexprENSG$sample <- KIRP_Eexpr$sample
KIRP_LexprENSG$E_L_Stage <- KIRP_Lexpr$E_L_Stage
KIRP_LexprENSG$Riskgroup <- KIRP_Lexpr$Riskgroup
KIRP_LexprENSG$OS.time <- KIRP_Lexpr$OS.time
KIRP_LexprENSG$OS <- KIRP_Lexpr$OS
KIRP_LexprENSG$sample <- KIRP_Lexpr$sample
# save the data
write_csv(KIRP_EexprENSG,'Output/KIRPEarlyRiskGroupExpr.csv')
write_csv(KIRP_LexprENSG,'Output/KIRPLateRiskGroupExpr.csv')
```




# Drug prediction

## Environment 
```shell
conda activate oncoPredict2
cd /home/emily2835/EarlyLateStageProject/DrugPrediction_Oncopredict
```
## Import package 
```r
library(oncoPredict)
library(tidyverse)
library(ggpubr)
```
## Import data
```r
KIRP_GeneExprEarly <- read.csv('Output/KIRPEarlyRiskGroupExpr.csv')
KIRP_GeneExprLate <- read.csv('Output/KIRPLateRiskGroupExpr.csv')
```
## Data preprocessing
```r
# select the gene expression data
Early_testExprData <- KIRP_GeneExprEarly %>% select(-c(E_L_Stage,Riskgroup,OS.time,OS,sample))
Late_testExprData <- KIRP_GeneExprLate %>% select(-c(E_L_Stage,Riskgroup,OS.time,OS,sample))
# log2 transform
Early_testExprData <- log2(Early_testExprData+1)
Late_testExprData <- log2(Late_testExprData+1)
```

## Drug prediction
```r
# import drug data
dir='./Data/Training_data'
GDSC2_Expr = readRDS(file=file.path(dir,'GDSC2_Expr (RMA Normalized and Log Transformed).rds'))
GDSC2_Res = readRDS(file = file.path(dir,"GDSC2_Res.rds"))
GDSC2_Res <- exp(GDSC2_Res) 
testExprData <- as.matrix(t(Late_testExprData))
# set the colnames
colnames(testExprData) <- KIRP_GeneExprLate$sample

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
# read the result
calcPhenotypeResult <- read_csv('calcPhenotype_Output/DrugPredictions.csv')
# add label
calcPhenotypeResult$E_L_Stage <- KIRP_GeneExprLate$E_L_Stage
calcPhenotypeResult$Riskgroup <- KIRP_GeneExprLate$Riskgroup
calcPhenotypeResult$sample <- KIRP_GeneExprLate$sample
# t.test for every drug sensitivity between high and low risk group
drug_list <- colnames(calcPhenotypeResult)[2:199]
p_values <- numeric(length(drug_list))
names(p_values) <- drug_list

for (i in drug_list){
    a <- calcPhenotypeResult[calcPhenotypeResult$Riskgroup == 'HighRisk',i]
    b <- calcPhenotypeResult[calcPhenotypeResult$Riskgroup == 'LowRisk',i]
    # remove NA
    a <- a[!is.na(a)]
    b <- b[!is.na(b)]
    if(length(a) > 0 & length(b) > 0){
        wilcox.test_result <- wilcox.test(a, b)
        p_values[i] <- wilcox.test_result$p.value
    } else {
        p_values[i] <- NA
    }
}

p_values_df <- tibble(Drug = names(p_values), p_value = p_values)
# Select the p-value < 0.05
sigdrug <- p_values_df %>% filter(p_value < 0.05) %>% arrange(p_value)
# plot boxplot
calcPhenotypeResult %>% 
    ggboxplot(x = 'Riskgroup', y = 'AZD1332_1463', fill = 'Riskgroup', palette = 'jco') +
    stat_compare_means(label = 'p.format', label.y = 0.5) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1)) +
    labs(title = 'AZD1332_1463 Drug Sensitivity Between High and Low Risk Group', x = 'Risk Group', y = 'AZD1332_1463IC50') 
ggsave('Figure/DrugSensitivityBetweenHighandLowRiskGroup.png', width = 10, height = 10, units = 'in')

# read the R2 result
R2 <- read.table('calcPhenotype_Output/R^2.txt', header = TRUE)
as_tibble(R2)
# read the cor result
PC <- read.table('calcPhenotype_Output/cors.txt', header = TRUE)
as_tibble(PC)
```




## Patient impute drug biomarker
```r
calcPhenotypeResult <- read_csv('calcPhenotype_Output/DrugPredictions.csv')
Mut <- read_csv('Data/PatientData/TCGAPanCancermc3SilentGeneMut.csv')
Mut <- Mut %>% select(-c(...1 ))
calcPhenotypeResult <- as.data.frame(calcPhenotypeResult)
rownames(calcPhenotypeResult) <- calcPhenotypeResult$...1
Mut <- as.data.frame(Mut)
idwas(calcPhenotypeResult, Mut, n = 10, cnv = TRUE)
```
