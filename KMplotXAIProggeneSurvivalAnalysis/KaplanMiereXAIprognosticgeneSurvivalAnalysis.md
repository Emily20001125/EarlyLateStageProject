# Kaplan Miere XAI prognostic gene Survival Analysis
Author: "Shu-jing,Huang"
Date: "2024/01/02"



# XAI Gene Cox Regression Analysis
## environment
```shell
conda activate survival
cd /home/emily2835/EarlyLateStageProject/KMplotXAIProggeneSurvivalAnalysis
```
## import package
```r
library(tidyverse)
library(tictoc)
library(survminer)
library(survival)
```
## Import data
```r
KIRCearlystageCox <- read.csv("Data/KIRCearlystageCoxAllgene.csv",sep = ",")
KIRClatestageCox <- read.csv("Data/KIRClatestageCoxAllgene.csv",sep = ",")
KIRPearystageCox <- read.csv("Data/KIRPearlystageCoxAllgene.csv",sep = ",")
KIRPlatestageCox <- read.csv("Data/KIRPlatestageCoxAllgene.csv",sep = ",")
GeneCompare <- read.csv("Data/probeMap_gencode.v23.annotation.gene.probemap",sep = "\t")
KIRCIntersectionGenelist <- read.csv("Data/KIRCIntersectionGenelist.csv",sep = ",")
KIRPIntersectionGenelist <- read.csv("Data/KIRPIntersectionGenelist.csv",sep = ",")
```
## Function`Cox_regression`: XAI gene's Cox Regression result
```r
Cox_regression <- function(df,IntersectionGenelist) {
  # df rownames remove "low"
  current_rownames = rownames(df)
  new_rownames = gsub("low", "", current_rownames)
  rownames(df) = new_rownames
  # select the IntersectionGenelist from GeneCompare
  GeneCompare <- GeneCompare[GeneCompare$gene %in% IntersectionGenelist$Gene.symbol,]
  # remove the version number
  GeneCompare$id <- sapply(strsplit(GeneCompare$id, split = "\\."), `[[`, 1)
  # change the rownames to gene symbol using GeneCompare
  combined_df <- merge(GeneCompare, df, by.x = "id", by.y = "row.names")
  # change column name Row.names to gene ensemble id
  colnames(combined_df)[1] <- "ensembl_gene_id"
  colnames(combined_df)[11] <- "p.value"
  # Sort the combined_df by abs(1-exp.coef.)
  combined_df["abs(exp.coef.)"] = abs(1 - combined_df$exp.coef.)
  # return the combined_df
  return(combined_df)
}
```
## process KIRCearlystageCox and KIRClatestageCox
```r
KIRCearlystageCox_processed = Cox_regression(KIRCearlystageCox,KIRCIntersectionGenelist)
KIRClatestageCox_processed = Cox_regression(KIRClatestageCox,KIRCIntersectionGenelist)
KIRPearlystageCox_processed = Cox_regression(KIRPearystageCox,KIRPIntersectionGenelist)
KIRPlatestageCox_processed = Cox_regression(KIRPlatestageCox,KIRPIntersectionGenelist)
```
## save the processed data
```r
write.csv(KIRCearlystageCox_processed, "Output/KIRCXAIearlystageCox.csv", row.names = FALSE)
write.csv(KIRClatestageCox_processed, "Output/KIRCXAIlateststageCox.csv", row.names = FALSE)
write.csv(KIRPearlystageCox_processed, "Output/KIRPXAIearlystageCox.csv", row.names = FALSE)
write.csv(KIRPlatestageCox_processed, "Output/KIRPXAIlateststageCox.csv", row.names = FALSE)
```

# XAI 5 model SHAP value summary

## environment
```shell
conda activate python3_10
cd /home/emily2835/EarlyLateStageProject/KMplotXAIProggeneSurvivalAnalysis
```
## import package
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from matplotlib.lines import Line2D
from lifelines import CoxPHFitter
```
## import data
```python
KIRCLightgbmSHAP = pd.read_csv('Data/KIRCLightgbmSHAPMeanValue.csv')
KIRCRandomForestSHAP = pd.read_csv('Data/KIRCRandomForestSHAPMeanValue.csv')
KIRCSVCSHAP = pd.read_csv('Data/KIRCSVCSHAPMeanValue.csv')
KIRCXGBoostSHAP = pd.read_csv('Data/KIRCXGBOOSTSHAPMeanValue.csv')
KIRCLogisticRegressionSHAP = pd.read_csv('Data/KIRCLogisticRegressionSHAPMeanValue.csv')
KIRCIntersectionGenelist = pd.read_csv('Data/KIRCIntersectionGenelist.csv')
```
## Select the XAI genes from each model
```python
KIRCSummarySHAP = pd.DataFrame()
a = ["Feature","Gene symbol","Gene ensemble id"]
KIRCSummarySHAP = KIRCLightgbmSHAP[a]
KIRCLightgbmSHAP = KIRCLightgbmSHAP[KIRCLightgbmSHAP['Gene symbol'].isin(KIRCIntersectionGenelist['Gene symbol'].tolist())]
KIRCLightgbmSHAP['Lightgbm SHAP Rank'] = KIRCLightgbmSHAP['SHAP Value'].rank()
KIRCRandomForestSHAP = KIRCRandomForestSHAP[KIRCRandomForestSHAP['Gene symbol'].isin(KIRCIntersectionGenelist['Gene symbol'].tolist())]
KIRCRandomForestSHAP['RandomForest SHAP Rank'] = KIRCRandomForestSHAP['SHAP Value'].rank()
KIRCSVCSHAP = KIRCSVCSHAP[KIRCSVCSHAP['Gene symbol'].isin(KIRCIntersectionGenelist['Gene symbol'].tolist())]
KIRCSVCSHAP['SVC SHAP Rank'] = KIRCSVCSHAP['SHAP Value'].rank()
KIRCXGBoostSHAP = KIRCXGBoostSHAP[KIRCXGBoostSHAP['Gene symbol'].isin(KIRCIntersectionGenelist['Gene symbol'].tolist())]
KIRCXGBoostSHAP['XGBoost SHAP Rank'] = KIRCXGBoostSHAP['SHAP Value'].rank()
KIRCLogisticRegressionSHAP = KIRCLogisticRegressionSHAP[KIRCLogisticRegressionSHAP['Gene symbol'].isin(KIRCIntersectionGenelist['Gene symbol'].tolist())]
KIRCLogisticRegressionSHAP['LogisticRegression SHAP Rank'] = KIRCLogisticRegressionSHAP['SHAP Value'].rank()
# Combine the SHAP Rank from each model
KIRCSummarySHAP = KIRCSummarySHAP.merge(KIRCLightgbmSHAP[['Gene symbol', 'Lightgbm SHAP Rank']], on='Gene symbol', how='left')
KIRCSummarySHAP = KIRCSummarySHAP.merge(KIRCRandomForestSHAP[['Gene symbol', 'RandomForest SHAP Rank']], on='Gene symbol', how='left')
KIRCSummarySHAP = KIRCSummarySHAP.merge(KIRCSVCSHAP[['Gene symbol', 'SVC SHAP Rank']], on='Gene symbol', how='left')
KIRCSummarySHAP = KIRCSummarySHAP.merge(KIRCXGBoostSHAP[['Gene symbol', 'XGBoost SHAP Rank']], on='Gene symbol', how='left')
KIRCSummarySHAP = KIRCSummarySHAP.merge(KIRCLogisticRegressionSHAP[['Gene symbol', 'LogisticRegression SHAP Rank']], on='Gene symbol', how='left')
# sum the SHAP Rank from each model
KIRCSummarySHAP['Sum of SHAP Rank'] = KIRCSummarySHAP['Lightgbm SHAP Rank'] + KIRCSummarySHAP['RandomForest SHAP Rank'] + KIRCSummarySHAP['SVC SHAP Rank'] + KIRCSummarySHAP['XGBoost SHAP Rank'] + KIRCSummarySHAP['LogisticRegression SHAP Rank']
# sort the SHAP Rank from each model
KIRCSummarySHAP = KIRCSummarySHAP.sort_values(by=['Sum of SHAP Rank'], ascending=False)
# save the data
KIRCSummarySHAP.to_csv('Output/KIRCXAI5ModelSHAPRank.csv', index=False)
```

## import data
```python
KIRPLightgbmSHAP = pd.read_csv('Data/KIRPLightgbmSHAPMeanValue.csv')
KIRPRandomForestSHAP = pd.read_csv('Data/KIRPRandomForestSHAPMeanValue.csv')
KIRPSVCSHAP = pd.read_csv('Data/KIRPSVCSHAPMeanValue.csv')
KIRPXGBoostSHAP = pd.read_csv('Data/KIRPXGBOOSTSHAPMeanValue.csv')
KIRPLogisticRegressionSHAP = pd.read_csv('Data/KIRPLogisticRegressionSHAPMeanValue.csv')
KIRPIntersectionGenelist = pd.read_csv('Data/KIRPIntersectionGenelist.csv')
```
## Select the XAI genes from each model
```python
KIRPSummarySHAP = pd.DataFrame()
a = ["Feature","Gene symbol","Gene ensemble id"]
KIRPSummarySHAP = KIRPLightgbmSHAP[a]
KIRPLightgbmSHAP = KIRPLightgbmSHAP[KIRPLightgbmSHAP['Gene symbol'].isin(KIRPIntersectionGenelist['Gene symbol'].tolist())]
KIRPLightgbmSHAP['Lightgbm SHAP Rank'] = KIRPLightgbmSHAP['SHAP Value'].rank()
KIRPRandomForestSHAP = KIRPRandomForestSHAP[KIRPRandomForestSHAP['Gene symbol'].isin(KIRPIntersectionGenelist['Gene symbol'].tolist())]
KIRPRandomForestSHAP['RandomForest SHAP Rank'] = KIRPRandomForestSHAP['SHAP Value'].rank()
KIRPSVCSHAP = KIRPSVCSHAP[KIRPSVCSHAP['Gene symbol'].isin(KIRPIntersectionGenelist['Gene symbol'].tolist())]
KIRPSVCSHAP['SVC SHAP Rank'] = KIRPSVCSHAP['SHAP Value'].rank()
KIRPXGBoostSHAP = KIRPXGBoostSHAP[KIRPXGBoostSHAP['Gene symbol'].isin(KIRPIntersectionGenelist['Gene symbol'].tolist())]
KIRPXGBoostSHAP['XGBoost SHAP Rank'] = KIRPXGBoostSHAP['SHAP Value'].rank()
KIRPLogisticRegressionSHAP = KIRPLogisticRegressionSHAP[KIRPLogisticRegressionSHAP['Gene symbol'].isin(KIRPIntersectionGenelist['Gene symbol'].tolist())]
KIRPLogisticRegressionSHAP['LogisticRegression SHAP Rank'] = KIRPLogisticRegressionSHAP['SHAP Value'].rank()
# Combine the SHAP Rank from each model
KIRPSummarySHAP = KIRPSummarySHAP.merge(KIRPLightgbmSHAP[['Gene symbol', 'Lightgbm SHAP Rank']], on='Gene symbol', how='left')
KIRPSummarySHAP = KIRPSummarySHAP.merge(KIRPRandomForestSHAP[['Gene symbol', 'RandomForest SHAP Rank']], on='Gene symbol', how='left')
KIRPSummarySHAP = KIRPSummarySHAP.merge(KIRPSVCSHAP[['Gene symbol', 'SVC SHAP Rank']], on='Gene symbol', how='left')
KIRPSummarySHAP = KIRPSummarySHAP.merge(KIRPXGBoostSHAP[['Gene symbol', 'XGBoost SHAP Rank']], on='Gene symbol', how='left')
KIRPSummarySHAP = KIRPSummarySHAP.merge(KIRPLogisticRegressionSHAP[['Gene symbol', 'LogisticRegression SHAP Rank']], on='Gene symbol', how='left')
# sum the SHAP Rank from each model
KIRPSummarySHAP['Sum of SHAP Rank'] = KIRPSummarySHAP['Lightgbm SHAP Rank'] + KIRPSummarySHAP['RandomForest SHAP Rank'] + KIRPSummarySHAP['SVC SHAP Rank'] + KIRPSummarySHAP['XGBoost SHAP Rank'] + KIRPSummarySHAP['LogisticRegression SHAP Rank']
# sort the SHAP Rank from each model
KIRPSummarySHAP = KIRPSummarySHAP.sort_values(by=['Sum of SHAP Rank'], ascending=False)
# save the data
KIRPSummarySHAP.to_csv('Output/KIRPXAI5ModelSHAPRank.csv', index=False)
```



## Function`survival_analysis`: XAI gene's Kaplan Meier Survival Analysis
```python
# Choose cancer type
cancer_type = 'KIRC'
# Separate the data into early and late stages
median_expression = data['PPP1CB'].median()
high_expression = data[data['PPP1CB'] >= median_expression]
low_expression = data[data['PPP1CB'] < median_expression]
# Perform the log-rank test
results = logrank_test(high_expression['OS.time'], low_expression['OS.time'], event_observed_A=high_expression['OS'], event_observed_B=low_expression['OS'])
# Get p-value
p_value = results.p_value
p_value = "{:.2e}".format(p_value)
print(f'p-value for {cancer_type}: {p_value}')
# Initialize the KaplanMeierFitter model
kmf = KaplanMeierFitter()
kmf.fit(high_expression['OS.time'], event_observed=high_expression['OS'], label='High expression')
ax = kmf.plot_survival_function(ci_show=False, color='#236192')
kmf.fit(low_expression['OS.time'], event_observed=low_expression['OS'], label='Low expression')
kmf.plot_survival_function(ax=ax, ci_show=False, color='#B33D26')
legend_elements = [Line2D([0], [0], color='#236192', lw=2, label=f'High expression (n={len(high_expression)})'),
                   Line2D([0], [0], color='#B33D26', lw=2, label=f'Low expression (n={len(low_expression)})'),
                   Line2D([0], [0], color='white', label=f'p-value: {p_value}')]
plt.legend(handles=legend_elements)
plt.title(f'Survival Analysis of {cancer_type} Across Different Stages')
plt.xlabel('Time in months')
plt.ylabel('Survival probability')
plt.savefig(f'Figure/{cancer_type}earlyandlateKMplot.png')
plt.close()
```