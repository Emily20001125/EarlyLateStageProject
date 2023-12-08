# PathwayEnrichment_ssGSEAAnalysis
Author: Huang,Shu-Jing
Date: 2023-11-28

## Environment
```shell
conda create --name ssGSEA
conda activate ssGSEA
cd /home/emily2835/EarlyLateStageProject/PathwayEnrichment_ssGSEA
mamba install -c bioconda bioconductor-fgsea
mamba install -c conda-forge r-tidyverse
mamba install -c bioconda bioconductor-gsva

```
## Import packages
```r
library(fgsea)
library(tidyverse)
library(ggplot2)
library(GSVA)
library(GSEABase)
library(ggpubr)
library(rstatix)
```

## Import data
```r
KIRPearlystageExpr <- read.csv("Data/KIRPearlystageExprandClin.csv", row.names = 1)
KIRPlatestageExpr <- read.csv("Data/KIRPlatestageExprandClin.csv", row.names = 1)
KIRPELgene <- read.csv("Data/KIRPearlylatelabelCoxProggene005.csv", row.names = 1)
KIRCearlystageExpr <- read.csv("Data/KIRCearlystageExprandClin.csv", row.names = 1)
KIRClatestageExpr <- read.csv("Data/KIRClatestageExprandClin.csv", row.names = 1)
KIRCELgene <- read.csv("Data/KIRCearlylatelabelCoxProggene005.csv", row.names = 1)
Geneanno <- read.csv("Data/probeMap_gencode.v23.annotation.gene.probemap",sep = "\t")
```


## Funcion`process_expression_data` : Data preprocessing function
```r
process_expression_data <- function(earlystageExpr, latestageExpr, ELgene, Geneanno) {
  # extract ProgStage filtwe out the earlyOnly and earlyLate  
  earlystageExpr <- earlystageExpr[,ELgene$ProgStage == "earlyOnly" | ELgene$ProgStage == "earlylate"]
  latestageExpr <- latestageExpr[,ELgene$ProgStage == "lateOnly" | ELgene$ProgStage == "earlylate"]
  # remove version number
  Geneanno$id <- sapply(strsplit(as.character(Geneanno$id), split = "\\."), function(x) x[1])
  # transform gene symbol to gene name
  id_to_symbol <- setNames(Geneanno$gene, Geneanno$id)
  colnames(earlystageExpr) <- id_to_symbol[colnames(earlystageExpr)]
  colnames(latestageExpr) <- id_to_symbol[colnames(latestageExpr)]
  list(earlystageExpr = earlystageExpr, latestageExpr = latestageExpr)
}
# call function
KIRPdata <- process_expression_data(KIRPearlystageExpr, KIRPlatestageExpr, KIRPELgene, Geneanno)
KIRCdata <- process_expression_data(KIRCearlystageExpr, KIRClatestageExpr, KIRCELgene, Geneanno)
```


## [Function] Function`perform_ssGSEA` : Single sample Gene Set Enrichment Analysis function
```r
perform_ssGSEA <- function(cancertype, earlystageExpr, latestageExpr, hallmarkgspath,database) {
  hallmark.gs <- getGmt(paste0(hallmarkgspath))
  hallmark.gs <- as.list(hallmark.gs)
  # convert to matrix
  earlystageExpr <- as.matrix(earlystageExpr)
  latestageExpr <- as.matrix(latestageExpr)
  earlystageExpr <- apply(earlystageExpr, 2, function(x) as.numeric(as.character(x)))
  latestageExpr <- apply(latestageExpr, 2, function(x) as.numeric(as.character(x)))
  # Transposing the matrix
  earlystageExpr <- t(earlystageExpr)
  latestageExpr <- t(latestageExpr)
  # ssGSEA save as csv
  ssgsea.resultsE <- gsva(earlystageExpr, hallmark.gs, method="ssgsea")
  ssgsea.resultsL <- gsva(latestageExpr, hallmark.gs, method="ssgsea") 
  # Transposing the matrix
  ssgsea.resultsE2 <- t(ssgsea.resultsE)
  ssgsea.resultsL2 <- t(ssgsea.resultsL)
  # as data frame
  ssgsea.resultsE3 <- as.data.frame(ssgsea.resultsE2)
  ssgsea.resultsL3 <- as.data.frame(ssgsea.resultsL2)
  # select interested columns between ssgsea.resultsE3 and ssgsea.resultsL3
  common_cols <- intersect(colnames(ssgsea.resultsE3), colnames(ssgsea.resultsL3))
  # Optionally, you can subset both data frames to keep only the common columns
  ssgsea.resultsE3 <- ssgsea.resultsE3[, common_cols]
  ssgsea.resultsL3 <- ssgsea.resultsL3[, common_cols]
  # add column stage 
  ssgsea.resultsE3$Stage <- "Early"
  ssgsea.resultsL3$Stage <- "Late"
  # combine early and late dataframes
  print(ncol(ssgsea.resultsE3))
  print(ncol(ssgsea.resultsL3))
  results <- rbind(ssgsea.resultsE3, ssgsea.resultsL3)
  write.csv(ssgsea.resultsE, file=paste0("Output/", cancertype,database, "earlystagessgseaValueTable.csv"))
  write.csv(ssgsea.resultsL, file=paste0("Output/", cancertype,database, "latestagessgseaValueTable.csv"))
  write.csv(results, file=paste0("Output/", cancertype,database, "EarlylateStagessgseaValueTable.csv"))
  list(ssgsea.resultsE = ssgsea.resultsE, ssgsea.resultsL = ssgsea.resultsL, results = results)
}



# call function
KIRPgsearesult <- perform_ssGSEA("KIRP", KIRPdata$earlystageExpr, KIRPdata$latestageExpr, "Data/h.all.v2023.2.Hs.symbols.gmt", "hallmark")
KIRCgsearesult <- perform_ssGSEA("KIRC", KIRCdata$earlystageExpr, KIRCdata$latestageExpr, "Data/h.all.v2023.2.Hs.symbols.gmt", "hallmark")
KIRPgsearesultbp <- perform_ssGSEA("KIRP", KIRPdata$earlystageExpr, KIRPdata$latestageExpr, "Data/c5.go.bp.v2023.2.Hs.symbols.gmt", "bp")
KIRCgsearesultbp <- perform_ssGSEA("KIRC", KIRCdata$earlystageExpr, KIRCdata$latestageExpr, "Data/c5.go.bp.v2023.2.Hs.symbols.gmt", "bp")
KIRPgsearesultcp <- perform_ssGSEA("KIRP", KIRPdata$earlystageExpr, KIRPdata$latestageExpr, "Data/c2.cp.v2023.2.Hs.symbols.gmt", "cp")
KIRCgsearesultcp <- perform_ssGSEA("KIRC", KIRCdata$earlystageExpr, KIRCdata$latestageExpr, "Data/c2.cp.v2023.2.Hs.symbols.gmt", "cp")
```

## [Function] Function`calculate_statistic` : Wilcoxon rank sum p-value function
```r
calculate_statistic <- function(cancertype, ssgsea_resultsE, ssgsea_resultsL, database) {
  p_values <- numeric(nrow(ssgsea_resultsE))
  effect_sizes <- numeric(nrow(ssgsea_resultsE))
  Median_difLtoE <- numeric(nrow(ssgsea_resultsE))
  for (i in seq_len(nrow(ssgsea_resultsE))) {
    test_result <- wilcox.test(ssgsea_resultsE[i, ], ssgsea_resultsL[i, ])
    p_values[i] <- test_result$p.value
    # create a data frame for wilcox_effsize
    data <- data.frame(
      ES = c(ssgsea_resultsE[i, ], ssgsea_resultsL[i, ]),
      Stage = c(rep("Early", length(ssgsea_resultsE[i, ])), rep("Late", length(ssgsea_resultsL[i, ])))
    ) 
    # calculate effect size
    effect_sizes[i] <- wilcox_effsize(ES ~ Stage, data = data, paired = FALSE, conf.level = 0.95)$effsize  
    # calculate median difference
    Median_difLtoE[i] <-  median(ssgsea_resultsL[i, ]) - median(ssgsea_resultsE[i, ]) 
  }
  p_values <- p.adjust(p_values, method = "BH")
  hallmark_p_values <- data.frame(Hallmark = rownames(ssgsea_resultsE), P_Value = p_values, Wilcox_effsize = effect_sizes, Median_difLtoE = Median_difLtoE)
  # Selecting pathways with adjusted p-value < 0.05
  hallmark_p_values <- subset(hallmark_p_values, P_Value < 0.05)
  # Sorting the pathways by adjusted p-value
  hallmark_p_values <- hallmark_p_values[order(hallmark_p_values$Wilcox_effsize, decreasing = TRUE), ]
  # Saving the results
  write.csv(hallmark_p_values, file = paste0("Output/", cancertype,database ,"EarlylateStagessgseaPvalues.csv"), row.names = FALSE)
}

# call function
calculate_statistic("KIRP", KIRPgsearesult$ssgsea.resultsE, KIRPgsearesult$ssgsea.resultsL, "hallmark")
calculate_statistic("KIRC", KIRCgsearesult$ssgsea.resultsE, KIRCgsearesult$ssgsea.resultsL, "hallmark")
calculate_statistic("KIRP", KIRPgsearesultbp$ssgsea.resultsE, KIRPgsearesultbp$ssgsea.resultsL, "bp")
calculate_statistic("KIRC", KIRCgsearesultbp$ssgsea.resultsE, KIRCgsearesultbp$ssgsea.resultsL, "bp")
calculate_statistic("KIRP", KIRPgsearesultcp$ssgsea.resultsE, KIRPgsearesultcp$ssgsea.resultsL, "cp")
calculate_statistic("KIRC", KIRCgsearesultcp$ssgsea.resultsE, KIRCgsearesultcp$ssgsea.resultsL, "cp")
```



# [Plot-the-Pathway-enrich] Plotting the box plots for the top 10 pathways (Compare the ES scores between the early and late stages)
## Environment
```shell
conda activate python3_10 
cd /home/emily2835/EarlyLateStageProject/PathwayEnrichment_ssGSEA
```
## Import packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## [Function] Function`plot_boxplot` :Plotting the box plots for the top 10 pathways (Compare the ES scores between the early and late stages)
```python 
def plot_boxplot(cancer_type, database,index):
    data_file = f"Output/{cancer_type}{database}EarlylateStagessgseaValueTable.csv"
    stat_file = f"Output/{cancer_type}{database}EarlylateStagessgseaPvalues.csv"
    data = pd.read_csv(data_file, index_col=0)
    stat = pd.read_csv(stat_file, index_col=0)
    data_long = pd.melt(data, id_vars=['Stage'], value_vars=data.columns[:-1], var_name='Hallmark', value_name='ES')
    fig, ax = plt.subplots(figsize=(6, 8))
    ax = sns.boxplot(x="Hallmark", y="ES", hue="Stage", data=data_long[data_long['Hallmark'] == stat.index[index]], width=0.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=360)
    ax.set_ylabel("Enrichment Score (ES)")
    ax.set_title(stat.index[index])
    text_str = f"Wilcox effsize: {stat['Wilcox_effsize'][index]:.2f}\np-value: {stat['P_Value'][index]:.2e}\nMedian difference: {stat['Median_difLtoE'][index]:.2f}"
    ax.text(0.03, 0.03, text_str, transform=ax.transAxes, fontsize=10)
    plt.savefig(f"Figure/{cancer_type}" + stat.index[index] + database + "Barplot.png", dpi=300)
    plt.savefig(f"Figure/{cancer_type}" + stat.index[index] + database + "Barplot.svg", dpi=300)
    plt.close()


# call function
for i in range(1,20):
  plot_boxplot("KIRP", "hallmark", i)
for i in range(1,20):
  plot_boxplot("KIRC", "bp", i)
for i in range(1,20):
  plot_boxplot("KIRP", "cp", i)
```
# END
