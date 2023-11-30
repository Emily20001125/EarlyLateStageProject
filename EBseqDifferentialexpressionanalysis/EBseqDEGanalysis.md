# Prognostic Gene Summary
Author: Huang,Shu-Jing
Date: 2023-11-27

## environment
```shell
conda create --name EBSeq
conda activate EBSeq
cd /home/emily2835/stage_project_git3/EBseqDifferentialexpressionanalysis
```
## Import packages
```R
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
BiocManager::install("EBSeq")
install.packages("magrittr")
install.packages("tidyverse")
install.packages("tibble")
library(EBSeq)
library(tidyverse)
library(magrittr)
library(tibble)
```

## Function`process_cancer_data` : intersect of earlyProg and lateProg
```R
process_cancer_data <- function(cancer_type) {
  # Construct the file paths
  early_stage_file = paste0("Data/", cancer_type, "earlystageExprandClin.csv")
  late_stage_file = paste0("Data/", cancer_type, "latestageExprandClin.csv")
  # Read the data
  Eexpr = read.csv(early_stage_file, row.names = 1)
  Lexpr = read.csv(late_stage_file, row.names = 1)
  ELgene = read.csv(paste0("Data/", cancer_type, "earlylatelabelCoxProggene005.csv"))
  # Add stage label
  Eexpr$E_L_Stage = "early"
  Lexpr$E_L_Stage = "late"
  # Combine early and late stage
  Expr = rbind(Eexpr, Lexpr)
  # Extract gene expression use regular expression for ELgene-earlyOnly and ELgene-lateOnly
  earlyOnly_genes = ELgene$gene[ELgene$ProgStage == 'earlyOnly']
  Eselect_gene = c(earlyOnly_genes, "E_L_Stage")
  lateOnly_genes = ELgene$gene[ELgene$ProgStage == 'lateOnly']
  Lselect_gene = c(lateOnly_genes, "E_L_Stage")
  Expr_Esub = Expr[,Eselect_gene]
  Expr_Lsub = Expr[,Lselect_gene]
  # Combine earlyOnly and lateOnly
  Expr_sub = cbind(Expr_Esub,Expr_Lsub)
  print(cancer_type + " is done")
  return(Expr_sub)
}

# Call function
COAD_Expr_sub = process_cancer_data("COAD")
KIRC_Expr_sub = process_cancer_data("KIRC")
KIRP_Expr_sub = process_cancer_data("KIRP")
LIHC_Expr_sub = process_cancer_data("LIHC")
LUAD_Expr_sub = process_cancer_data("LUAD")
LUSC_Expr_sub = process_cancer_data("LUSC")
STAD_Expr_sub = process_cancer_data("STAD")
ESCA_Expr_sub = process_cancer_data("ESCA")
```

## Function`Prog_differential_expression_analysis` : Differential expression analysis
```R
Prog_differential_expression_analysis <- function(cancer_type, Expr_sub,FDRvalue) {
  # Set condition from E_L_Stage column in Expr early and late stage label
  Conditions = Expr_sub$E_L_Stage %>% as.factor()
  # Delete last column
  Expr_sub = Expr_sub[,-ncol(Expr_sub)]
  # Transpose and Convert to numeric matrix
  Expr_sub[] <- lapply(Expr_sub, function(x) as.numeric(as.character(x)))
  Expr_sub <- t(as.matrix(Expr_sub))
  Expr_sub <- Expr_sub[complete.cases(Expr_sub), ]
  # Size factor
  Sizes = QuantileNorm(Expr_sub,.75)
  # DEG analysis
  EBOut = EBTest(Data = Expr_sub, Conditions = Conditions, sizeFactors = Sizes, maxround = 5)
  # Extract DEG pvalue < 0.05
  EBDERes = GetDEResults(EBOut, FDR=FDRvalue)
  EBDERes = as.data.frame(EBDERes$ DEfound)
  # Add gene log2FC
  GeneFC=PostFC(EBOut)
  GeneFC = as.data.frame(GeneFC)
  # Combine DEG and log2FC
  colnames(EBDERes) = "DEfound"
  GeneFC$Gene = rownames(GeneFC)
  EBDEResM = merge(EBDERes,GeneFC,by.x="DEfound",by.y="Gene")
  # Write to csv
  if (FDRvalue == 0.05) {
    output_file = paste0("Output/", cancer_type, "ELProggeneDEGFDR005.csv")
  } else {
  output_file = paste0("Output/", cancer_type, "ELProggeneDEGFDR01.csv")
  write.csv(EBDEResM, output_file)
  print(paste0(cancer_type, " is done"))
}

# Call function
differential_expression_analysis("COAD", COAD_Expr_sub,0.05)
differential_expression_analysis("KIRC", KIRC_Expr_sub,0.05)
differential_expression_analysis("KIRP", KIRP_Expr_sub,0.05)
differential_expression_analysis("LIHC", LIHC_Expr_sub,0.05) # Non DEG
differential_expression_analysis("LUAD", LUAD_Expr_sub,0.05) # Non DEG
differential_expression_analysis("LUSC", LUSC_Expr_sub,0.05) # Non DEG
differential_expression_analysis("STAD", STAD_Expr_sub,0.05) # Non DEG
differential_expression_analysis("ESCA", ESCA_Expr_sub,0.05)  
differential_expression_analysis("COAD", COAD_Expr_sub,0.1)
differential_expression_analysis("KIRC", KIRC_Expr_sub,0.1)
differential_expression_analysis("KIRP", KIRP_Expr_sub,0.1)
differential_expression_analysis("LIHC", LIHC_Expr_sub,0.1) 
differential_expression_analysis("LUAD", LUAD_Expr_sub,0.1) 
differential_expression_analysis("LUSC", LUSC_Expr_sub,0.1) 
differential_expression_analysis("STAD", STAD_Expr_sub,0.1) 
differential_expression_analysis("ESCA", ESCA_Expr_sub,0.1)  
```

## `All_differential_expression_analysis` : Differential expression analysis for all genes
```R
All_differential_expression_analysis <- function(cancer_type) {
  Eexpr = read.csv(paste0("Data/", cancer_type, "earlystageExprandClin.csv"),row.names = 1)
  Lexpr = read.csv(paste0("Data/", cancer_type, "latestageExprandClin.csv"),row.names = 1)
  # Combine early and late stage
  # Add stage label
  Eexpr$E_L_Stage = "early"
  Lexpr$E_L_Stage = "late"
  # Combine early and late stage
  Expr = rbind(Eexpr,Lexpr)
  # Set condition from E_L_Stage column in Expr early and late stage label
  Conditions = Expr$E_L_Stage %>% as.factor
  # Select colnames start at ENSG
  Expr = Expr[,grep("^ENSG",colnames(Expr))] 
  # Transpose and Convert to numeric matrix
  Expr[] <- lapply(Expr, as.numeric)
  Expr <- t(as.matrix(Expr))
  # Expr <- Expr[complete.cases(Expr), ]
  # Size factor
  Sizes = QuantileNorm(Expr,.75)
  # DEG analysis
  EBOut = EBTest(Data = Expr, Conditions = Conditions, sizeFactors = Sizes, maxround = 5)
  # Extract DEG pvalue < 0.05
  EBDERes = GetDEResults(EBOut, FDR=0.1)
  EBDERes = as.data.frame(EBDERes$ DEfound)
  # Add gene log2FC
  GeneFC=PostFC(EBOut)
  GeneFC = as.data.frame(GeneFC)
  # Combine DEG and log2FC
  colnames(EBDERes) = "DEfound"
  GeneFC$Gene = rownames(GeneFC)
  EBDEResM = merge(EBDERes,GeneFC,by.x="DEfound",by.y="Gene")
  # Write to csv
  output_file = paste0("Output/", cancer_type, "ELgeneDEGFDR01.csv")
  write.csv(EBDEResM, output_file)
}

# Call function
COAD_Expr_DEGAll = All_differential_expression_analysis("COAD")
KIRC_Expr_DEGAll = All_differential_expression_analysis("KIRC")
KIRP_Expr_DEGAll = All_differential_expression_analysis("KIRP")
LIHC_Expr_DEGAll = All_differential_expression_analysis("LIHC")
LUAD_Expr_DEGAll = All_differential_expression_analysis("LUAD")
LUSC_Expr_DEGAll = All_differential_expression_analysis("LUSC")
STAD_Expr_DEGAll = All_differential_expression_analysis("STAD")
ESCA_Expr_DEGAll = All_differential_expression_analysis("ESCA")
```


## Plotting Heatmap
```shell
conda activate python3_10
cd /home/emily2835/stage_project_git3/EBseqDifferentialexpressionanalysis
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
```python
# Import data
KIRC_ELgene = pd.read_csv('Output/KIRCELProggeneDEGFDR005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
```
```python
# Extract gene expression use regular expression for KIRC_ELgene-earlyOnly and KIRC_ELgene-lateOnly
KIRC_Expr_sub = KIRC_Expr[KIRC_ELgene['DEfound']]
# Log2 transformation
KIRC_Expr_sub = np.log2(KIRC_Expr_sub+1)
# Plotting Heatmap
# Prepare a vector of color mapped to the 'E_L_Stage' column
my_palette = dict(zip(KIRC_Expr.E_L_Stage.unique(), ["red","blue"]))
row_colors = KIRC_Expr.E_L_Stage.map(my_palette)
plt.figure(figsize=(10,10))
sns.clustermap(KIRC_Expr_sub,metric="euclidean", cmap="RdBu_r", row_colors=row_colors,row_cluster=False,col_cluster=True,square=True,z_score=1)
plt.savefig('Output/KIRCrow_clusterFalsecol_clusterTruegene_expression_heatmap.png')
plt.close()
```



## Function`plot_cancer_heatmap` : Heatmap of earlyOnly, lateOnly
```python
# function
def plot_cancer_heatmap(cancer_type):
    # Import data
    ELgene = pd.read_csv(f'Output/{cancer_type}ELProggeneDEGFDR005.csv', index_col=0)
    Eexpr = pd.read_csv(f'Data/{cancer_type}earlystageExprandClin.csv', index_col=0)
    Lexpr = pd.read_csv(f'Data/{cancer_type}latestageExprandClin.csv', index_col=0)
    # Combine early and late stage
    Expr = pd.concat([Eexpr,Lexpr],axis=0)
    # Extract gene expression use regular expression for ELgene-earlyOnly and ELgene-lateOnly
    Expr_sub = Expr[ELgene['DEfound']]
    # Log2 transformation
    Expr_sub = np.log2(Expr_sub+1)
    # Plotting Heatmap
    # Prepare a vector of color mapped to the 'E_L_Stage' column
    my_palette = dict(zip(Expr.E_L_Stage.unique(), ["red","blue"]))
    row_colors = Expr.E_L_Stage.map(my_palette)
    plt.figure(figsize=(10,10))
    sns.clustermap(Expr_sub,metric="euclidean", cmap="RdBu_r", row_colors=row_colors,row_cluster=False,col_cluster=True,square=True,z_score=1)
    plt.savefig(f'Figure/{cancer_type}row_clusterFalsecol_clusterTruegene_expression_heatmap.png')
    plt.close()

# Call function
for i in ['KIRC', 'KIRP']:
    plot_cancer_heatmap(i)
```