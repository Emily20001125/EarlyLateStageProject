# Raw data download
Author : Huang,Shu-jing
Date : 2023/11/22

- [Raw data download](#raw-data-download)
  - [1.1. Environment](#11-environment)
  - [1.2. Import package](#12-import-package)
  - [1.3. Download data - clinical data and expression data](#13-download-data---clinical-data-and-expression-data)
  - [1.3. Download data - Mutation data](#13-download-data---mutation-data)

## 1.1. Environment 
```shell
conda activate basic_work
cd stage_project_git/Raw_data
```

## 1.2. Import package 
```r
library(tidyverse)
library(magrittr)
#install.packages('UCSCXenaTools')
library(UCSCXenaTools)
```

## 1.3. Download data - clinical data and expression data
```r
# set the VROOM_CONNECTION_SIZE
Sys.setenv("VROOM_CONNECTION_SIZE"=500072)
# Download clinical data from UCSC Xena
datasetID = 'Survival_SupplementalTable_S1_20171025_xena_sp'

file_op3 = 
    XenaGenerate() %>% 
    XenaFilter(filterDatasets = datasetID)
file_op3@datasets

clin_data = 
    file_op3 %>%
    XenaQuery() %>%
    XenaDownload() %>%
    XenaPrepare()

head(clin_data)

datasetID = 'tcga_gene_expected_count'
file_op3 = 
    XenaGenerate() %>% 
    XenaFilter(filterDatasets = datasetID)
file_op3@datasets

expr_data = 
    file_op3 %>%
    XenaQuery() %>%
    XenaDownload() %>%
    XenaPrepare()

head(expr_data)
```
Save clinical file as rds file
```r
# save clinical file as rds file
saveRDS(clin_data, file = "Survival_SupplementalTable_S1_20171025_xena_sp.rds")
saveRDS(expr_data,file="TCGAPanCancer_tcgageneexpectedcount.rds")
```

## 1.3. Download data - Mutation data
```r
# set the VROOM_CONNECTION_SIZE
Sys.setenv("VROOM_CONNECTION_SIZE"=500072)
# Download clinical data from UCSC Xena
datasetID = 'mc3.v0.2.8.PUBLIC.nonsilentGene.xena'

file_op3 = 
    XenaGenerate() %>% 
    XenaFilter(filterDatasets = datasetID)
file_op3@datasets

Mut_data = 
    file_op3 %>%
    XenaQuery() %>%
    XenaDownload() %>%
    XenaPrepare()

head(Mut_data)

# save Mutation file as rds file
write.csv(Mut_data,file="TCGAPanCancermc3SilentGeneMut.csv")
```
