# Gene Annotation Table Data Error Remove
Author : Huang,Shu-jing
Date : 2024/01/17

# Python Test
## Environment 
```shell
cd /home/emily2835/EarlyLateStageProject/Annotation_DateErrorRemove
conda activate python3_10
```

## Import Package
```python
import pandas as pd
import numpy as np
from datetime import datetime
```

## Import Data
```python
# import data with excel format (which may change some gene name to date)
df_excel = pd.read_excel('Data/test.xlsx')
# import data with probemap format 
df_probemap = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap',sep='\t')
# select the gene name may change to date
df_date = df_excel.gene[df_excel.gene.apply(lambda x: isinstance(x,datetime))].index
# Select the dategene in df2
df_probemap.loc[df_date]
# save csv
df_probemap.loc[df_date].to_csv('Output/probeMap_gencode.csv',index=False)
```

# R Test
## Environment 
```shell
cd /home/emily2835/EarlyLateStageProject/Annotation_DateErrorRemove
conda activate basic_work
```
## Import Package
```r
library(tidyverse)
```
## Import Data
```r
df_excel <- read_excel('Data/test.xlsx')
df_probemap_csv <- read.csv('Data/probeMap_gencode.v23.annotation.gene.probemap',sep='\t')
df_probemap_table <- read.table('Data/probeMap_gencode.v23.annotation.gene.probemap',sep='\t')
```