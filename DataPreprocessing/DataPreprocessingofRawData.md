# Data preprocessing of raw data
Author : Huang,Shu-jing.
Date : 2022/08/24

- [Data preprocessing of raw data](#data-preprocessing-of-raw-data)
  - [1.1. Environment](#11-environment)
  - [1.2. Import package](#12-import-package)
  - [1.3. Import data](#13-import-data)
- [2. Data preprocessing](#2-data-preprocessing)
  - [2.1. Stage classification](#21-stage-classification)
  - [2.2. Romove missing data](#22-romove-missing-data)
  - [2.3. Add a new column stage\_group to indicate the stage\_I, stage\_II, stage\_III, stage\_IV](#23-add-a-new-column-stage_group-to-indicate-the-stage_i-stage_ii-stage_iii-stage_iv)
  - [2.4. Add early\_stage and late\_stage](#24-add-early_stage-and-late_stage)
  - [2.5. Set and transpose the expression data](#25-set-and-transpose-the-expression-data)
  - [2.6. Combine the expr\_data\_sur with clin\_data\_sur](#26-combine-the-expr_data_sur-with-clin_data_sur)
  - [2.7. Save the combined data as csv file](#27-save-the-combined-data-as-csv-file)
  - [2.8. Select the data base on the cancer type](#28-select-the-data-base-on-the-cancer-type)
  - [2.9. Save the combined data as csv file](#29-save-the-combined-data-as-csv-file)
  - [Select sample for different cancer type](#select-sample-for-different-cancer-type)
- [3. Plotting the sample size of different stage](#3-plotting-the-sample-size-of-different-stage)
  - [3.1. Environment](#31-environment)
  - [3.2. Import package](#32-import-package)
  - [3.3. Stage I and Stage II sample size bar plot](#33-stage-i-and-stage-ii-sample-size-bar-plot)


## 1.1. Environment 
```shell
conda activate basic_work
cd /home/emily2835/stage_project_git3/DataPreprocessing
```
## 1.2. Import package 
```
library(tidyverse)
library(magrittr)
```
## 1.3. Import data
```
expr_data<-readRDS("Data/TCGAPanCancer_tcgageneexpectedcount.rds")
clin_data<-readRDS("Data/Survival_SupplementalTable_S1_20171025_xena_sp.rds")
```
# 2. Data preprocessing
## 2.1. Stage classification
unique(clin_data$ajcc_pathologic_tumor_stage)
```
 [1] "[Unknown]"  "I/II NOS"   "IS"         "Stage 0"    "Stage I"
 [6] "Stage IA"   "Stage IB"   "Stage II"   "Stage IIA"  "Stage IIB"
[11] "Stage IIC"  "Stage III"  "Stage IIIA" "Stage IIIB" "Stage IIIC"
[16] "Stage IV"   "Stage IVA"  "Stage IVB"  "Stage IVC"  "Stage X"
```
```r
Stage_I<-c("Stage I","Stage IA","Stage IB")
Stage_II<-c("Stage II","Stage IIA","Stage IIB","Stage IIC")
Stage_III<-c("Stage III","Stage IIIA","Stage IIIB","Stage IIIC")
Stage_IV<-c("Stage IV","Stage IVA","Stage IVB","Stage IVC")
```
## 2.2. Romove missing data
```r
clin_data_sub<-clin_data %>% 
    filter(str_sub(clin_data$sample,14, 15) %in% c("01"), OS != 'NA', OS.time != 'NA', OS.time != '0')
```
## 2.3. Add a new column stage_group to indicate the stage_I, stage_II, stage_III, stage_IV
```r
clin_data_sub <- clin_data_sub %>% 
    mutate(stage_group = case_when(
        ajcc_pathologic_tumor_stage %in% Stage_I ~ "Stage_I",
        ajcc_pathologic_tumor_stage %in% Stage_II ~ "Stage_II",
        ajcc_pathologic_tumor_stage %in% Stage_III ~ "Stage_III",
        ajcc_pathologic_tumor_stage %in% Stage_IV ~ "Stage_IV",
        TRUE ~ "Stage_Unknown"
    ))
```
## 2.4. Add early_stage and late_stage
```r
clin_data_sub <- clin_data_sub %>% 
    mutate(E_L_Stage = case_when(
        ajcc_pathologic_tumor_stage %in% Stage_I ~ "early_stage",
        ajcc_pathologic_tumor_stage %in% Stage_II ~ "early_stage",
        TRUE ~ "late_stage"
    ))
```
## 2.5. Set and transpose the expression data
```r
expr_data_pre <- expr_data %>%
    select(c(sample,any_of(clin_data_sub$sample))) %>% 
    t() %>%
    as.data.frame()
colnames(expr_data_pre)<-expr_data_pre[1,]
expr_data_pre<-expr_data_pre[-1,]
expr_data_pre$sample<-rownames(expr_data_pre)
```
## 2.6. Combine the expr_data_sur with clin_data_sur
```r
com_expr_clin_data <- expr_data_pre %>%
    left_join(clin_data_sub, by= "sample")

row.names(com_expr_clin_data)<-com_expr_clin_data$sample
```
## 2.7. Save the combined data as csv file
```r
write.csv(com_expr_clin_data, file = "Output/TCGAPanCancerExprandClin.csv")
```
## 2.8. Select the data base on the cancer type
<8 cancer types>
LIHC/LUAD/STAD/COAD/ESCA/KIRC/KIRP/LUSC
```r
com_expr_clin_data_sub <- com_expr_clin_data %>% 
    filter(`cancer type abbreviation` %in% c("LIHC","LUAD","STAD","COAD","ESCA","KIRC","KIRP","LUSC"))
```

## 2.9. Save the combined data as csv file
```r
write.csv(com_expr_clin_data_sub, file = "Output/TCGAPanCancerExprandClin8cancertype.csv")
```
## Select sample for different cancer type
```r
com_expr_clin_data_sub2 <- com_expr_clin_data_sub %>% 
    select(`cancer type abbreviation`,E_L_Stage)
```
```r
write.csv(com_expr_clin_data_sub2, file = "Output/TCGAPanCancer8cancersamplelist.csv")
```





# 3. Plotting the sample size of different stage
## 3.1. Environment 
```shell
conda activate python3_10
cd /home/emily2835/stage_project_git3/DataPreprocessing
```
## 3.2. Import package 
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```
## 3.3. Stage I and Stage II sample size bar plot
```python
# 1. Read the CSV file with pandas
expr_clin_data = pd.read_csv("Output/TCGAPanCancerExprandClin8cancertype.csv",low_memory=False)
# 2. Process the data to count samples for each cancer type and stage
plottable = expr_clin_data.groupby(['cancer type abbreviation', 'E_L_Stage']).size().reset_index(name='n').fillna(0)
# Pivot the data to get stages as columns
plottable = plottable.pivot(index='cancer type abbreviation', columns='E_L_Stage', values='n').fillna(0)
# 3. Plotting
colors = ['#D66965', '#9BCBEB']  # Add more colors if there are more stages
ax = plottable.plot(kind='bar', stacked=True, color=colors, figsize=(20, 10))
# 5. Annotate bars with the sum of Stage I and Stage II samples
for index, row in plottable.iterrows():
    total_samples = row.sum()
    ax.annotate(f'{int(total_samples)}', 
                (list(plottable.index).index(index), total_samples),
                ha='center', va='bottom',
                color='black', size=10)         
# Adjusting the visuals using matplotlib
plt.title("Sample Size of Early-Stage and Late-Stage", loc='center') 
plt.xlabel("Cancer type")
plt.ylabel("Sample size")
plt.legend(title='Stage status')
plt.tight_layout()
# Save the figure
plt.savefig("Figure/SampleSizeofEarlyStageandLateStage.png", dpi=300)
```