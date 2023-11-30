# Heatmap between early and late stage
Data:2023/11/29
## 0.1. Environment 
```shell
conda activate python3_10
cd /home/emily2835/stage_project_git3/HeatmapCheckExprDifferentiation
```
## 0.2. Import package 
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
## Import data
```python
COADEarly = pd.read_csv("Data/COADearlystageExprandClin.csv",index_col=0)
COADLate = pd.read_csv("Data/COADlatestageExprandClin.csv",index_col=0)
```
```
>>> COAD
              sample                X     X_PATIENT  ... ENSG00000009694  ENSG00000123685 ENSG00000105063
1    TCGA-3L-AA1B-01  TCGA-3L-AA1B-01  TCGA-3L-AA1B  ...        8.686461        31.851892     5297.191301
2    TCGA-4T-AA8H-01  TCGA-4T-AA8H-01  TCGA-4T-AA8H  ...        0.000000        40.419007     7766.347315
3    TCGA-5M-AATE-01  TCGA-5M-AATE-01  TCGA-5M-AATE  ...        2.896772        36.207205     6715.499079
4    TCGA-A6-2675-01  TCGA-A6-2675-01  TCGA-A6-2675  ...       11.061385        90.703031     5633.764413
```

## 0.3. Data preprocessing
```python
# Combine early and late stage
COAD = pd.concat([COADEarly,COADLate],axis=0)
```

## Plotting Heatmap
```python
# Extract gene expression use regular expression "start with ENSG"
COAD_gene = COAD.filter(regex="^ENSG",axis=1)
# Log2 transformation
COAD_gene = np.log2(COAD_gene+1)
# Plotting Heatmap
# Prepare a vector of color mapped to the 'E_L_Stage' column
my_palette = dict(zip(COAD.E_L_Stage.unique(), ["red","blue"]))
row_colors = COAD.E_L_Stage.map(my_palette)

plt.figure(figsize=(10,10))
sns.clustermap(COAD_gene, z_score=1 ,metric="correlation", cmap="seismic", row_colors=row_colors,row_cluster=False,col_cluster=True)
plt.savefig('Output/row_clusterFalsecol_clusterTruegene_expression_heatmap.png')
plt.close()
```

## Function `plot_gene_expression_heatmap`
```python
def plot_gene_expression_heatmap(early_stage_file, late_stage_file, output_file):
    # 讀取數據
    early_stage_data = pd.read_csv(early_stage_file, index_col=0)
    late_stage_data = pd.read_csv(late_stage_file, index_col=0)
    data = pd.concat([early_stage_data, late_stage_data], axis=0)
    # 提取基因表達數據
    gene_data = data.filter(regex="^ENSG", axis=1)
    # Log2轉換
    gene_data = np.log2(gene_data + 1)
    # 準備顏色映射
    my_palette = dict(zip(data.E_L_Stage.unique(), ["red", "blue"]))
    row_colors = data.E_L_Stage.map(my_palette)
    # 繪製熱圖
    plt.figure(figsize=(10, 10))
    sns.clustermap(gene_data ,metric="correlation", cmap="seismic", row_colors=row_colors,row_cluster=False,col_cluster=True,z_score=1)
    #sns.clustermap(gene_data,z_score=1, metric="correlation", cmap="seismic", row_colors=row_colors)
    # 儲存圖片
    plt.savefig(output_file)
    plt.close()
```
## Call the function `plot_gene_expression_heatmap`
```python 
# cluster row and column
for cancer_type in ["COAD", "ESCA", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "STAD"]:
    early_stage_file = f"Data/{cancer_type}earlystageExprandClin.csv"
    late_stage_file = f"Data/{cancer_type}latestageExprandClin.csv"
    output_file = f"Output/{cancer_type}exprheatmap.png"
    plot_gene_expression_heatmap(early_stage_file, late_stage_file, output_file)
# cluster col only
for cancer_type in ["COAD", "ESCA", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "STAD"]:
    early_stage_file = f"Data/{cancer_type}earlystageExprandClin.csv"
    late_stage_file = f"Data/{cancer_type}latestageExprandClin.csv"
    output_file = f"Output/{cancer_type}clustercolonlyexprheatmap.png"
    plot_gene_expression_heatmap(early_stage_file, late_stage_file, output_file)
```