# Prognostic Gene Summary
Author: Huang,Shu-Jing11
Date: 2023-11-24

## environment
```shell
conda activate python3_10
cd /home/emily2835/stage_project_git3/HeatmpaCheckExprDifferentionforProggeneEarlyonlyandLateonly
```
## Import packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
from matplotlib import pyplot as plt
```

## Function`plot_Heatmap` : Heatmap of earlyOnly, lateOnly
```python
#import data
COAD_ELgene = pd.read_csv('Data/COADearlylatelabelCoxProggene005.csv', index_col=0)
COAD_Eexpr = pd.read_csv('Data/COADearlystageExprandClin.csv', index_col=0)
COAD_Lexpr = pd.read_csv('Data/COADlatestageExprandClin.csv', index_col=0)
# Combine early and late stage
COAD_Expr = pd.concat([COAD_Eexpr,COAD_Lexpr],axis=0)
```

## Plotting Heatmap
```python
# Extract gene expression use regular expression for COAD_ELgene-earlyOnly and COAD_ELgene-lateOnly
COAD_Expr_Esub = COAD_Expr[COAD_ELgene[COAD_ELgene['ProgStage']=='earlyOnly']['gene']]
COAD_Expr_Lsub = COAD_Expr[COAD_ELgene[COAD_ELgene['ProgStage']=='lateOnly']['gene']]
# Combine earlyOnly and lateOnly
COAD_Expr_sub = pd.concat([COAD_Expr_Esub,COAD_Expr_Lsub],axis=1)
# Log2 transformation
COAD_Expr_sub = np.log2(COAD_Expr_sub+1)
# Plotting Heatmap
# Prepare a vector of color mapped to the 'E_L_Stage' column
my_palette = dict(zip(COAD_Expr.E_L_Stage.unique(), ["red","blue"]))
row_colors = COAD_Expr.E_L_Stage.map(my_palette)
plt.figure(figsize=(10,10))
sns.clustermap(COAD_Expr_sub,metric="correlation", cmap="seismic", row_colors=row_colors,row_cluster=False,col_cluster=True,z_score=1)
plt.savefig('Output/row_clusterFalsecol_clusterTruegene_expression_heatmap.png')
plt.close()
```

```python
def plot_heatmap(cancer_type):
    # 讀取數據
    ELgene_file = f'Data/{cancer_type}earlylatelabelCoxProggene005.csv'
    Eexpr_file = f'Data/{cancer_type}earlystageExprandClin.csv'
    Lexpr_file = f'Data/{cancer_type}latestageExprandClin.csv' 
    ELgene_data = pd.read_csv(ELgene_file, index_col=0)
    Eexpr_data = pd.read_csv(Eexpr_file, index_col=0)
    Lexpr_data = pd.read_csv(Lexpr_file, index_col=0)
    # 組合早期和晚期階段
    Expr_data = pd.concat([Eexpr_data, Lexpr_data], axis=0)
    # 提取基因表達式
    Expr_Esub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='earlyOnly']['gene']]
    Expr_Lsub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='lateOnly']['gene']]
    # 組合earlyOnly和lateOnly
    Expr_sub_data = pd.concat([Expr_Esub_data, Expr_Lsub_data], axis=1)
    # Log2轉換
    Expr_sub_data = np.log2(Expr_sub_data+1)
    # 繪製熱圖
    my_palette = dict(zip(Expr_data.E_L_Stage.unique(), ["red","blue"]))
    row_colors = Expr_data.E_L_Stage.map(my_palette)
    plt.figure(figsize=(10,10))
    sns.clustermap(Expr_sub_data, metric="correlation", cmap="seismic", row_colors=row_colors, row_cluster=False, col_cluster=True,z_score=1)
    #sns.clustermap(Expr_sub_data,z_score=1, metric="correlation", cmap="seismic", row_colors=row_colors)
    plt.savefig(f'Figure/{cancer_type}_sample_clusterFalse_geneclusterTruegene_expression_heatmap.png')
    #plt.savefig(f'Figure/{cancer_type}_sample_clusterTrue_geneclusterTruegene_expression_heatmap.png')
    plt.close()

for cancer_type in ["COAD", "ESCA", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "STAD"]:
    plot_heatmap(cancer_type)
```
## Varience of earlyOnly and lateOnly
```python
def plot_heatmap_var(cancer_type):
    # 讀取數據
    ELgene_file = f'Data/{cancer_type}earlylatelabelCoxProggene005.csv'
    Eexpr_file = f'Data/{cancer_type}earlystageExprandClin.csv'
    Lexpr_file = f'Data/{cancer_type}latestageExprandClin.csv'
    ELgene_data = pd.read_csv(ELgene_file, index_col=0)
    Eexpr_data = pd.read_csv(Eexpr_file, index_col=0)
    Lexpr_data = pd.read_csv(Lexpr_file, index_col=0) 
    # 組合早期和晚期階段
    Expr_data = pd.concat([Eexpr_data, Lexpr_data], axis=0) 
    # 提取基因表達式
    Expr_Esub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='earlyOnly']['gene']]
    Expr_Lsub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='lateOnly']['gene']]  
    # 組合earlyOnly和lateOnly
    Expr_sub_data = pd.concat([Expr_Esub_data, Expr_Lsub_data], axis=1)  
    # Log2轉換
    Expr_sub_data = np.log2(Expr_sub_data+1)
    # 計算變異數並選擇變異數最大的前1000個基因
    Expr_sub_var = Expr_sub_data.var(axis=0)
    Expr_sub_var = Expr_sub_var.sort_values(ascending=False)
    Expr_sub_var = Expr_sub_var[0:100]
    # 提取變異數最大的前1000個基因的表達式
    Expr_sub_var = Expr_sub_data[Expr_sub_var.index]
    # 繪製熱圖
    my_palette = dict(zip(Expr_data.E_L_Stage.unique(), ["red","blue"]))
    row_colors = Expr_data.E_L_Stage.map(my_palette)
    plt.figure(figsize=(10,10))
    sns.clustermap(Expr_sub_var, metric="correlation", cmap="seismic", row_colors=row_colors, row_cluster=False, col_cluster=True, z_score=1)
    plt.savefig(f'Figure/{cancer_type}Var100_row_clusterFalsecol_clusterTruegene_expression_heatmap.png')
    plt.close()

for cancer_type in ["COAD", "ESCA", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "STAD"]:
    plot_heatmap_var(cancer_type)
```



## environment
```shell
conda activate DimensionalityReduction
```
## Import packages 
```python
import umap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Function`plot_umap` : UMAP of earlyOnly, lateOnly
```python
def plot_umap(cancer_type):
    # 讀取數據
    ELgene_file = f'Data/{cancer_type}earlylatelabelCoxProggene005.csv'
    Eexpr_file = f'Data/{cancer_type}earlystageExprandClin.csv'
    Lexpr_file = f'Data/{cancer_type}latestageExprandClin.csv'
    ELgene_data = pd.read_csv(ELgene_file, index_col=0)
    Eexpr_data = pd.read_csv(Eexpr_file, index_col=0)
    Lexpr_data = pd.read_csv(Lexpr_file, index_col=0) 
    # 組合早期和晚期階段
    Expr_data = pd.concat([Eexpr_data, Lexpr_data], axis=0) 
    # 提取基因表達式
    Expr_Esub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='earlyOnly']['gene']]
    Expr_Lsub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='lateOnly']['gene']]  
    # 組合earlyOnly和lateOnly
    Expr_sub_data = pd.concat([Expr_Esub_data, Expr_Lsub_data], axis=1)  
    # Log2轉換
    Expr_sub_data = np.log2(Expr_sub_data+1)
    # Draw UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(Expr_sub_data)
    # Plot UMAP
    my_palette = dict(zip(Expr_data.E_L_Stage.unique(), ["red","blue"]))
    row_colors = Expr_data.E_L_Stage.map(my_palette)
    plt.figure(figsize=(10,10))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=Expr_data.E_L_Stage, palette=my_palette)
    plt.savefig(f'Figure/{cancer_type}_UMAP.png')
    plt.close()

for cancer_type in ["COAD", "ESCA", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "STAD"]:
    plot_umap(cancer_type)

```


## Function`plot_umap_var` : UMAP of earlyOnly, lateOnly for Top100 varience
```python
def plot_umap_var(cancer_type):
    # 讀取數據
    ELgene_file = f'Data/{cancer_type}earlylatelabelCoxProggene005.csv'
    Eexpr_file = f'Data/{cancer_type}earlystageExprandClin.csv'
    Lexpr_file = f'Data/{cancer_type}latestageExprandClin.csv'
    ELgene_data = pd.read_csv(ELgene_file, index_col=0)
    Eexpr_data = pd.read_csv(Eexpr_file, index_col=0)
    Lexpr_data = pd.read_csv(Lexpr_file, index_col=0) 
    # 組合早期和晚期階段
    Expr_data = pd.concat([Eexpr_data, Lexpr_data], axis=0) 
    # 提取基因表達式
    Expr_Esub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='earlyOnly']['gene']]
    Expr_Lsub_data = Expr_data[ELgene_data[ELgene_data['ProgStage']=='lateOnly']['gene']]  
    # 組合earlyOnly和lateOnly
    Expr_sub_data = pd.concat([Expr_Esub_data, Expr_Lsub_data], axis=1)  
    # Log2轉換
    Expr_sub_data = np.log2(Expr_sub_data+1)
    # 計算變異數並選擇變異數最大的前100個基因
    Expr_sub_var = Expr_sub_data.var(axis=0)
    Expr_sub_var = Expr_sub_var.sort_values(ascending=False)
    Expr_sub_var = Expr_sub_var[0:200]
    # 提取變異數最大的前100個基因的表達式
    Expr_sub_var = Expr_sub_data[Expr_sub_var.index]
    # Draw UMAP
    reducer = umap.UMAP()
    embedding = reducer.fit_transform(Expr_sub_var)
    # Plot UMAP
    my_palette = dict(zip(Expr_data.E_L_Stage.unique(), ["red","blue"]))
    row_colors = Expr_data.E_L_Stage.map(my_palette)
    plt.figure(figsize=(10,10))
    sns.scatterplot(x=embedding[:,0], y=embedding[:,1], hue=Expr_data.E_L_Stage, palette=my_palette)
    plt.savefig(f'Figure/{cancer_type}Var200_UMAP.png')
    plt.close()


for cancer_type in ["COAD", "ESCA", "KIRC", "KIRP", "LIHC", "LUAD", "LUSC", "STAD"]:
    plot_umap_var(cancer_type)
```


