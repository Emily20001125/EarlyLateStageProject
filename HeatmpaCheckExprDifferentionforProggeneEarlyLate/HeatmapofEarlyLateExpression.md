# Prognostic Gene Summary
Author: Huang,Shu-Jing
Date: 2023-11-24

## environment
```shell
conda activate python3_10
cd /home/emily2835/stage_project_git3/HeatmpaCheckExprDifferentionforProggeneEarlyLate
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
# Combine earlyOnly and lateOnly
COAD_Expr_sub = COAD_Expr[COAD_ELgene[COAD_ELgene['ProgStage']=='earlylate']['gene']]
# Log2 transformation
COAD_Expr_sub = np.log2(COAD_Expr_sub+1)
# Plotting Heatmap
# Prepare a vector of color mapped to the 'E_L_Stage' column
my_palette = dict(zip(COAD_Expr.E_L_Stage.unique(), ["red","blue"]))
row_colors = COAD_Expr.E_L_Stage.map(my_palette)
plt.figure(figsize=(10,10))
sns.clustermap(COAD_Expr_sub,metric="correlation", cmap="seismic", row_colors=row_colors,row_cluster=False,col_cluster=True,z_score=1)
plt.savefig('Figure/COAD_earlylate_row_clusterFalsecol_clusterTruegene_expression_heatmap.png')
plt.close()
```

```python
def plot_early_late_heatmap(cancer_type):
    # Load data
    EExpr = pd.read_csv(f'Data/{cancer_type}earlystageExprandClin.csv', index_col=0)
    LExpr = pd.read_csv(f'Data/{cancer_type}latestageExprandClin.csv', index_col=0)
    ELgene = pd.read_csv(f'Data/{cancer_type}earlylatelabelCoxProggene005.csv', index_col=0)
    # Combine early and late stage
    Expr = pd.concat([EExpr,LExpr],axis=0)
    # Combine earlyOnly and lateOnly
    Expr_sub = Expr[ELgene[ELgene['ProgStage']=='earlylate']['gene']]
    # Log2 transformation
    Expr_sub = np.log2(Expr_sub+1)
    # Plotting Heatmap
    # Prepare a vector of color mapped to the 'E_L_Stage' column
    my_palette = dict(zip(Expr.E_L_Stage.unique(), ["red","blue"]))
    row_colors = Expr.E_L_Stage.map(my_palette)
    plt.figure(figsize=(10,10))
    sns.clustermap(Expr_sub,metric="correlation", cmap="seismic", row_colors=row_colors,row_cluster=False,col_cluster=True,z_score=1)
    plt.savefig(f'Figure/{cancer_type}_earlylate_row_clusterFalsecol_clusterTruegene_expression_heatmap.png')
    plt.close()

for cancer_type in ['COAD','KIRC','KIRP','LIHC','LUAD','LUSC','STAD','ESCA']:
    plot_early_late_heatmap(cancer_type)
```

## All Proggene Heatmap
```python
def plot_Allproggene_heatmap(cancer_type):
    # Load data
    EExpr = pd.read_csv(f'Data/{cancer_type}earlystageExprandClin.csv', index_col=0)
    LExpr = pd.read_csv(f'Data/{cancer_type}latestageExprandClin.csv', index_col=0)
    ELgene = pd.read_csv(f'Data/{cancer_type}earlylatelabelCoxProggene005.csv', index_col=0)
    # Combine early and late stage
    Expr = pd.concat([EExpr,LExpr],axis=0)
    # Extract gene expression from ELgene
    Expr_sub = Expr[ELgene['gene']]
    # Log2 transformation
    Expr_sub = np.log2(Expr_sub+1)
    # Plotting Heatmap
    # Prepare a vector of color mapped to the 'E_L_Stage' column
    my_palette = dict(zip(Expr.E_L_Stage.unique(), ["red","blue"]))
    row_colors = Expr.E_L_Stage.map(my_palette)
    plt.figure(figsize=(10,10))
    sns.clustermap(Expr_sub,metric="correlation", cmap="seismic", row_colors=row_colors,row_cluster=False,col_cluster=True,z_score=1)
    plt.savefig(f'Figure/{cancer_type}_Allproggene_row_clusterFalsecol_clusterTruegene_expression_heatmap.png')
    plt.close()

for cancer_type in ['COAD','KIRC','KIRP','LIHC','LUAD','LUSC','STAD','ESCA']:
    plot_Allproggene_heatmap(cancer_type)
```