# MachineLearning for StagecClassification
Author: Huang,Shu-Jing
Date: 2023-11-30

## environment
```shell
conda activate python3_10
cd /home/emily2835/EarlyLateStageProject/PanKidneycancerProggeneAnalysis
```
## 1. Data preparation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib_venn import venn2, venn3
from matplotlib import pyplot as plt
```
## 1.1 Import data
```python
# read data of KIRC and KIRP
KIRPearlylate = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRCearlylate = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
```
## Data preprocessing
```python
# Extract the row names of earlylate
KIRPearlylate = KIRPearlylate[KIRPearlylate['ProgStage']=='earlylate']
KIRCearlylate = KIRCearlylate[KIRCearlylate['ProgStage']=='earlylate']
```

## Plot the venn diagram of earlylate in KIRP and KIRC
```python
plt.figure(figsize=(20,16))
venn2(subsets=[set(KIRPearlylate.index), set(KIRCearlylate.index)], 
      set_labels=('KIRP', 'KIRC'),
      set_colors=["#236192", "#B33D26"])
plt.rcParams.update({'font.size': 30})
plt.savefig('Figure/KIRPKIRCearlylateProgVenn.png', dpi=300)
plt.close()
```
## Plot the Heatmap of earlylate in KIRP and KIRC intersect
```python
# Import data
KIRPlateExpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
KIRClateExpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
KIRPearlyExpr= pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRCearlyExpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
# Combine early and late stage
KIRPExpr = pd.concat([KIRPearlyExpr,KIRPlateExpr],axis=0)
KIRCExpr = pd.concat([KIRCearlyExpr,KIRClateExpr],axis=0)
# Select the intersect of earlylate in KIRP and KIRC
KIRPintergene= KIRCearlylate.index.intersection(KIRPearlylate.index)
# Select the intersect of earlylate in KIRP and KIRC
KIRPExpr_sub = KIRPExpr[KIRPintergene]
KIRCExpr_sub = KIRCExpr[KIRPintergene]
# add stage label and cancer type label
KIRPExpr_sub['E_L_Stage'] = KIRPExpr['E_L_Stage']
KIRCExpr_sub['E_L_Stage'] = KIRCExpr['E_L_Stage']
KIRPExpr_sub['CancerType'] = 'KIRP'
KIRCExpr_sub['CancerType'] = 'KIRC'
# Combine KIRP and KIRC
KIRPKIRCExpr_sub = pd.concat([KIRPExpr_sub,KIRCExpr_sub],axis=0)
# Extract gene name without E_L_Stage and CancerType
KIRPKIRCExpr_subdrop = KIRPKIRCExpr_sub.drop(['E_L_Stage','CancerType'],axis=1)
# Log2 transformation
KIRPKIRCExpr_subdrop = np.log10(KIRPKIRCExpr_subdrop+1)
# Plotting Heatmap
# Prepare a vector of color mapped to the 'E_L_Stage' column
my_palette = dict(zip(KIRPKIRCExpr_sub.CancerType.unique(), ["red","blue"]))
row_colors = KIRPKIRCExpr_sub.CancerType.map(my_palette)
plt.figure(figsize=(10,10))
sns.set(font_scale=1.2)
sns.clustermap(KIRPKIRCExpr_subdrop,metric="correlation", cmap="seismic", row_colors=row_colors,row_cluster=False,col_cluster=True,square=True,xticklabels=False,yticklabels=False,z_score=1)
plt.savefig('Figure/KIRPKIRCrow_clusterFalsecol_clusterTruegene_expression_heatmap.png')
