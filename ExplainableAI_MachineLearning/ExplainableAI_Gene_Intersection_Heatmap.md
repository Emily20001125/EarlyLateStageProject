# ExplainableAI Gene Intersection Heatmap Analysis
Author: Huang,Shu-Jing
Date: 2024-01-15

## environment
```shell
conda activate python3_10
cd /home/emily2835/EarlyLateStageProject/ExplainableAI_MachineLearning
```

# import package
```shell
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
## Gene Intersection Heatmap Analysis

```python
KIRPIntersect = pd.read_csv("Output/KIRPIntersectionGenelist.csv")
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# select the gene use regular expression for "ENSG"
KIRP_ExprGene = KIRP_Expr[KIRP_Expr.columns[KIRP_Expr.columns.str.contains('ENSG')]]
# check the number of early and late stage
KIRP_ExprGene.columns = GENEcompare[GENEcompare['id'].isin(KIRP_ExprGene.columns)]['gene'].values
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_ExprGene[KIRPIntersect['Gene symbol']]
# Add E_L_Stage
KIRP_GeneExpr['E_L_Stage'] = KIRP_Expr['E_L_Stage']
# Add sample name
KIRP_GeneExpr.index = KIRP_Expr['sample']
# plot heatmap
plottable = KIRP_GeneExpr.drop('E_L_Stage',axis=1)
# Normalize the data
log2plottable = np.log2(plottable+1)
sns.clustermap(log2plottable, col_cluster=True, row_cluster=False, cmap='RdBu_r', figsize=(10,10),z_score=1)
plt.savefig('Figure/KIRP_Gene_Intersection_Heatmap.png', dpi=300)
```
