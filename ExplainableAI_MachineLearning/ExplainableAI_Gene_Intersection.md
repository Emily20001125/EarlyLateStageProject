# Get the intersection of 5 models by SHAP value Rank
Author: Huang,Shu-Jing
Date: 2023-12-26

## environment
```shell
conda activate python3_10
cd /home/emily2835/EarlyLateStageProject/ExplainableAI_MachineLearning
```

## import package
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from upsetplot import plot, from_contents
```

## Get the intersection of 5 models SHAP value Rank
### [KIRC]
```python
# import data
KIRCSVC = pd.read_csv("Output/KIRC_SVC_SHAPMeanValue.csv")
KIRCRF = pd.read_csv("Output/KIRCRandomForestSHAPMeanValue.csv")
KIRCXGB = pd.read_csv("Output/KIRCXGBOOSTSHAPMeanValue.csv")
KIRCLGBM = pd.read_csv("Output/KIRCLIGHTGBMSHAPMeanValue.csv")
KIRCLG = pd.read_csv("Output/KIRCLogisticRegressionSHAPMeanValue.csv")

# Get the 300 genes with the highest SHAP value
# Drop duplicate gene symbol of KIRCLG
duplicate = KIRCLG[KIRCLG.duplicated(["Gene symbol"])]
KIRCLG[KIRCLG["Gene symbol"]=="ELFN2"]
KIRCLG = KIRCLG.drop_duplicates(["Gene symbol"])

# Get gene symbol
KIRCSVC300 = KIRCSVC["Gene symbol"][0:300]
KIRCRF300 = KIRCRF["Gene symbol"][0:300]
KIRCXGB300 = KIRCXGB["Gene symbol"][0:300]
KIRCLGBM300 = KIRCLGBM["Gene symbol"][0:300]
KIRCLG300= KIRCLG["Gene symbol"][0:300]

# Get the intersection of 5 models
KIRCSVC300 = set(KIRCSVC300)
KIRCRF300 = set(KIRCRF300)
KIRCXGB300 = set(KIRCXGB300)
KIRCLGBM300 = set(KIRCLGBM300)
KIRCLG300 = set(KIRCLG300)
KIRCintersection = KIRCSVC300.intersection(KIRCRF300,KIRCXGB300,KIRCLGBM300,KIRCLG300)
# Save csv
KIRCintersection = pd.DataFrame(KIRCintersection,columns=["Gene symbol"])
KIRCintersection.to_csv("Output/KIRCIntersectionGenelist.csv",index=False,header=True)
# Plot upset plot of 5 models
data = {
    'SVC': set(KIRCSVC["Gene symbol"][0:300]),
    'RF': set(KIRCRF["Gene symbol"][0:300]),
    'XGB': set(KIRCXGB["Gene symbol"][0:300]),
    'LGBM': set(KIRCLGBM["Gene symbol"][0:300]),
    'LR': set(KIRCLG["Gene symbol"][0:300])
}
df = from_contents(data)
plot(df,show_counts='%d',facecolor="#236192",sort_by='degree',min_degree=2)
plt.gca().yaxis.grid(False)
plt.gca().set_facecolor('white')
plt.savefig("Figure/KIRCUpsetPlot.png",dpi=300)
```

### [KIRP]
```python
# import data
KIRPRF = pd.read_csv("Output/KIRPRandomForestSHAPMeanValue.csv")
KIRPXGB = pd.read_csv("Output/KIRPXGBOOSTSHAPMeanValue.csv")
KIRPLGBM = pd.read_csv("Output/KIRPLIGHTGBMSHAPMeanValue.csv")
KIRPLG = pd.read_csv("Output/KIRPLogisticRegressionSHAPMeanValue.csv")
KIRPSVC = pd.read_csv("Output/KIRPSVCSHAPMeanValue.csv")
# Get the 300 genes with the highest SHAP value
KIRPRF300 = KIRPRF["Gene symbol"][0:300]
KIRPXGB300 = KIRPXGB["Gene symbol"][0:300]
KIRPLGBM300 = KIRPLGBM["Gene symbol"][0:300]
KIRPLG300= KIRPLG["Gene symbol"][0:300]
KIRPSVC300 = KIRPSVC["Gene symbol"][0:300]
# Get the intersection of 5 models
KIRPRF300 = set(KIRPRF300)
KIRPXGB300 = set(KIRPXGB300)
KIRPLGBM300 = set(KIRPLGBM300)
KIRPLG300 = set(KIRPLG300)
KIRPSVC300 = set(KIRPSVC300)
KIRPintersection = KIRPRF300.intersection(KIRPXGB300,KIRPLGBM300,KIRPLG300,KIRPSVC300)
# Save csv
KIRPintersection = pd.DataFrame(KIRPintersection,columns=["Gene symbol"])
KIRPintersection.to_csv("Output/KIRPIntersectionGenelist.csv",index=False,header=True)
# Plot upset plot of 5 models
data = {
    'SVC': set(KIRPSVC["Gene symbol"][0:300]),
    'RF': set(KIRPRF["Gene symbol"][0:300]),
    'XGB': set(KIRPXGB["Gene symbol"][0:300]),
    'LGBM': set(KIRPLGBM["Gene symbol"][0:300]),
    'LR': set(KIRPLG["Gene symbol"][0:300])
}
df = from_contents(data)
plot(df,show_counts='%d',facecolor="#236192",sort_by='degree',min_degree=2)
plt.gca().yaxis.grid(False)
plt.gca().set_facecolor('white')
plt.savefig("Figure/KIRPUpsetPlot.png",dpi=300)
```





# END