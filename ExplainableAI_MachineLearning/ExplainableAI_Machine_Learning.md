# ExplainableAI Machine Learning
Author: Huang,Shu-Jing
Date: 2023-11-28

- [ExplainableAI Machine Learning](#explainableai-machine-learning)
  - [environment](#environment)
  - [import package](#import-package)
- [Explainable AI - Explain the model of KIRC with SHAP](#explainable-ai---explain-the-model-of-kirc-with-shap)
  - [KIRC-LightGBM](#kirc-lightgbm)
  - [KIRC-RandomForest](#kirc-randomforest)
  - [KIRC-XGBoost](#kirc-xgboost)
  - [KIRP-SCV](#kirp-scv)
  - [KIRC-LogisticRegression](#kirc-logisticregression)
- [Explainable AI - Explain the model of KIRP with SHAP](#explainable-ai---explain-the-model-of-kirp-with-shap)
  - [KIRP-LightGBM](#kirp-lightgbm)
  - [KIRP-RandomForest](#kirp-randomforest)
  - [KIRP-XGBoost](#kirp-xgboost)
  - [KIRP-SVC](#kirp-svc)
  - [KIRP-LogisticRegression](#kirp-logisticregression)
- [Function`plot_dependence` - plot dependence plot for gene](#functionplot_dependence---plot-dependence-plot-for-gene)



## environment
```shell
conda activate SHAP
cd /home/emily2835/EarlyLateStageProject/ExplainableAI_MachineLearning
```

## import package
```python
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import sklearn
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
import xgboost as xg
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_sample_weight
import time
from sklearn.preprocessing import LabelEncoder
```

# Explainable AI - Explain the model of KIRC with SHAP 
## KIRC-LightGBM
* Import Model
```python
filename = 'MLModelFiles/KIRCLightGBMModel_gbm.py'
with open(filename) as file:
    exec(file.read())
```

* Explain Model
```python
# Explain Model
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(KIRC_GeneExprpre)
KIRC_feature_names = KIRC_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[0]
df = pd.DataFrame(save_shap, columns=KIRC_feature_names)
df.index = KIRC_Expr['sample']
df.to_csv('Output/KIRCLightgbmSHAPValue.csv')
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRC_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRCLightgbmSHAPMeanValue.csv', index=False)
```
* Summary Plot
```python
# summary plot for all features
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names, plot_type='bar',max_display=20)
plt.savefig('Figure/KIRCLightgbmSHAP.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for dot plot (alive)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[0], KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRCLightgbmSHAP_early.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for dot plot (dead)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRCLightgbmSHAP_late.png', dpi=300, bbox_inches='tight')
plt.close()
```
* Dependence Plot
```python
plot_dependence("LightGBM",'KIRC', 'FBXO48', KIRC_GeneExpr, KIRC_GeneExprpre, 'late')
plot_dependence("LightGBM",'KIRC', 'GPR108', KIRC_GeneExpr, KIRC_GeneExprpre, 'late')
plot_dependence("LightGBM",'KIRC', 'VCPIP1', KIRC_GeneExpr, KIRC_GeneExprpre, 'late')
```
* Force Plot
```python
# force plot for dead
plt.figure(figsize=(10, 8))
sample = pd.DataFrame([KIRC_GeneExprpre[1,:]], columns=KIRC_feature_names)
shap.force_plot(explainer.expected_value[1], shap_values[1][1,:],sample,matplotlib=True)
plt.savefig('Figure/KIRCLightgbmSHAPForceplotLatePatient1.svg', dpi=300, bbox_inches='tight')
plt.close()
```
---
## KIRC-RandomForest
* Import Model
```python
filename = 'MLModelFiles/KIRCRandomForestModel_rf.py'
with open(filename) as file:
    exec(file.read())
```
* Explain Model
```python
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(KIRC_GeneExprpre)
KIRC_feature_names = KIRC_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[1]
df = pd.DataFrame(save_shap, columns=KIRC_feature_names)
df.index = KIRC_Expr['sample']
df.to_csv('Output/KIRCRandomForestSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRC_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
# Rank the features by SHAP value
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRCRandomForestSHAPMeanValue.csv', index=False)
```
* Summary Plot
```python
# summary plot for dotplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRCRandomForestSHAP.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20,plot_type='bar')
plt.savefig('Figure/KIRCRandomForestSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
plt.close()
```
* Dependence Plot
```python
plot_dependence("RandomForest",'KIRC', 'FBXO48', KIRC_GeneExpr, KIRC_GeneExprpre, 'late')
plot_dependence("RandomForest",'KIRC', 'GPR108', KIRC_GeneExpr, KIRC_GeneExprpre, 'late')
```

---
## KIRC-XGBoost
* Import Model
```python
filename = 'MLModelFiles/KIRCXGBoostModel_xgb.py'
with open(filename) as file:
    exec(file.read())
```

* Explain Model
```python
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(KIRC_GeneExprpre)  #(528, 5299)
df = pd.DataFrame(shap_values, columns=KIRC_GeneExpr.columns)
df.index = KIRC_Expr['sample']\
# save shap value for all features
df.to_csv('Output/KIRCXGBOOSTSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(shap_values).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRC_GeneExpr.columns, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRCXGBOOSTSHAPMeanValue.csv', index=False)
```

* summary plot
```python
# summary plot for dotplot
KIRC_feature_names = KIRC_GeneExpr.columns
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRCXGBOOSTSHAP.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20,plot_type='bar')
plt.savefig('Figure/KIRCXGBOOSTSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
```
---
## KIRP-SCV
* Import Model
```python
filename = 'MLModelFiles/KIRCSVCModel_svc.py'
with open(filename) as file:
    exec(file.read())
```
Explain Model
```python
X_train_summary = shap.kmeans(KIRC_GeneExprpre, 100) # sample size dimension reduction
explainer = shap.KernelExplainer(svc.predict_proba, X_train_summary, link="logit")
start = time.time()
shap_values = explainer.shap_values(KIRC_GeneExprpre, l1_reg="num_features(500)")
end = time.time()
print('Time: ', end - start) # Time:  304862.29461216927
KIRC_feature_names = KIRC_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[1]
# as dataframe
df = pd.DataFrame(save_shap, columns=KIRC_feature_names)
df.to_csv('Output/KIRCSVCSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRC_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRCSVCSHAPMeanValue.csv', index=False)
```

Summary Plot
```python
# summary plot for dotplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20,plot_type='dot')
plt.savefig('Figure/KIRCSVCSHAP_test.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRCSVCSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
plt.close()
```
---
## KIRC-LogisticRegression
* Import Model
```python
filename = 'MLModelFiles/KIRCLogisticRegressionModel_lr.py'
with open(filename) as file:
    exec(file.read())
```
* Explain Model
```python
explainer = shap.LinearExplainer(logreg,KIRC_GeneExprpre, feature_dependence="independent")
shap_values = explainer.shap_values(KIRC_GeneExprpre)
KIRC_feature_names = KIRC_GeneExpr.columns
df = pd.DataFrame(shap_values, columns=KIRC_feature_names)
df.index = KIRC_Expr['sample']
# save shap value for all features
df.to_csv('Output/KIRCLogisticRegressionSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(shap_values).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRC_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRCLogisticRegressionSHAPMeanValue.csv', index=False)
```
* summary plot
```python
# summary plot for dotplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRCLogisticRegressionSHAP.png', dpi=300, bbox_inches='tight')
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20,plot_type='bar')
plt.savefig('Figure/KIRCLogisticRegressionSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
```

---



# Explainable AI - Explain the model of KIRP with SHAP 
## KIRP-LightGBM
* Import Model
```python
filename = 'MLModelFiles/KIRPLightGBMModel_gbm.py'
with open(filename) as file:
    exec(file.read())
```
* Explain Model
```python
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(KIRP_GeneExprpre)
KIRP_feature_names = KIRP_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[0]
df = pd.DataFrame(save_shap, columns=KIRP_feature_names)
df.index = KIRP_Expr['sample']
df.to_csv('Output/KIRPLightgbmSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRP_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
# Rank the features by SHAP value
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRPLightgbmSHAPMeanValue.csv', index=False)
```
* summary plot
```python
# summary plot for all features
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names, plot_type='bar',max_display=20)
plt.savefig('Figure/KIRPLightgbmSHAP.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for alive
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[0], KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRPLightgbmSHAP_early.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for dead
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRPLightgbmSHAP_late.png', dpi=300, bbox_inches='tight')
plt.close()
```

---
##  KIRP-RandomForest
* Import Model
```python
filename = 'MLModelFiles/KIRPRandomForestModel_rf.py'
with open(filename) as file:
    exec(file.read())
```
* Explain Model
```python
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(KIRP_GeneExprpre)
KIRP_feature_names = KIRP_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[1]
df = pd.DataFrame(save_shap, columns=KIRP_feature_names)
df.index = KIRP_Expr['sample']
df.to_csv('Output/KIRPRandomForestSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRP_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
# Rank the features by SHAP value
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRPRandomForestSHAPMeanValue.csv', index=False)
```

* summary plot
```python
# summary plot for dotplot
plt.figure(figsize=(10, 8))                       
shap.summary_plot(shap_values[1], KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRPRandomForestSHAP.png', dpi=300, bbox_inches='tight')
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20,plot_type='bar')
plt.savefig('Figure/KIRPRandomForestSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
```
---
## KIRP-XGBoost
* Import Model
```python
filename = 'MLModelFiles/KIRPXGBoostModel_xgb.py'
with open(filename) as file:
    exec(file.read())
```
* Explain Model
```python
explainer = shap.TreeExplainer(xgb)
shap_values = explainer.shap_values(KIRP_GeneExprpre)  #(528, 5299)
df = pd.DataFrame(shap_values, columns=KIRP_GeneExpr.columns)
df.index = KIRP_Expr['sample']
# save shap value for all features
df.to_csv('Output/KIRPXGBOOSTSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(shap_values).mean(axis=0)
# create a dataframe to store the mean of all features
KIRP_feature_names = KIRP_GeneExpr.columns
df = pd.DataFrame(list(zip(KIRP_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRPXGBOOSTSHAPMeanValue.csv', index=False)
```
* summary plot
```python
# summary plot for dotplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRPXGBOOSTSHAP.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20,plot_type='bar')
plt.savefig('Figure/KIRPXGBOOSTSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
plt.close()
```
---
## KIRP-SVC
* Import Model
```python
filename = 'MLModelFiles/KIRPSVCModel_svc.py'
with open(filename) as file:
    exec(file.read())
```
* Explain Model
```python
X_train_summary = shap.kmeans(KIRP_GeneExprpre, 100)
explainer = shap.KernelExplainer(svc.predict_proba, X_train_summary, link="logit")
start = time.time()
shap_values = explainer.shap_values(KIRP_GeneExprpre, l1_reg="num_features(500)")
end = time.time()
print('Time: ', end - start) #Time:  2539.343339920044
KIRP_feature_names = KIRP_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[1]
# as dataframe
df = pd.DataFrame(save_shap, columns=KIRP_feature_names)
df.to_csv('Output/KIRPSVCSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRP_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRPSVCSHAPMeanValue.csv', index=False)
```

* summary plot
```python
# summary plot for dotplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRPSVCSHAP.png', dpi=300, bbox_inches='tight')
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20, plot_type='bar')
plt.savefig('Figure/KIRPSVCSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
```


---
## KIRP-LogisticRegression
* Import Model
```python
filename = 'MLModelFiles/KIRPLogisticRegressionModel_lr.py'
with open(filename) as file:
    exec(file.read())
```
* Explain Model
```python
explainer = shap.LinearExplainer(lr,KIRP_GeneExprpre, feature_dependence="independent")
shap_values = explainer.shap_values(KIRP_GeneExprpre)
KIRP_feature_names = KIRP_GeneExpr.columns
df = pd.DataFrame(shap_values, columns=KIRP_feature_names)
df.index = KIRP_Expr['sample']
# save shap value for all features
df.to_csv('Output/KIRPLogisticRegressionSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(shap_values).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRP_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRPLogisticRegressionSHAPMeanValue.csv', index=False)
```
* summary plot
```python
# summary plot for dotplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRPLogisticRegressionSHAP.png', dpi=300, bbox_inches='tight')
# summary plot for barplot
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20, plot_type='bar')
plt.savefig('Figure/KIRPLogisticRegressionSHAP_BARPLOT.png', dpi=300, bbox_inches='tight')
```



# Function`plot_dependence` - plot dependence plot for gene
```python
def plot_dependence(modelname,cancertype, gene, GeneExpr, GeneExprpre,stage):
    gene_column = GeneExpr.columns[GeneExpr.columns.str.contains(gene)]
    gene_column = gene_column.tolist()[0]
    gene_num = GeneExpr.columns.get_loc(gene_column)
    plt.figure(figsize=(10, 8))
    if stage == 'early':
        shap.dependence_plot(gene_num, shap_values[0], GeneExprpre, interaction_index=None)
    elif stage == 'late':
        shap.dependence_plot(gene_num, shap_values[1], GeneExprpre, interaction_index=None)
    elif stage == 'None':
        shap.dependence_plot(gene_num, shap_values, GeneExprpre, interaction_index=None)
    plt.xlabel(gene_column, fontsize=10)
    plt.ylabel(f'SHAP value of {gene}', fontsize=10)
    plt.savefig(f'Figure/{cancertype}{modelname}SHAPlatedependenceplot{gene}.png', dpi=300, bbox_inches='tight')
    plt.close()
```